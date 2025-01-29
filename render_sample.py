import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json 
from PIL import Image
import os
import math

def cartesian_to_spherical(x, y, z):
        # Compute the radial distance
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Compute the polar angle (θ), ensuring r is not zero to avoid division errors
        theta = np.arccos(np.clip(z / r, -1.0, 1.0))  # Clipping to avoid precision issues
        
        # Compute the azimuthal angle (φ), handles quadrant correctly
        phi = np.arctan2(y, x)  # Gives values in range [-pi, pi]
        
        # Ensure phi is in range [0, 2π]
        phi = float(np.where(phi < 0, phi + 2 * np.pi, phi))
        
        return r, np.degrees(theta), np.degrees(phi)

class NerfDataset:
    def __init__(self, dataPath, imagePath, imageType=".png"):
        with open(dataPath, "r") as f:
            meta = json.load(f)
        self.FOV = meta['camera_angle_x']
        self.items = []

        for frame in tqdm(meta['frames']):
            filepath = os.path.join(imagePath, frame['file_path']) + imageType
            transform = np.array(frame['transform_matrix'], dtype=np.float32)
            
            img = Image.open(filepath)
            img = np.array(img)
            H, W, _ = img.shape 
            camera = transform[:3, -1]

            fx = (W * 0.5) / np.tan(self.FOV * 0.5)
            fy = (H * 0.5) / np.tan(self.FOV * 0.5)
            r, theta, phi = cartesian_to_spherical(*camera)

            self.items.append({
                "image" : img,
                "pose" : torch.from_numpy(transform),
                "camera" : [r, theta, phi],
                "W" : W,
                "H" : H,
                "fx" : fx,
                "fy" : fy
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def data2input(data):
    out = []

    for entry in tqdm(data):
        transform = entry['pose'].float()
        ray_origin = transform[:3,3]

        W, H = entry['W'], entry['H']
        fx, fy = entry['fx'], entry['fy']
        cx, cy = W / 2, H / 2 # point for image center
        polar = entry['camera'][1]
        azimuthal = entry['camera'][2]

        # convert pixel into camera coordinate
        for _ in range(4000):
            i = torch.randint(0, H, (1,)).item()
            j = torch.randint(0, W, (1,)).item()
            colour = entry['image'][i, j]
            x, y = (i-cx) / fx, (j-cy) / fy
            ray_dir = transform @ torch.tensor([x, y, -1.0, 0.0], dtype=torch.float32)
            ray_dir = F.normalize(ray_dir[:3], p=2, dim=0)
            out.append((
                ray_origin,
                ray_dir,
                colour,
                polar,
                azimuthal                    
            ))
    
    return out

class Nerf(nn.Module):
    def __init__(self, hidden_dim=256, emb_pos=10, emb_dir=4, num_layers=8, skip_layer=4):
        super(Nerf, self).__init__()
        
        self.emb_pos = emb_pos
        self.emb_dir = emb_dir
        self.num_layers = num_layers
        self.skip_layer = skip_layer
        
        # Positional encoding dimensions
        pos_dim = emb_pos * 6 + 3
        dir_dim = emb_dir * 6 + 3
        
        # Shared backbone for density
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(pos_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        
        for i in range(1, num_layers):
            if i == skip_layer:
                self.layers.append(nn.Linear(hidden_dim + pos_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        
        # Output layer for density (sigma)
        self.sigma_layer = nn.Linear(hidden_dim, 1)
        
        # Feature layer after the backbone
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        
        # Color network
        self.color_layers = nn.ModuleList()
        self.color_layers.append(nn.Linear(hidden_dim + dir_dim, hidden_dim // 2))
        self.color_layers.append(nn.ReLU())
        self.color_layers.append(nn.Linear(hidden_dim // 2, 3))  # RGB output
        
    def forward(self, x, d):
        # Positional encoding
        emb_x = self.positional_encoding(x, self.emb_pos)  # (N, pos_dim)
        emb_d = self.positional_encoding(d, self.emb_dir)  # (N, dir_dim)
        
        # Shared backbone
        h = emb_x
        for i in range(0, len(self.layers), 2):
            linear = self.layers[i]
            relu = self.layers[i+1]
            if (i // 2) == self.skip_layer:
                h = torch.cat([h, emb_x], dim=-1)
            h = linear(h)
            h = relu(h)
        
        # Density (sigma)
        sigma = F.relu(self.sigma_layer(h))  # (N, 1)
        
        # Features for color
        features = self.feature_layer(h)  # (N, hidden_dim)
        
        # Color network
        color_input = torch.cat([features, emb_d], dim=-1)  # (N, hidden_dim + dir_dim)
        color = self.color_layers[0](color_input)
        color = self.color_layers[1](color)
        color = torch.sigmoid(self.color_layers[2](color))  # (N, 3)
        
        return color, sigma

    @staticmethod
    def positional_encoding(x, L):
        """
        Apply positional encoding to input tensor x.
        Args:
            x: Input tensor of shape (..., 3)
            L: Number of frequency bands
        Returns:
            Encoded tensor of shape (..., 3 + 6*L)
        """
        out = [x]
        freq_bands = 2.0 ** torch.linspace(0., L-1, L, device=x.device)
        for freq in freq_bands:
            out.append(torch.sin(math.pi * freq * x))
            out.append(torch.cos(math.pi * freq * x))
        return torch.cat(out, dim=-1)
        
    

def prepare_nerf_batch(dataset, batch_size, device='cpu'):
    '''
    Args:
        dataset (list): list of dictionaries containing dataset entries
        batch_size (int): number of ray per batch
        num_samples (int): number of points to be sampled along each ray
        near (float): near bound for rendering
        far (float): far bound for rendering
        device (str): device to store tensors
    Returns:
        dict: Dictionary containing batched ray origins, direction, sample positions, viewing direction, and ground-truth colors
    '''
    ray_origin_list, ray_direction_list, gt_colour_list, t_vals_list, camera_pos_list = list(), list(), list(), list(), list()

    # Randomly Select Images
    selected_entries = torch.randint(0, len(dataset), (len(dataset),))

    for idx in selected_entries:
        entry = dataset[idx]

        # Camera parameter
        pose = entry['pose'].to(device) # homogeneous transformation
        fx, fy = entry['fx'], entry['fy']
        cx, cy, = entry['W'] / 2, entry['H'] / 2

        # Get Normalized image
        image = torch.tensor(entry['image'], dtype=torch.float32, device=device) / 255.0
        H, W = entry['H'], entry['W']

        # Sample pixels
        px = torch.randint(0, W, (batch_size,), device=device)
        py = torch.randint(0, H, (batch_size,), device=device)

        # Get gt colour
        gt_colour = image[px, py, :]
        gt_colour_list.append(gt_colour)

        # Pixel to camera space
        x = (px.float() - cx) / fx
        y = (py.float() - cy) / fy
        z = -torch.ones_like(x)

        # Create normalize ray direction in camera sapce
        ray_dir = torch.stack([x, y, z], dim=1)

        # Transform to world coordinate
        R = pose[:3, :3] # rotation
        t = pose[:3, 3]
        camera_pos = torch.tensor([entry['camera'][1], entry['camera'][2]])  # translation
        ray_direction = F.normalize(ray_dir @ R.T, p=2, dim=1)
        ray_origin = t.expand_as(ray_direction)
        camera_origin = camera_pos.expand(batch_size, 2)

        # Append to list
        ray_origin_list.append(ray_origin)
        ray_direction_list.append(ray_direction)
        camera_pos_list.append(camera_origin)

    # Stack all ray origin and direction along batchsize
    ray_origin = torch.cat(ray_origin_list, dim=0).to(device)
    ray_direction = torch.cat(ray_direction_list, dim=0)
    gt_colour = torch.cat(gt_colour_list, dim=0)
    camera_pos = torch.cat(camera_pos_list, dim=0)

    # print(camera_pos.shape)
    # print(ray_origin.shape)
    # print(ray_direction.shape)
    # print(gt_colour.shape)

    return {
        'camera_pos' : camera_pos.float(),         
        'ray_origin' : ray_origin.float(),         
        'ray_direction' : ray_direction.float(),
        'gt_colour' : gt_colour.float()      
    }

def sample_stratified(t_vals, n_rays, n_samples, perturb=True, device='cpu'):
    # Get intervals between samples
    mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
    upper = torch.cat([mids, t_vals[..., -1:]], -1)
    lower = torch.cat([t_vals[..., :1], mids], -1)
    
    # Stratified samples in those intervals
    t_rand = torch.rand(n_rays, n_samples).to(device) if perturb else 0.5
    t_vals = lower + (upper - lower) * t_rand
    return t_vals

def render_rays(rgb, sigma, N_rays, t_vals, viewing_directions, num_samples, near, far, device):
    '''
    Args:
        rgb: tensor of shape (N_rays * num_samples, 3)
        sigma: tensor of shape (N_rays * num_samples, 1)
        N_rays: number of rays
        t_vals: tensor of shape (N_rays, num_samples)
        viewing_directions: tensor of shape (N_rays, num_samples, 3)
        num_samples: number of samples per ray
        near: near plane distance
        far: far plane distance
    Return:
        pixel_colours: Tensor of shape (N_rays, 3)
    '''
    # Reshape RGB and sigma to include sample dimension
    rgb = rgb.view(N_rays, num_samples, 3)
    sigma = sigma.view(N_rays, num_samples)

    # Sample points along the ray
    t_vals = sample_stratified(t_vals, N_rays, num_samples, perturb=True, device=device)

    # Compute distances between adjacent samples
    delta = t_vals[..., 1:] - t_vals[..., :-1]
    delta = torch.cat([delta, torch.ones_like(delta[..., :1]) * 1e10], dim=-1)
    
    # Compute ray lengths - use the original viewing directions
    # We only need one direction per ray, so take the first sample's direction
    ray_directions = viewing_directions[:, 0, :]  # Shape: (N_rays, 3)
    ray_lengths = torch.norm(ray_directions, dim=-1, keepdim=True)  # Shape: (N_rays, 1)
    
    # Scale delta by ray length
    delta = delta * ray_lengths  # Will broadcast correctly
    
    # Compute alpha (opacity)
    alpha = 1 - torch.exp(-sigma * delta)
    
    # Compute weights for volume rendering
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), (1 - alpha + 1e-10)], dim=-1),
        dim=-1)[:, :-1]
    
    # Compute final colors
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # Shape: (N_rays, 3)
    
    return rgb_map

def render_image_spherical(model, theta, phi, radius=4.3, H=800, W=800, fx=1111.11103, fy=1111.11103, 
                           near=1.8, far=4.5, num_samples=64, batch_size=1024, device='cpu'):
    """
    Renders an image using the trained NeRF model from a given spherical position in batches.

    Args:
        model: Trained NeRF model.
        theta (float): Azimuthal angle (horizontal rotation) in degrees.
        phi (float): Elevation angle (vertical rotation) in degrees.
        radius (float): Distance of the camera from the origin (default=4.3).
        H (int): Image height (default=800).
        W (int): Image width (default=800).
        fx (float): Focal length in x (default=1111.11103).
        fy (float): Focal length in y (default=1111.11103).
        near (float): Near plane distance.
        far (float): Far plane distance.
        num_samples (int): Number of samples per ray.
        batch_size (int): Number of rays to process at once to avoid memory issues.
        device (str): 'cpu' or 'cuda' to run the computations.

    Returns:
        None (Displays the rendered image)
    """
    
    torch.cuda.empty_cache()
    model.to(device)

    # Convert degrees to radians
    theta = np.radians(theta)
    phi = np.radians(phi)

    # Convert spherical coordinates to Cartesian coordinates (Camera position)
    cam_x = radius * np.sin(theta) * np.cos(phi)
    cam_y = radius * np.sin(theta) * np.sin(phi)
    cam_z = radius * np.cos(theta)

    camera_position = torch.tensor([cam_x, cam_y, cam_z], dtype=torch.float32, device=device)

    # Look at the origin (assuming target at the origin)
    target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    # Compute camera coordinate system
    z_axis = F.normalize(camera_position - target, dim=0)
    x_axis = F.normalize(torch.cross(up, z_axis), dim=0)
    y_axis = torch.cross(z_axis, x_axis)

    # Construct 4x4 camera pose matrix
    pose = torch.eye(4, dtype=torch.float32, device=device)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = camera_position

    # Generate ray directions for each pixel in the image plane
    i, j = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
    i, j = i.flatten().float(), j.flatten().float()

    # Convert pixel coordinates to normalized camera coordinates
    x = (i - W / 2) / fx
    y = (j - H / 2) / fy
    z = -torch.ones_like(x)  # Forward direction is negative z

    ray_directions = torch.stack([x, y, z], dim=-1)
    ray_directions = F.normalize(ray_directions, dim=-1)

    # Transform ray directions to world coordinates
    ray_directions = ray_directions @ pose[:3, :3].T
    ray_origins = pose[:3, 3].expand(ray_directions.shape)

    # Prepare output tensor
    rgb_image = torch.zeros(H * W, 3, device=device)

    # Process rays in batches to avoid OOM
    for start in tqdm(range(0, H * W, batch_size), desc="Rendering", leave=True):
        end = min(start + batch_size, H * W)

        # Get batch of rays
        ray_origins_batch = ray_origins[start:end]
        ray_directions_batch = ray_directions[start:end]

        # Sample distances along rays
        t_vals = torch.linspace(near, far, steps=num_samples, device=device).expand(ray_directions_batch.shape[0], num_samples)

        # Stratified sampling with jittering
        t_vals = sample_stratified(t_vals, ray_directions_batch.shape[0], num_samples, perturb=True, device=device)

        # Compute sample positions in world space
        sample_positions = ray_origins_batch.unsqueeze(1) + ray_directions_batch.unsqueeze(1) * t_vals.unsqueeze(-1)
        sample_positions_flat = sample_positions.reshape(-1, 3)
        viewing_directions_flat = ray_directions_batch.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 3)

        # Query NeRF model for colors and densities
        with torch.no_grad():
            rgb_pred, sigma_pred = model(sample_positions_flat, viewing_directions_flat)

        # Render final colors using volume rendering equation
        rendered_colours = render_rays(rgb_pred, sigma_pred, ray_directions_batch.shape[0], t_vals, 
                                       ray_directions_batch.unsqueeze(1), num_samples, near, far, device=device)

        # Store the rendered colors
        rgb_image[start:end] = rendered_colours

        # Clear memory after processing each batch
        del sample_positions, viewing_directions_flat, rgb_pred, sigma_pred, rendered_colours
        torch.cuda.empty_cache()

    # Reshape predicted colors to image dimensions
    rgb_image = torch.rot90(rgb_image.view(H, W, 3), k=-1, dims=(0, 1))
    rgb_image = rgb_image.cpu().numpy()

    # Display the rendered image
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.title(f'Rendered Image - θ: {np.degrees(theta):.2f}, φ: {np.degrees(phi):.2f}, radius: {radius:.2f}')
    plt.savefig(f"60/rendered_image_{np.degrees(theta):.1f}_{np.degrees(phi):.1f}.jpg")
    # plt.show()

if __name__ == "__main__":
    device = "cpu"
    model = Nerf().to(device)
    model.load_state_dict(torch.load('166000EpochWt.pth', map_location=torch.device(device)))

    # for i in tqdm(range(20, 90, 20), leave=False):
    for j in tqdm(range(0, 360, 10), leave=True):
        render_image_spherical(model, 
                            theta=60, phi=j, radius=4.031600318,
                            H=800, W=800,
                            fx=1111.11103, fy=1111.11103, 
                            near=1.8, far=4.5,
                            num_samples=128,
                            batch_size=2048,
                            device=device)