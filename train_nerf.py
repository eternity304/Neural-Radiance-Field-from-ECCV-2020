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

if __name__ == "__main__":
    device = "cpu"
    raw_train = NerfDataset("lego\\transforms_train.json", "lego")
    model = Nerf().to(device)
    model.load_state_dict(torch.load('166000EpochWt.pth',  map_location=torch.device(device)))

    batch_size = 256
    near = 1.8
    far = 4.5
    num_samples = 128
    num_epoch = 50000
    lr = 1e-5

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epoch,
        eta_min=lr * 0.1
    )
    criterion = torch.nn.MSELoss()
    overall_loss = []

    for epoch in tqdm(range(num_epoch), desc="epoch"):
        epoch_loss = 0
        num_batches = 0

        # Shuffle the dataset each epoch
        indices = torch.randperm(100 * batch_size)

        data = prepare_nerf_batch(raw_train, batch_size=batch_size, device=device)

        for i in tqdm(range(0, 100 * batch_size, batch_size), desc="batch", leave=False):
            # Get batch indices
            batch_idx = indices[i:i+batch_size]
            
            # Get batch data
            camera_pos = data['camera_pos'][batch_idx].to(device)
            ray_origin = data['ray_origin'][batch_idx].to(device)
            ray_direction = data['ray_direction'][batch_idx].to(device)
            gt_colour = data['gt_colour'][batch_idx, :3].to(device)
            N_rays = batch_size

            # Smaple distance along ray
            t_vals = torch.linspace(near, far, steps=num_samples, device=device)
            t_vals = t_vals.expand(N_rays, num_samples)

            # When sampling points along the ray
            sample_positions = ray_origin.unsqueeze(1) + ray_direction.unsqueeze(1) * t_vals.unsqueeze(-1)  # (N_rays, num_samples, 3)
            sample_positions_flat = sample_positions.reshape(-1, 3)  # (N_rays * num_samples, 3)

            # The viewing direction for each sample is just the normalized ray direction
            viewing_directions = ray_direction.unsqueeze(1).expand(-1, num_samples, -1)  # (N_rays, num_samples, 3)
            viewing_directions_flat = viewing_directions.reshape(-1, 3)  # (N_rays * num_samples, 3)

            # Forward pass through the model
            rgb_pred, sigma_pred = model(sample_positions_flat, viewing_directions_flat)
        
            rendered_colours = render_rays(rgb_pred, sigma_pred, N_rays, t_vals, viewing_directions, num_samples, near, far, device)

            # Add total variation regularization
            tv_loss = torch.mean(torch.abs(rgb_pred[..., :, 1:] - rgb_pred[..., :, :-1])) + \
                    torch.mean(torch.abs(rgb_pred[..., 1:, :] - rgb_pred[..., :-1, :]))

            # Backpropagation
            loss = criterion(rendered_colours, gt_colour) + 0.1 * tv_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add gradient clipping
            optimizer.step()

            # print(f"Epoch {epoch+1}/{num_epoch}, Loss: {loss.item():.6f}")
            epoch_loss += loss.item()
            num_batches += 1
        
        overall_loss.append(epoch_loss)
        
        # Adjust learning rate more gradually
        if (epoch + 1) % 30 == 0:
            scheduler.step()
        
        if (epoch + 1) % 2000 == 0:
            torch.save(model.state_dict(), f"{166000 + epoch + 1 }EpochWt.pth")

        if epoch != 0 and (epoch + 1) % 10 == 0:
            plt.figure(figsize=(20,8))
            plt.plot(torch.tensor(overall_loss).numpy())
            plt.title(f"Epoch {epoch+1} Loss (Average: {epoch_loss/num_batches:.6f})")
            plt.grid()
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.tight_layout()
            plt.gca().spines['left'].set_visible(True)
            plt.savefig("Loss.jpg")
