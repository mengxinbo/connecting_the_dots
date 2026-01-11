import torch
import torch.utils.data
import numpy as np
from pathlib import Path
import torchext


class PolarizationDataset(torchext.BaseDataset):
    '''
    Dataset for Polarization Camera Data
    
    Loads 4 polarization direction images and computes various polarization representations.
    Supports multiple input modes: raw, stokes, dolp_aolp, full
    '''
    def __init__(self, data_root, sample_paths, input_type='stokes', train=True):
        """
        Args:
            data_root: Root directory containing polarization data
            sample_paths: List of sample directory paths
            input_type: Type of polarization input
                - 'raw': 4 polarization directions [4, H, W]
                - 'stokes': Stokes parameters [3, H, W] (S0, S1, S2)
                - 'dolp_aolp': S0 + DoLP + AoLP [3, H, W]
                - 'full': All information [6, H, W] (S0, S1, S2, DoLP, AoLP, intensity)
            train: Whether this is training or validation set
        """
        super().__init__(train=train)
        
        self.data_root = Path(data_root)
        self.sample_paths = sample_paths
        self.input_type = input_type
        self.train = train
        
        # Validate input type
        valid_types = ['raw', 'stokes', 'dolp_aolp', 'full']
        if input_type not in valid_types:
            raise ValueError(f"input_type must be one of {valid_types}, got {input_type}")
    
    def __len__(self):
        return len(self.sample_paths)
    
    def _compute_stokes(self, I0, I45, I90, I135):
        """
        Compute Stokes parameters from 4 polarization directions
        
        Args:
            I0, I45, I90, I135: Polarization images at 0, 45, 90, 135 degrees
            
        Returns:
            S0, S1, S2: Stokes parameters
        """
        # S0: Total intensity
        S0 = (I0 + I45 + I90 + I135) / 2.0
        
        # S1: Linear polarization (0°-90°)
        S1 = I0 - I90
        
        # S2: Linear polarization (45°-135°)
        S2 = I45 - I135
        
        return S0, S1, S2
    
    def _compute_dolp_aolp(self, S0, S1, S2):
        """
        Compute Degree of Linear Polarization (DoLP) and Angle of Linear Polarization (AoLP)
        
        Args:
            S0, S1, S2: Stokes parameters
            
        Returns:
            DoLP, AoLP: Degree and angle of linear polarization
        """
        # DoLP: Degree of Linear Polarization
        DoLP = np.sqrt(S1**2 + S2**2) / (S0 + 1e-8)
        
        # AoLP: Angle of Linear Polarization
        AoLP = 0.5 * np.arctan2(S2, S1)
        
        return DoLP, AoLP
    
    def __getitem__(self, idx):
        if not self.train:
            rng = self.get_rng(idx)
        else:
            rng = np.random.RandomState()
            
        sample_path = self.sample_paths[idx]
        
        ret = {}
        ret['id'] = idx
        
        # Load 4 polarization direction images
        I0 = np.load(sample_path / 'pol_0.npy').astype(np.float32)
        I45 = np.load(sample_path / 'pol_45.npy').astype(np.float32)
        I90 = np.load(sample_path / 'pol_90.npy').astype(np.float32)
        I135 = np.load(sample_path / 'pol_135.npy').astype(np.float32)
        
        # Compute Stokes parameters
        S0, S1, S2 = self._compute_stokes(I0, I45, I90, I135)
        
        # Compute DoLP and AoLP
        DoLP, AoLP = self._compute_dolp_aolp(S0, S1, S2)
        
        # Prepare input based on input_type
        if self.input_type == 'raw':
            # Stack 4 polarization directions
            pol_input = np.stack([I0, I45, I90, I135], axis=0)  # [4, H, W]
        elif self.input_type == 'stokes':
            # Stack Stokes parameters
            pol_input = np.stack([S0, S1, S2], axis=0)  # [3, H, W]
        elif self.input_type == 'dolp_aolp':
            # Stack S0, DoLP, AoLP
            pol_input = np.stack([S0, DoLP, AoLP], axis=0)  # [3, H, W]
        elif self.input_type == 'full':
            # Stack all information
            intensity = (I0 + I45 + I90 + I135) / 4.0
            pol_input = np.stack([S0, S1, S2, DoLP, AoLP, intensity], axis=0)  # [6, H, W]
        
        ret['pol_input'] = pol_input
        
        # Load depth ground truth if available
        depth_path = sample_path / 'depth.npy'
        if depth_path.exists():
            depth = np.load(depth_path).astype(np.float32)
            ret['depth'] = depth[None]  # [1, H, W]
        
        # Load normal ground truth if available
        normal_path = sample_path / 'normal.npy'
        if normal_path.exists():
            normal = np.load(normal_path).astype(np.float32)
            # Assume normal is [H, W, 3], convert to [3, H, W]
            if len(normal.shape) == 3 and normal.shape[2] == 3:
                normal = np.transpose(normal, (2, 0, 1))
            ret['normal'] = normal  # [3, H, W]
        
        return ret
    
    def get_num_channels(self):
        """Return the number of input channels based on input_type"""
        channel_map = {
            'raw': 4,
            'stokes': 3,
            'dolp_aolp': 3,
            'full': 6
        }
        return channel_map[self.input_type]


if __name__ == '__main__':
    pass
