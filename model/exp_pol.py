import torch
import numpy as np
import time
from pathlib import Path
import logging
import sys
import itertools
import json
import matplotlib.pyplot as plt
import co
import torchext
from model import pol_networks
from data import pol_dataset


class PolarizationWorker(torchext.Worker):
    '''
    Worker for training polarization camera depth estimation
    '''
    def __init__(self, args, num_workers=18, train_batch_size=8, test_batch_size=8, save_frequency=1, **kwargs):
        super().__init__(args.output_dir, args.exp_name, epochs=args.epochs, 
                         num_workers=num_workers, train_batch_size=train_batch_size, 
                         test_batch_size=test_batch_size, save_frequency=save_frequency, **kwargs)
        
        self.input_type = args.input_type
        self.dp_weight = args.dp_weight
        self.normal_weight = args.normal_weight
        self.ms = args.ms
        
        # Image sizes at different scales
        self.imsizes = [(480, 640)]
        for iter in range(3):
            self.imsizes.append((int(self.imsizes[-1][0]/2), int(self.imsizes[-1][1]/2)))
        
        # Load data paths from config
        with open('config.json') as fp:
            config = json.load(fp)
            data_root = Path(config.get('POL_DATA_ROOT', config.get('DATA_ROOT', 'data/polarization')))
        
        self.data_root = data_root
        sample_paths = sorted(data_root.glob('0*/'))
        
        # Split into train and test
        self.train_paths = sample_paths[256:]  # Use most for training
        self.test_paths = sample_paths[:256]    # Use first 256 for testing
        
        # Loss modules
        self.disparity_loss = pol_networks.DisparityLoss() if hasattr(pol_networks, 'DisparityLoss') else None
        if self.disparity_loss is None:
            # Import from base networks if not in pol_networks
            from model.networks import DisparityLoss
            self.disparity_loss = DisparityLoss()
        
        self.edge_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.1]).to(self.train_device))
        
        # Evaluation mask (similar to original code)
        self.eval_mask = np.zeros(self.imsizes[0])
        self.eval_mask[13:self.imsizes[0][0]-13, 140:self.imsizes[0][1]-13] = 1
        self.eval_mask = self.eval_mask.astype(np.bool_)
        self.eval_h = self.imsizes[0][0] - 2*13
        self.eval_w = self.imsizes[0][1] - 13 - 140
    
    def get_train_set(self):
        train_set = pol_dataset.PolarizationDataset(
            self.data_root, 
            self.train_paths, 
            input_type=self.input_type,
            train=True
        )
        return train_set
    
    def get_test_sets(self):
        test_sets = torchext.TestSets()
        test_set = pol_dataset.PolarizationDataset(
            self.data_root,
            self.test_paths,
            input_type=self.input_type,
            train=False
        )
        test_sets.append('simple', test_set, test_frequency=1)
        return test_sets
    
    def copy_data(self, data, device, requires_grad, train):
        self.data = {}
        for key, val in data.items():
            grad = 'pol_input' in key and requires_grad
            self.data[key] = val.to(device).requires_grad_(requires_grad=grad)
    
    def net_forward(self, net, train):
        out = net(self.data['pol_input'])
        return out
    
    def loss_forward(self, out, train):
        # Handle different network outputs
        if len(out) == 2:
            # Only disp and edge (PolarizationDispEdgeDecoders)
            disp_out, edge_out = out
            normal_out = None
        elif len(out) == 3:
            # Disp, edge, and normal (PolarizationMultiTaskNet)
            disp_out, edge_out, normal_out = out
        else:
            raise ValueError(f"Unexpected network output length: {len(out)}")
        
        # Ensure outputs are lists for multi-scale
        if not(isinstance(disp_out, tuple) or isinstance(disp_out, list)):
            disp_out = [disp_out]
        if not(isinstance(edge_out, tuple) or isinstance(edge_out, list)):
            edge_out = [edge_out]
        if normal_out is not None and not(isinstance(normal_out, tuple) or isinstance(normal_out, list)):
            normal_out = [normal_out]
        
        vals = []
        
        # Depth/Disparity loss (L1 loss if ground truth available)
        if 'depth' in self.data:
            depth_gt = self.data['depth']
            for s, disp_pred in enumerate(disp_out):
                # Simple L1 loss between predicted and ground truth disparity
                if s == 0:
                    # Only compute loss at full resolution
                    loss_val = torch.mean(torch.abs(disp_pred - depth_gt))
                    vals.append(loss_val)
        
        # Disparity smoothness loss
        if self.dp_weight > 0:
            # Get edge prediction for disparity regularization
            edge0 = 1 - torch.sigmoid(edge_out[0])
            val = self.disparity_loss(disp_out[0], edge0)
            vals.append(val * self.dp_weight)
        
        # Edge loss (if we have ground truth edges)
        # For now, we don't have edge ground truth in polarization data
        # This could be computed from depth gradients if needed
        
        # Normal loss (if ground truth normals available)
        if normal_out is not None and 'normal' in self.data and self.normal_weight > 0:
            normal_gt = self.data['normal']
            for s, normal_pred in enumerate(normal_out):
                if s == 0:
                    # Cosine similarity loss (1 - dot product)
                    # Both normals should be normalized
                    cos_sim = torch.sum(normal_pred * normal_gt, dim=1, keepdim=True)
                    normal_loss = torch.mean(1.0 - cos_sim)
                    vals.append(normal_loss * self.normal_weight)
        
        # Store outputs for visualization
        self.disp_pred = disp_out[0].detach()
        if edge_out is not None:
            self.edge_pred = torch.sigmoid(edge_out[0]).detach()
        if normal_out is not None:
            self.normal_pred = normal_out[0].detach()
        
        return vals
    
    def numpy_in_out(self, output):
        # Handle different network outputs
        if len(output) == 2:
            disp_out, edge_out = output
        elif len(output) == 3:
            disp_out, edge_out, normal_out = output
        else:
            disp_out = output
        
        if not(isinstance(disp_out, tuple) or isinstance(disp_out, list)):
            disp_out = [disp_out]
        
        es = disp_out[0].detach().to('cpu').numpy()
        
        # Get ground truth if available
        if 'depth' in self.data:
            gt = self.data['depth'].to('cpu').numpy().astype(np.float32)
        else:
            gt = np.zeros_like(es)
        
        # Get input image (use first channel of polarization input)
        im = self.data['pol_input'][:, 0:1, ...].detach().to('cpu').numpy()
        
        ma = gt > 0
        return es, gt, im, ma
    
    def write_img(self, out_path, es, gt, im, ma):
        logging.info(f'write img {out_path}')
        
        diff = np.abs(es - gt)
        
        vmin, vmax = np.nanmin(gt), np.nanmax(gt)
        if vmax > vmin:
            vmin = vmin - 0.2 * (vmax - vmin)
            vmax = vmax + 0.2 * (vmax - vmin)
        
        fig = plt.figure(figsize=(16, 12))
        es_ = co.cmap.color_depth_map(es, scale=vmax) if vmax > 0 else es
        gt_ = co.cmap.color_depth_map(gt, scale=vmax) if vmax > 0 else gt
        diff_ = co.cmap.color_error_image(diff, BGR=True)
        
        # Plot disparities/depths
        ax = plt.subplot(2, 3, 1)
        plt.imshow(es_[..., [2, 1, 0]])
        plt.xticks([])
        plt.yticks([])
        ax.set_title(f'Depth Est. {es.min():.4f}/{es.max():.4f}')
        
        ax = plt.subplot(2, 3, 2)
        plt.imshow(gt_[..., [2, 1, 0]])
        plt.xticks([])
        plt.yticks([])
        ax.set_title(f'Depth GT {np.nanmin(gt):.4f}/{np.nanmax(gt):.4f}')
        
        ax = plt.subplot(2, 3, 3)
        plt.imshow(diff_[..., [2, 1, 0]])
        plt.xticks([])
        plt.yticks([])
        ax.set_title(f'Depth Err. {diff.mean():.5f}')
        
        # Plot input and edge
        ax = plt.subplot(2, 3, 4)
        plt.imshow(im, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        ax.set_title(f'Polarization Input')
        
        if hasattr(self, 'edge_pred'):
            edge = self.edge_pred.to('cpu').numpy()[0, 0]
            ax = plt.subplot(2, 3, 5)
            plt.imshow(edge, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            ax.set_title(f'Edge Prediction')
        
        # Plot normal if available
        if hasattr(self, 'normal_pred'):
            normal = self.normal_pred.to('cpu').numpy()[0]
            # Convert normal from [-1,1] to [0,1] for visualization
            normal_vis = (normal + 1.0) / 2.0
            normal_vis = np.transpose(normal_vis, (1, 2, 0))
            ax = plt.subplot(2, 3, 6)
            plt.imshow(normal_vis)
            plt.xticks([])
            plt.yticks([])
            ax.set_title(f'Normal Prediction')
        
        plt.tight_layout()
        plt.savefig(str(out_path))
        plt.close(fig)
    
    def callback_train_post_backward(self, net, errs, output, epoch, batch_idx, masks=[]):
        if batch_idx % 512 == 0:
            out_path = self.exp_out_root / f'train_{epoch:03d}_{batch_idx:04d}.png'
            es, gt, im, ma = self.numpy_in_out(output)
            self.write_img(out_path, es[0, 0], gt[0, 0], im[0, 0], ma[0, 0])
    
    def callback_test_start(self, epoch, set_idx):
        self.metric = co.metric.MultipleMetric(
            co.metric.DistanceMetric(vec_length=1),
            co.metric.OutlierFractionMetric(vec_length=1, thresholds=[0.1, 0.5, 1, 2, 5])
        )
    
    def callback_test_add(self, epoch, set_idx, batch_idx, n_batches, output, masks=[]):
        es, gt, im, ma = self.numpy_in_out(output)
        
        if batch_idx % 8 == 0:
            out_path = self.exp_out_root / f'test_{epoch:03d}_{batch_idx:04d}.png'
            self.write_img(out_path, es[0, 0], gt[0, 0], im[0, 0], ma[0, 0])
        
        es = es.reshape(-1, 1)
        gt = gt.reshape(-1, 1)
        ma = ma.ravel()
        self.metric.add(es, gt, ma)
    
    def callback_test_stop(self, epoch, set_idx, loss):
        logging.info(f'{self.metric}')
        for k, v in self.metric.items():
            self.metric_add_test(epoch, set_idx, k, v)


if __name__ == '__main__':
    pass
