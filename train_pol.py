import os
import argparse
import torch
from model import exp_pol
from model import pol_networks
from co.utils import str2bool


def parse_args():
    """Parse command line arguments for polarization camera training"""
    parser = argparse.ArgumentParser(description='Train polarization camera depth estimation')
    
    # Output and experiment settings
    parser.add_argument('--output_dir',
                        help='Output directory',
                        default='./output', type=str)
    parser.add_argument('--exp_name',
                        help='Experiment name',
                        default='exp_pol', type=str)
    
    # Training command
    parser.add_argument('--cmd',
                        help='Training command',
                        default='resume', choices=['retrain', 'resume', 'retest'], type=str)
    parser.add_argument('--epoch',
                        help='If larger than -1, retest on the specified epoch',
                        default=-1, type=int)
    parser.add_argument('--epochs',
                        help='Training epochs',
                        default=100, type=int)
    
    # Polarization-specific settings
    parser.add_argument('--input_type',
                        help='Polarization input type',
                        default='stokes', 
                        choices=['raw', 'stokes', 'dolp_aolp', 'full'], 
                        type=str)
    
    # Loss weights
    parser.add_argument('--dp_weight',
                        help='Weight of the disparity smoothness loss',
                        default=0.02, type=float)
    parser.add_argument('--normal_weight',
                        help='Weight of the normal loss (if ground truth available)',
                        default=0.1, type=float)
    
    # Network settings
    parser.add_argument('--ms',
                        help='If true, use multiscale loss',
                        default=True, type=str2bool)
    parser.add_argument('--max_disp',
                        help='Maximum disparity',
                        default=128, type=int)
    
    # Multi-task learning
    parser.add_argument('--use_normal',
                        help='Whether to use normal prediction (multi-task)',
                        default=False, type=str2bool)
    
    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_args()
    
    # Create worker
    worker = exp_pol.PolarizationWorker(args)
    
    # Determine number of input channels based on input type
    channel_map = {
        'raw': 4,
        'stokes': 3,
        'dolp_aolp': 3,
        'full': 6
    }
    channels_in = channel_map[args.input_type]
    
    # Set up network
    if args.use_normal:
        # Multi-task network with normal prediction
        net = pol_networks.PolarizationMultiTaskNet(
            channels_in=channels_in,
            max_disp=args.max_disp,
            imsizes=worker.imsizes,
            output_ms=args.ms
        )
    else:
        # Only disparity and edge
        net = pol_networks.PolarizationDispEdgeDecoders(
            channels_in=channels_in,
            max_disp=args.max_disp,
            imsizes=worker.imsizes,
            output_ms=args.ms
        )
    
    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    
    # Start training/testing
    worker.do(net, optimizer)


if __name__ == '__main__':
    main()
