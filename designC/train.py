#!/usr/bin/env python3
"""
Design C - Training Script
Train PyTorch VRN for volumetric face reconstruction
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
import time
import json
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.vrn_pytorch import VRNPyTorch
from models.losses import CombinedLoss, BCEWithLogitsLoss3D, FocalLoss3D
from data.facescape_loader import FaceScapeDataset, TeacherVolumeDataset, collate_fn_allow_none


class Trainer:
    """Training harness for Design C"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Model
        print(f"Initializing model on {self.device}...")
        self.model = VRNPyTorch(
            input_size=tuple(args.image_size),
            volume_shape=tuple(args.volume_shape)
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Loss function
        if args.loss_type == 'bce':
            self.criterion = BCEWithLogitsLoss3D()
        elif args.loss_type == 'focal':
            self.criterion = FocalLoss3D(alpha=args.focal_alpha, gamma=args.focal_gamma)
        elif args.loss_type == 'combined':
            self.criterion = CombinedLoss(
                use_bce=True, bce_weight=1.0,
                use_focal=args.use_focal, focal_weight=args.focal_weight,
                use_boundary=args.use_boundary, boundary_weight_loss=args.boundary_weight
            )
        else:
            raise ValueError(f"Unknown loss type: {args.loss_type}")
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=args.scheduler_patience,
            verbose=True
        )
        
        # Dataset and dataloaders
        print("Loading datasets...")
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # Resume from checkpoint if specified
        if args.resume:
            self._load_checkpoint(args.resume)
    
    def _create_dataloaders(self):
        """Create train/val dataloaders"""
        args = self.args
        
        if args.dataset_type == 'facescape':
            # FaceScape voxelized shapes
            train_dataset = FaceScapeDataset(
                image_dir=args.image_dir if hasattr(args, 'image_dir') else None,
                volume_dir=args.volume_dir,
                image_size=tuple(args.image_size),
                expression_filter=args.expression_filter
            )
            
            # For now, use same dataset for validation (will split later)
            val_dataset = train_dataset
            
        elif args.dataset_type == 'teacher':
            # Teacher-student distillation
            train_dataset = TeacherVolumeDataset(
                image_dir=args.image_dir,
                teacher_volume_dir=args.volume_dir,
                image_size=tuple(args.image_size)
            )
            val_dataset = train_dataset
        else:
            raise ValueError(f"Unknown dataset type: {args.dataset_type}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_allow_none
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_allow_none
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Skip if no images available yet
            if batch['image'] is None:
                print("Warning: No images in batch, skipping...")
                continue
            
            # Move to device
            images = batch['image'].to(self.device)
            volumes = batch['volume'].to(self.device)
            
            # Forward pass
            logits = self.model(images)
            
            # Compute loss
            if isinstance(self.criterion, CombinedLoss):
                loss, loss_dict = self.criterion(logits, volumes)
            else:
                loss = self.criterion(logits, volumes)
                loss_dict = {'total': loss.item()}
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.args.grad_clip
                )
            
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            pbar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # TensorBoard logging
            if self.global_step % self.args.log_interval == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}_loss', value, self.global_step)
                
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        return avg_loss
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                if batch['image'] is None:
                    continue
                
                images = batch['image'].to(self.device)
                volumes = batch['volume'].to(self.device)
                
                logits = self.model(images)
                
                if isinstance(self.criterion, CombinedLoss):
                    loss, _ = self.criterion(logits, volumes)
                else:
                    loss = self.criterion(logits, volumes)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        
        # TensorBoard logging
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        
        print(f"Validation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'args': vars(self.args)
        }
        
        # Save latest
        checkpoint_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path}")
        
        # Save periodic
        if epoch % self.args.save_interval == 0:
            periodic_path = self.checkpoint_dir / f'epoch_{epoch:03d}.pth'
            torch.save(checkpoint, periodic_path)
    
    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.global_step = checkpoint['global_step']
        
        print(f"Resumed from epoch {checkpoint['epoch']}")
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.args.epochs} epochs...")
        print(f"Device: {self.device}")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.args.epochs} ===")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            if (epoch + 1) % self.args.val_interval == 0:
                val_loss = self.validate(epoch)
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                # Save checkpoint
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                self.save_checkpoint(epoch, val_loss, is_best=is_best)
            else:
                # Save checkpoint without validation
                self.save_checkpoint(epoch, train_loss, is_best=False)
        
        print("\n✓ Training complete!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Design C PyTorch VRN')
    
    # Dataset
    parser.add_argument('--dataset-type', type=str, default='facescape',
                        choices=['facescape', 'teacher'],
                        help='Dataset type: facescape (voxelized shapes) or teacher (distillation)')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Image directory (optional for facescape, required for teacher)')
    parser.add_argument('--volume-dir', type=str, required=True,
                        help='Volume directory (.npy files)')
    parser.add_argument('--expression-filter', type=str, default='1_neutral',
                        help='Expression filter for FaceScape (e.g., 1_neutral)')
    
    # Model
    parser.add_argument('--image-size', type=int, nargs=2, default=[224, 224],
                        help='Input image size')
    parser.add_argument('--volume-shape', type=int, nargs=3, default=[200, 192, 192],
                        help='Output volume shape')
    
    # Loss
    parser.add_argument('--loss-type', type=str, default='bce',
                        choices=['bce', 'focal', 'combined'],
                        help='Loss function type')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                        help='Focal loss alpha')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma')
    parser.add_argument('--use-focal', action='store_true',
                        help='Use focal loss in combined loss')
    parser.add_argument('--focal-weight', type=float, default=0.5,
                        help='Focal loss weight in combined loss')
    parser.add_argument('--use-boundary', action='store_true',
                        help='Use boundary-weighted loss')
    parser.add_argument('--boundary-weight', type=float, default=0.5,
                        help='Boundary loss weight')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size (small due to 3D volumes)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping (0 to disable)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    
    # Scheduler
    parser.add_argument('--scheduler-patience', type=int, default=5,
                        help='LR scheduler patience')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='designC/checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='designC/logs',
                        help='TensorBoard log directory')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Validate every N epochs')
    
    args = parser.parse_args()
    
    # Save config
    config_path = Path(args.checkpoint_dir) / 'config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Train
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
