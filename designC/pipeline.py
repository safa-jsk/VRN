#!/usr/bin/env python3
"""
Design C - Integrated Inference Pipeline
PyTorch VRN → Design B CUDA Marching Cubes → Mesh Export
"""

import torch
import numpy as np
from pathlib import Path
import sys
import time
from PIL import Image
import torchvision.transforms as transforms

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from models.vrn_pytorch import VRNPyTorch

# Import Design B components
from designB.python.marching_cubes_cuda import MarchingCubesCUDA
from designB.python.volume_io import save_mesh_obj


class DesignCPipeline:
    """
    End-to-end Design C pipeline
    
    Stages:
    1. Image preprocessing
    2. PyTorch VRN inference (GPU)
    3. Design B CUDA marching cubes (GPU)
    4. Design B validated post-processing
    """
    
    def __init__(self, 
                 model_path,
                 device='cuda',
                 image_size=(224, 224),
                 threshold=0.5,
                 apply_vrn_transform=True,
                 vertex_merge_tolerance=0.1):
        """
        Args:
            model_path: Path to trained PyTorch VRN checkpoint
            device: 'cuda' or 'cpu'
            image_size: Input image size
            threshold: Occupancy threshold (0.5 from Design B)
            apply_vrn_transform: Apply Design B validated transforms
            vertex_merge_tolerance: Trimesh vertex merge (0.1 from Design B)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.threshold = threshold
        self.apply_vrn_transform = apply_vrn_transform
        self.vertex_merge_tolerance = vertex_merge_tolerance
        
        # Load model
        print(f"Loading PyTorch VRN from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine volume shape from checkpoint
        if 'args' in checkpoint:
            volume_shape = tuple(checkpoint['args']['volume_shape'])
        else:
            volume_shape = (200, 192, 192)
        
        self.model = VRNPyTorch(
            input_size=image_size,
            volume_shape=volume_shape
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # CUDA Marching Cubes (Design B)
        print("Initializing CUDA Marching Cubes...")
        self.mc = MarchingCubesCUDA(device=self.device)
        
        print("✓ Pipeline initialized")
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess image
        
        Args:
            image_path: Path to input image
        
        Returns:
            image_tensor: (1, 3, H, W) preprocessed tensor
        """
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        return img_tensor
    
    def predict_volume(self, image_tensor, return_numpy=True):
        """
        Run PyTorch VRN inference
        
        Args:
            image_tensor: (B, 3, H, W) input image
            return_numpy: Return numpy array instead of torch tensor
        
        Returns:
            volume: (B, X, Y, Z) boolean occupancy volume
            inference_time: Time in seconds
        """
        image_tensor = image_tensor.to(self.device)
        
        # Warmup (first inference may be slow)
        if not hasattr(self, '_warmed_up'):
            with torch.no_grad():
                _ = self.model.predict(image_tensor, threshold=self.threshold)
            self._warmed_up = True
        
        # Timed inference
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        
        with torch.no_grad():
            volume = self.model.predict(image_tensor, threshold=self.threshold)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        inference_time = time.time() - start
        
        if return_numpy:
            volume = volume.cpu().numpy()
        
        return volume, inference_time
    
    def volume_to_mesh(self, volume, image_path=None):
        """
        Convert volume to mesh using Design B CUDA marching cubes
        
        Args:
            volume: (X, Y, Z) boolean numpy array or torch tensor
            image_path: Optional path to image for color mapping
        
        Returns:
            mesh_data: dict with 'vertices' and 'faces'
            mc_time: Marching cubes time in seconds
        """
        # Convert to numpy if needed
        if torch.is_tensor(volume):
            volume = volume.cpu().numpy()
        
        # Remove batch dimension if present
        if volume.ndim == 4:
            volume = volume[0]
        
        # Run CUDA marching cubes
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        vertices, faces = self.mc.marching_cubes(volume)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        mc_time = time.time() - start
        
        mesh_data = {
            'vertices': vertices,
            'faces': faces
        }
        
        return mesh_data, mc_time
    
    def save_mesh(self, mesh_data, output_path, image_path=None, return_timing=True):
        """
        Save mesh using Design B validated post-processing
        
        Args:
            mesh_data: dict with 'vertices' and 'faces'
            output_path: Output .obj path
            image_path: Input image for color mapping
            return_timing: Return post-processing time
        
        Returns:
            post_time: Post-processing time (if return_timing)
        """
        start = time.time()
        
        # Use Design B's save_mesh_obj with validated parameters
        save_mesh_obj(
            vertices=mesh_data['vertices'],
            faces=mesh_data['faces'],
            output_path=output_path,
            apply_vrn_transform=self.apply_vrn_transform,
            image_path=image_path,
            vertex_merge_tolerance=self.vertex_merge_tolerance
        )
        
        post_time = time.time() - start
        
        if return_timing:
            return post_time
    
    def process_image(self, image_path, output_path, verbose=True):
        """
        End-to-end pipeline: image → mesh
        
        Args:
            image_path: Input image path
            output_path: Output .obj path
            verbose: Print timing breakdown
        
        Returns:
            timing: dict with stage timings
        """
        timing = {}
        
        # 1. Preprocess
        start = time.time()
        image_tensor = self.preprocess_image(image_path)
        timing['preprocess'] = time.time() - start
        
        # 2. VRN inference (GPU)
        volume, inference_time = self.predict_volume(image_tensor)
        timing['inference'] = inference_time
        
        # 3. CUDA marching cubes (GPU)
        mesh_data, mc_time = self.volume_to_mesh(volume, image_path)
        timing['marching_cubes'] = mc_time
        
        # 4. Post-processing (Design B validated)
        post_time = self.save_mesh(mesh_data, output_path, image_path)
        timing['post_processing'] = post_time
        
        # Total
        timing['total'] = sum(timing.values())
        
        if verbose:
            print(f"\nProcessed: {Path(image_path).name}")
            print(f"  Preprocess:      {timing['preprocess']*1000:7.2f} ms")
            print(f"  VRN Inference:   {timing['inference']*1000:7.2f} ms  (GPU)")
            print(f"  Marching Cubes:  {timing['marching_cubes']*1000:7.2f} ms  (GPU)")
            print(f"  Post-processing: {timing['post_processing']*1000:7.2f} ms")
            print(f"  ─────────────────────────────")
            print(f"  Total:           {timing['total']*1000:7.2f} ms")
            print(f"  Output: {output_path}")
        
        return timing


def main():
    """Test pipeline on a single image"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Design C Inference Pipeline')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Input image path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .obj path')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Occupancy threshold')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = DesignCPipeline(
        model_path=args.model,
        device=args.device,
        threshold=args.threshold
    )
    
    # Process image
    timing = pipeline.process_image(args.image, args.output)
    
    print("\n✓ Pipeline test complete!")


if __name__ == '__main__':
    main()
