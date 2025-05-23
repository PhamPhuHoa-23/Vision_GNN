#!/usr/bin/env python
"""
ViGNN Feature Extractor
Extract image embeddings using pre-trained ViG models
Save individual .npy files for each image
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import time

# Import ViG models
import pyramid_vig
import vig

class ViGFeatureExtractor(nn.Module):
    """
    ViG model without classification head for feature extraction
    """
    def __init__(self, model_name='pvig_s_224_gelu', pretrained_path=None, embedding_dim=None):
        super().__init__()
        
        # Load original ViG model
        if model_name.startswith('pvig'):
            self.model = getattr(pyramid_vig, model_name)(pretrained=False)
        else:
            self.model = getattr(vig, model_name)(pretrained=False)
        
        # Load pretrained weights if provided
        if pretrained_path:
            print(f"Loading pretrained weights from {pretrained_path}")
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                
                # Handle different state dict formats
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # Remove 'module.' prefix if present (from DataParallel)
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                
                self.model.load_state_dict(new_state_dict, strict=False)
                print("✅ Pretrained weights loaded successfully")
                
            except Exception as e:
                print(f"⚠️ Failed to load pretrained weights: {e}")
                print("Continuing with random initialization...")
        
        # Remove classification head and get feature dimension
        if hasattr(self.model, 'prediction'):
            # Pyramid ViG
            self.feature_dim = self.model.prediction[0].in_channels
            self.model.prediction = nn.Identity()
        elif hasattr(self.model, 'head'):
            # Other ViG variants
            self.feature_dim = self.model.head.in_features
            self.model.head = nn.Identity()
        else:
            # Default for original ViG
            self.feature_dim = 1024
            # Remove the last prediction layer
            if hasattr(self.model, 'backbone') and len(self.model.backbone) > 0:
                # Find the last conv layer for feature extraction
                for name, module in reversed(list(self.model.named_modules())):
                    if isinstance(module, nn.Conv2d) and '1024' in str(module.out_channels):
                        self.feature_dim = module.out_channels
                        break
        
        print(f"Feature dimension: {self.feature_dim}")
        
        # Optional: Add projection layer for custom embedding dimension
        if embedding_dim and embedding_dim != self.feature_dim:
            self.projection = nn.Linear(self.feature_dim, embedding_dim)
            self.feature_dim = embedding_dim
        else:
            self.projection = None
        
        self.model.eval()
    
    def forward(self, x):
        """Extract features from input images"""
        with torch.no_grad():
            # Forward through ViG backbone
            if hasattr(self.model, 'stem'):
                # Pyramid ViG
                x = self.model.stem(x) + self.model.pos_embed
                for block in self.model.backbone:
                    x = block(x)
                # Global average pooling
                x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
            else:
                # Original ViG
                x = self.model.stem(x) + self.model.pos_embed
                for i in range(self.model.n_blocks):
                    x = self.model.backbone[i](x)
                # Global average pooling
                x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
            
            # Optional projection
            if self.projection:
                x = self.projection(x)
            
            return x

class ImageDataset(Dataset):
    """Dataset for loading images from directory"""
    
    def __init__(self, image_dir, transform=None, supported_formats=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.supported_formats = supported_formats
        
        # Find all image files
        self.image_paths = []
        for ext in supported_formats:
            self.image_paths.extend(list(self.image_dir.glob(f"*{ext}")))
            self.image_paths.extend(list(self.image_dir.glob(f"*{ext.upper()}")))
        
        # Remove duplicates and sort
        self.image_paths = sorted(list(set(self.image_paths)))
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return {
                'image': image,
                'path': str(image_path),
                'filename': image_path.name,
                'stem': image_path.stem
            }
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                image = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                image = torch.zeros(3, 224, 224)
            
            return {
                'image': image,
                'path': str(image_path),
                'filename': image_path.name,
                'stem': image_path.stem
            }

def get_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def extract_features(model, dataloader, output_dir, device, save_format='npy'):
    """Extract features and save to files"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different formats
    if save_format == 'npy':
        features_dir = output_path / "features_npy"
    elif save_format == 'pt':
        features_dir = output_path / "features_pt"
    else:
        features_dir = output_path / "features"
    
    features_dir.mkdir(exist_ok=True)
    
    # Metadata for tracking
    metadata = []
    failed_files = []
    
    print(f"Extracting features to {features_dir}")
    print(f"Using device: {device}")
    
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
            try:
                images = batch['image'].to(device)
                
                # Extract features
                features = model(images)
                
                # Save individual feature files
                for i in range(len(features)):
                    feature_vector = features[i].cpu().numpy()
                    filename = batch['stem'][i]
                    
                    if save_format == 'npy':
                        feature_file = features_dir / f"{filename}.npy"
                        np.save(feature_file, feature_vector)
                    elif save_format == 'pt':
                        feature_file = features_dir / f"{filename}.pt"
                        torch.save(features[i].cpu(), feature_file)
                    
                    # Record metadata
                    metadata.append({
                        'original_path': batch['path'][i],
                        'filename': batch['filename'][i],
                        'stem': filename,
                        'feature_file': str(feature_file.relative_to(output_path)),
                        'feature_shape': list(feature_vector.shape),
                        'feature_dim': len(feature_vector)
                    })
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                for i in range(len(batch['filename'])):
                    failed_files.append({
                        'path': batch['path'][i],
                        'filename': batch['filename'][i],
                        'error': str(e)
                    })
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Save metadata
    metadata_file = output_path / "extraction_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            'total_images': len(metadata),
            'failed_images': len(failed_files),
            'processing_time_seconds': processing_time,
            'features_per_second': len(metadata) / processing_time if processing_time > 0 else 0,
            'feature_dimension': metadata[0]['feature_dim'] if metadata else 0,
            'save_format': save_format,
            'device_used': str(device),
            'successful_extractions': metadata,
            'failed_extractions': failed_files
        }, f, indent=2)
    
    print(f"\n✅ Feature extraction completed!")
    print(f"📊 Processed: {len(metadata)} images")
    print(f"❌ Failed: {len(failed_files)} images")
    print(f"⏱️ Time: {processing_time:.2f} seconds ({len(metadata)/processing_time:.1f} images/sec)")
    print(f"📁 Features saved to: {features_dir}")
    print(f"📋 Metadata saved to: {metadata_file}")
    
    return len(metadata), len(failed_files)

def load_and_test_feature(feature_file):
    """Test loading a saved feature file"""
    try:
        if feature_file.suffix == '.npy':
            feature = np.load(feature_file)
        elif feature_file.suffix == '.pt':
            feature = torch.load(feature_file)
            if isinstance(feature, torch.Tensor):
                feature = feature.numpy()
        
        print(f"✅ Successfully loaded feature from {feature_file}")
        print(f"   Shape: {feature.shape}")
        print(f"   Dtype: {feature.dtype}")
        print(f"   Range: [{feature.min():.4f}, {feature.max():.4f}]")
        print(f"   Mean: {feature.mean():.4f}, Std: {feature.std():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load {feature_file}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract ViGNN features from images')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='pvig_s_224_gelu',
                        choices=['pvig_ti_224_gelu', 'pvig_s_224_gelu', 'pvig_m_224_gelu', 'pvig_b_224_gelu',
                                'vig_ti_224_gelu', 'vig_s_224_gelu', 'vig_b_224_gelu'],
                        help='ViG model name')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained model weights')
    parser.add_argument('--embedding_dim', type=int, default=None,
                        help='Custom embedding dimension (optional projection)')
    
    # Data arguments
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for features')
    
    # Processing arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_format', type=str, default='npy',
                        choices=['npy', 'pt'],
                        help='Format to save features (numpy or pytorch)')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda, cuda:0, etc.)')
    
    # Testing arguments
    parser.add_argument('--test_loading', action='store_true',
                        help='Test loading a few saved features after extraction')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 ViGNN Feature Extractor")
    print(f"===========================")
    print(f"Model: {args.model_name}")
    print(f"Pretrained: {args.pretrained_path or 'None (random init)'}")
    print(f"Image dir: {args.image_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save format: {args.save_format}")
    if args.embedding_dim:
        print(f"Custom embedding dim: {args.embedding_dim}")
    print()
    
    # Create feature extractor
    print("Loading model...")
    feature_extractor = ViGFeatureExtractor(
        model_name=args.model_name,
        pretrained_path=args.pretrained_path,
        embedding_dim=args.embedding_dim
    ).to(device)
    
    # Create dataset and dataloader
    print("Setting up data loading...")
    transform = get_transforms()
    dataset = ImageDataset(args.image_dir, transform=transform)
    
    if len(dataset) == 0:
        print("❌ No images found in the specified directory!")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Extract features
    success_count, failed_count = extract_features(
        feature_extractor, 
        dataloader, 
        args.output_dir, 
        device,
        args.save_format
    )
    
    # Test loading some features
    if args.test_loading and success_count > 0:
        print("\n🧪 Testing feature loading...")
        
        if args.save_format == 'npy':
            features_dir = Path(args.output_dir) / "features_npy"
            test_files = list(features_dir.glob("*.npy"))[:3]
        else:
            features_dir = Path(args.output_dir) / "features_pt"
            test_files = list(features_dir.glob("*.pt"))[:3]
        
        for test_file in test_files:
            load_and_test_feature(test_file)
    
    print(f"\n🎉 Extraction completed!")
    print(f"📊 Success rate: {success_count}/{success_count + failed_count} ({100*success_count/(success_count + failed_count):.1f}%)")

if __name__ == '__main__':
    main()
