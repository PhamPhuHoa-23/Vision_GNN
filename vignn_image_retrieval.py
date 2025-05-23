#!/usr/bin/env python
"""
ViGNN Image-to-Image Retrieval System
Compare different embedding strategies for retrieval performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pyramid_vig
import vig

class ViGNNFeatureExtractor(nn.Module):
    """
    Extract features from different layers of ViGNN for retrieval
    """
    def __init__(self, model_name='pvig_s_224_gelu', layer_strategy='multiple'):
        super().__init__()
        
        # Load pretrained ViGNN
        self.vignn = getattr(pyramid_vig, model_name)(pretrained=False)
        self.layer_strategy = layer_strategy
        
        # Feature dimensions for different strategies
        self.feature_dims = {
            'final_features': 1024,  # Before classification head
            'stem_features': None,   # Will be set based on model
            'backbone_features': None,
            'graph_aggregated': 512,
            'multiple_layers': 2048  # Concatenated features
        }
        
        # Set feature dimensions based on model
        self._setup_feature_dims()
        
        # Remove classification head for feature extraction
        if hasattr(self.vignn, 'prediction'):
            self.vignn.prediction = nn.Identity()
        
        # Graph-level aggregation module
        self.graph_aggregation = GraphFeatureAggregator(
            input_dim=self._get_backbone_dim(),
            output_dim=512
        )
        
        # Multi-layer fusion
        self.multi_layer_fusion = MultiLayerFusion()
        
    def _setup_feature_dims(self):
        """Setup feature dimensions based on model architecture"""
        if 'ti' in self.vignn.__class__.__name__.lower():
            self.feature_dims['stem_features'] = 48
            self.feature_dims['backbone_features'] = 384
        elif 's' in self.vignn.__class__.__name__.lower():
            self.feature_dims['stem_features'] = 80
            self.feature_dims['backbone_features'] = 640
        else:
            # Default to small model dims
            self.feature_dims['stem_features'] = 80
            self.feature_dims['backbone_features'] = 640
    
    def _get_backbone_dim(self):
        """Get backbone feature dimension"""
        return self.feature_dims['backbone_features']
    
    def forward(self, x):
        """Extract features using specified strategy"""
        features = {}
        
        # 1. Stem features (early visual features)
        stem_out = self.vignn.stem(x) + self.vignn.pos_embed
        features['stem'] = F.adaptive_avg_pool2d(stem_out, 1).flatten(1)
        
        # 2. Backbone features (graph-processed features)
        backbone_features = stem_out
        intermediate_features = []
        
        for i, block in enumerate(self.vignn.backbone):
            backbone_features = block(backbone_features)
            
            # Collect intermediate features every few blocks
            if i % 3 == 0:
                pooled = F.adaptive_avg_pool2d(backbone_features, 1).flatten(1)
                intermediate_features.append(pooled)
        
        # 3. Final backbone features
        features['backbone'] = F.adaptive_avg_pool2d(backbone_features, 1).flatten(1)
        
        # 4. Graph-aggregated features
        features['graph_aggregated'] = self.graph_aggregation(backbone_features)
        
        # 5. Multi-layer fusion
        features['multi_layer'] = self.multi_layer_fusion(intermediate_features)
        
        # Return based on strategy
        if self.layer_strategy == 'stem':
            return features['stem']
        elif self.layer_strategy == 'backbone':
            return features['backbone']
        elif self.layer_strategy == 'graph_aggregated':
            return features['graph_aggregated']
        elif self.layer_strategy == 'multi_layer':
            return features['multi_layer']
        else:
            # Return all for comparison
            return features

class GraphFeatureAggregator(nn.Module):
    """
    Aggregate spatial graph features for retrieval
    Preserves spatial relationships while creating global representation
    """
    def __init__(self, input_dim, output_dim=512):
        super().__init__()
        
        # Spatial attention for graph nodes
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(input_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Graph-aware pooling
        self.graph_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )
        
        # Alternative: max pooling for distinctive features
        self.max_pooling = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(input_dim, output_dim)
        )
        
    def forward(self, x):
        # Apply spatial attention
        attention = self.spatial_attention(x)
        attended_features = x * attention
        
        # Combine avg and max pooling
        avg_features = self.graph_pooling(attended_features)
        max_features = self.max_pooling(attended_features)
        
        # Combine features
        combined = avg_features + max_features
        
        return F.normalize(combined, p=2, dim=1)

class MultiLayerFusion(nn.Module):
    """
    Fuse features from multiple layers
    """
    def __init__(self, output_dim=512):
        super().__init__()
        self.output_dim = output_dim
        
    def forward(self, feature_list):
        if not feature_list:
            return torch.zeros(1, self.output_dim)
        
        # Normalize each feature
        normalized_features = [F.normalize(f, p=2, dim=1) for f in feature_list]
        
        # Concatenate and project
        combined = torch.cat(normalized_features, dim=1)
        
        # Project to fixed dimension
        projection = nn.Linear(combined.size(1), self.output_dim).to(combined.device)
        output = projection(combined)
        
        return F.normalize(output, p=2, dim=1)

class ImageRetrievalDataset(Dataset):
    """
    Dataset for image retrieval evaluation
    """
    def __init__(self, data_path, transform=None, max_samples=None):
        self.data_path = Path(data_path)
        self.transform = transform
        
        # Load images
        self.images = list(self.data_path.glob("*.jpg")) + list(self.data_path.glob("*.png"))
        
        if max_samples:
            self.images = self.images[:max_samples]
        
        print(f"Loaded {len(self.images)} images for retrieval")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'path': str(image_path),
            'id': idx
        }

class ImageRetrievalEvaluator:
    """
    Evaluate different feature extraction strategies for image retrieval
    """
    def __init__(self, model_name='pvig_s_224_gelu'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Feature extractors for different strategies
        self.extractors = {
            'stem': ViGNNFeatureExtractor(model_name, 'stem'),
            'backbone': ViGNNFeatureExtractor(model_name, 'backbone'),
            'graph_aggregated': ViGNNFeatureExtractor(model_name, 'graph_aggregated'),
            'multi_layer': ViGNNFeatureExtractor(model_name, 'multi_layer'),
        }
        
        # Move to device
        for extractor in self.extractors.values():
            extractor.to(self.device)
            extractor.eval()
    
    def extract_features(self, dataset, strategy='backbone', batch_size=32):
        """Extract features for all images in dataset"""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        extractor = self.extractors[strategy]
        features = []
        image_paths = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f'Extracting {strategy} features'):
                images = batch['image'].to(self.device)
                paths = batch['path']
                
                batch_features = extractor(images)
                
                features.append(batch_features.cpu())
                image_paths.extend(paths)
        
        features = torch.cat(features, dim=0)
        return features.numpy(), image_paths
    
    def compute_retrieval_metrics(self, features, k_values=[1, 5, 10]):
        """Compute retrieval metrics"""
        similarity_matrix = cosine_similarity(features)
        
        metrics = {}
        
        for k in k_values:
            recall_at_k = []
            
            for i in range(len(features)):
                # Get top-k similar images (excluding self)
                similarities = similarity_matrix[i]
                top_k_indices = np.argsort(similarities)[::-1][1:k+1]  # Exclude self
                
                # For now, we consider any retrieval as correct
                # In practice, you'd have ground truth similar images
                recall_at_k.append(1.0)  # Placeholder
            
            metrics[f'recall@{k}'] = np.mean(recall_at_k)
        
        # Average similarity (excluding diagonal)
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        avg_similarity = similarity_matrix[mask].mean()
        metrics['avg_similarity'] = avg_similarity
        
        return metrics
    
    def compare_strategies(self, dataset, strategies=None):
        """Compare different feature extraction strategies"""
        if strategies is None:
            strategies = ['stem', 'backbone', 'graph_aggregated', 'multi_layer']
        
        results = {}
        
        for strategy in strategies:
            print(f"\n📊 Evaluating strategy: {strategy}")
            
            # Extract features
            features, paths = self.extract_features(dataset, strategy)
            
            # Compute metrics
            metrics = self.compute_retrieval_metrics(features)
            
            results[strategy] = {
                'features': features,
                'paths': paths,
                'metrics': metrics,
                'feature_dim': features.shape[1]
            }
            
            print(f"   Feature dim: {features.shape[1]}")
            print(f"   Avg similarity: {metrics['avg_similarity']:.3f}")
        
        return results
    
    def visualize_retrieval(self, query_idx, features, paths, top_k=5):
        """Visualize retrieval results for a query image"""
        similarity_matrix = cosine_similarity(features)
        similarities = similarity_matrix[query_idx]
        
        # Get top-k similar images
        top_k_indices = np.argsort(similarities)[::-1][:top_k+1]  # +1 to include query
        
        plt.figure(figsize=(15, 3))
        
        for i, idx in enumerate(top_k_indices):
            plt.subplot(1, len(top_k_indices), i+1)
            
            # Load and display image
            image = Image.open(paths[idx])
            plt.imshow(image)
            plt.axis('off')
            
            if i == 0:
                plt.title(f'Query\nSim: 1.000')
            else:
                plt.title(f'Top {i}\nSim: {similarities[idx]:.3f}')
        
        plt.tight_layout()
        plt.show()

def load_pretrained_vignn(model_name, pretrained_path=None):
    """Load pretrained ViGNN weights"""
    model = getattr(pyramid_vig, model_name)(pretrained=False)
    
    if pretrained_path:
        print(f"Loading pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    
    return model

def benchmark_retrieval_performance():
    """
    Benchmark different ViGNN configurations for image retrieval
    """
    print("🔍 ViGNN Image-to-Image Retrieval Benchmark")
    print("=" * 50)
    
    # Create sample dataset
    # You should replace this with your actual image dataset
    dataset_path = "data/sample_images"  # Folder with images
    
    if not os.path.exists(dataset_path):
        print(f"Creating sample dataset at {dataset_path}")
        os.makedirs(dataset_path, exist_ok=True)
        # Add some sample images here
        print("Please add images to the dataset folder and run again")
        return
    
    # Create dataset
    dataset = ImageRetrievalDataset(dataset_path, max_samples=100)
    
    if len(dataset) == 0:
        print("No images found in dataset!")
        return
    
    # Create evaluator
    evaluator = ImageRetrievalEvaluator('pvig_s_224_gelu')
    
    # Compare strategies
    results = evaluator.compare_strategies(dataset)
    
    # Print comparison
    print("\n📈 Strategy Comparison:")
    print("-" * 60)
    print(f"{'Strategy':<15} {'Feature Dim':<12} {'Avg Similarity':<15}")
    print("-" * 60)
    
    for strategy, result in results.items():
        metrics = result['metrics']
        dim = result['feature_dim']
        sim = metrics['avg_similarity']
        print(f"{strategy:<15} {dim:<12} {sim:<15.3f}")
    
    # Visualize results for first image
    if len(dataset) > 5:
        print("\n🖼️  Visualizing retrieval for first image...")
        best_strategy = max(results.keys(), key=lambda k: results[k]['metrics']['avg_similarity'])
        features = results[best_strategy]['features']
        paths = results[best_strategy]['paths']
        
        evaluator.visualize_retrieval(0, features, paths, top_k=5)
    
    return results

def main():
    """
    Main function to test ViGNN image retrieval
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='ViGNN Image Retrieval Evaluation')
    parser.add_argument('--dataset_path', type=str, default='data/sample_images',
                        help='Path to image dataset')
    parser.add_argument('--model_name', type=str, default='pvig_s_224_gelu',
                        choices=['pvig_ti_224_gelu', 'pvig_s_224_gelu', 'pvig_m_224_gelu'])
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained ViGNN weights')
    parser.add_argument('--max_samples', type=int, default=200,
                        help='Maximum number of images to evaluate')
    parser.add_argument('--strategies', nargs='+', default=['backbone', 'graph_aggregated'],
                        choices=['stem', 'backbone', 'graph_aggregated', 'multi_layer'],
                        help='Feature extraction strategies to compare')
    
    args = parser.parse_args()
    
    print(f"🔍 ViGNN Image Retrieval Evaluation")
    print(f"Dataset: {args.dataset_path}")
    print(f"Model: {args.model_name}")
    print(f"Strategies: {args.strategies}")
    print()
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageRetrievalDataset(args.dataset_path, transform, args.max_samples)
    
    if len(dataset) == 0:
        print("❌ No images found in dataset!")
        return
    
    # Create evaluator
    evaluator = ImageRetrievalEvaluator(args.model_name)
    
    # Load pretrained weights if provided
    if args.pretrained_path:
        for extractor in evaluator.extractors.values():
            extractor.vignn = load_pretrained_vignn(args.model_name, args.pretrained_path)
    
    # Compare strategies
    results = evaluator.compare_strategies(dataset, args.strategies)
    
    # Save results
    output_file = f"retrieval_results_{args.model_name}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for strategy, result in results.items():
        json_results[strategy] = {
            'metrics': result['metrics'],
            'feature_dim': result['feature_dim'],
            'num_images': len(result['paths'])
        }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")

if __name__ == '__main__':
    main()
