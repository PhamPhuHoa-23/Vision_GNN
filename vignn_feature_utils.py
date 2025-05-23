#!/usr/bin/env python
"""
Utilities for working with extracted ViGNN features
Loading, analysis, visualization, similarity search
"""

import numpy as np
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
import argparse

class FeatureManager:
    """Manager for loading and working with extracted features"""
    
    def __init__(self, features_dir, metadata_file=None):
        self.features_dir = Path(features_dir)
        self.metadata_file = metadata_file or self.features_dir.parent / "extraction_metadata.json"
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.feature_files = self._get_feature_files()
        
        print(f"📁 Features directory: {self.features_dir}")
        print(f"📊 Found {len(self.feature_files)} feature files")
        
    def _load_metadata(self):
        """Load extraction metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            print(f"⚠️ Metadata file not found: {self.metadata_file}")
            return {}
    
    def _get_feature_files(self):
        """Get list of all feature files"""
        feature_files = []
        
        # Support both .npy and .pt files
        for ext in ['.npy', '.pt']:
            feature_files.extend(list(self.features_dir.glob(f"*{ext}")))
        
        return sorted(feature_files)
    
    def load_feature(self, filename_or_path):
        """Load a single feature file"""
        if isinstance(filename_or_path, str):
            if '/' in filename_or_path or '\\' in filename_or_path:
                file_path = Path(filename_or_path)
            else:
                # Just filename, look in features directory
                file_path = None
                for ext in ['.npy', '.pt']:
                    candidate = self.features_dir / f"{filename_or_path}{ext}"
                    if candidate.exists():
                        file_path = candidate
                        break
                
                if file_path is None:
                    raise FileNotFoundError(f"Feature file not found: {filename_or_path}")
        else:
            file_path = Path(filename_or_path)
        
        try:
            if file_path.suffix == '.npy':
                return np.load(file_path)
            elif file_path.suffix == '.pt':
                feature = torch.load(file_path, map_location='cpu')
                if isinstance(feature, torch.Tensor):
                    return feature.numpy()
                return feature
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        except Exception as e:
            raise IOError(f"Failed to load {file_path}: {e}")
    
    def load_all_features(self, max_files=None):
        """Load all features into memory"""
        features = []
        filenames = []
        
        files_to_load = self.feature_files[:max_files] if max_files else self.feature_files
        
        print(f"Loading {len(files_to_load)} features...")
        
        for file_path in tqdm(files_to_load, desc="Loading features"):
            try:
                feature = self.load_feature(file_path)
                features.append(feature)
                filenames.append(file_path.stem)
            except Exception as e:
                print(f"⚠️ Failed to load {file_path}: {e}")
        
        if features:
            features_array = np.vstack(features)
            print(f"✅ Loaded {len(features)} features with shape {features_array.shape}")
            return features_array, filenames
        else:
            print("❌ No features loaded successfully")
            return None, None
    
    def get_feature_stats(self, max_files=1000):
        """Compute statistics across all features"""
        features, filenames = self.load_all_features(max_files)
        
        if features is None:
            return None
        
        stats = {
            'num_features': len(features),
            'feature_dim': features.shape[1],
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0),
            'global_mean': np.mean(features),
            'global_std': np.std(features),
            'global_min': np.min(features),
            'global_max': np.max(features)
        }
        
        return stats
    
    def find_similar_images(self, query_feature_or_file, top_k=10, metric='cosine'):
        """Find most similar images to a query"""
        # Load query feature
        if isinstance(query_feature_or_file, (str, Path)):
            query_feature = self.load_feature(query_feature_or_file)
        else:
            query_feature = query_feature_or_file
        
        # Load all features
        features, filenames = self.load_all_features()
        
        if features is None:
            return None
        
        # Compute similarities
        if metric == 'cosine':
            # Reshape for sklearn
            query_feature = query_feature.reshape(1, -1)
            similarities = cosine_similarity(query_feature, features)[0]
        elif metric == 'euclidean':
            # Negative distance (higher = more similar)
            similarities = -np.linalg.norm(features - query_feature, axis=1)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Get top-k similar
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'filename': filenames[idx],
                'similarity': similarities[idx],
                'feature_file': self.feature_files[idx]
            })
        
        return results
    
    def visualize_features(self, max_files=1000, method='pca', save_path=None):
        """Visualize features using dimensionality reduction"""
        features, filenames = self.load_all_features(max_files)
        
        if features is None:
            return
        
        print(f"Visualizing {len(features)} features using {method.upper()}...")
        
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            reduced_features = reducer.fit_transform(features)
            title = f"PCA Visualization of ViGNN Features\nExplained Variance: {reducer.explained_variance_ratio_.sum():.3f}"
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
            reduced_features = reducer.fit_transform(features)
            title = "t-SNE Visualization of ViGNN Features"
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.6, s=20)
        plt.title(title)
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to {save_path}")
        else:
            plt.show()
        
        return reduced_features
    
    def analyze_feature_distribution(self, max_files=1000, save_path=None):
        """Analyze and plot feature distributions"""
        stats = self.get_feature_stats(max_files)
        
        if stats is None:
            return
        
        # Create subplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Global statistics
        axes[0, 0].hist(stats['mean'], bins=50, alpha=0.7, color='blue', label='Mean')
        axes[0, 0].hist(stats['std'], bins=50, alpha=0.7, color='red', label='Std')
        axes[0, 0].set_title('Feature Dimension Statistics')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Min/Max ranges
        axes[0, 1].hist(stats['min'], bins=50, alpha=0.7, color='green', label='Min')
        axes[0, 1].hist(stats['max'], bins=50, alpha=0.7, color='orange', label='Max')
        axes[0, 1].set_title('Feature Value Ranges')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature dimension means over indices
        axes[1, 0].plot(stats['mean'], alpha=0.7, color='blue')
        axes[1, 0].set_title('Feature Means by Dimension')
        axes[1, 0].set_xlabel('Feature Dimension')
        axes[1, 0].set_ylabel('Mean Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature dimension stds over indices
        axes[1, 1].plot(stats['std'], alpha=0.7, color='red')
        axes[1, 1].set_title('Feature Std by Dimension')
        axes[1, 1].set_xlabel('Feature Dimension')
        axes[1, 1].set_ylabel('Std Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Distribution analysis saved to {save_path}")
        else:
            plt.show()
        
        # Print summary
        print(f"\n📊 Feature Statistics Summary:")
        print(f"   Number of features: {stats['num_features']}")
        print(f"   Feature dimension: {stats['feature_dim']}")
        print(f"   Global mean: {stats['global_mean']:.4f}")
        print(f"   Global std: {stats['global_std']:.4f}")
        print(f"   Global range: [{stats['global_min']:.4f}, {stats['global_max']:.4f}]")

def demo_feature_analysis(features_dir):
    """Demo feature analysis functionality"""
    print("🧪 Demo: ViGNN Feature Analysis")
    print("=" * 40)
    
    # Load feature manager
    fm = FeatureManager(features_dir)
    
    if len(fm.feature_files) == 0:
        print("❌ No features found!")
        return
    
    # Test loading single feature
    print("\n1. Loading single feature...")
    try:
        first_feature = fm.load_feature(fm.feature_files[0])
        print(f"✅ Loaded feature shape: {first_feature.shape}")
        print(f"   Range: [{first_feature.min():.4f}, {first_feature.max():.4f}]")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return
    
    # Compute statistics
    print("\n2. Computing feature statistics...")
    stats = fm.get_feature_stats(max_files=100)
    if stats:
        print(f"✅ Stats computed for {stats['num_features']} features")
        print(f"   Feature dim: {stats['feature_dim']}")
        print(f"   Mean activation: {stats['global_mean']:.4f}")
    
    # Find similar images
    print("\n3. Finding similar images...")
    similar = fm.find_similar_images(fm.feature_files[0], top_k=5)
    if similar:
        print("✅ Top 5 most similar images:")
        for i, result in enumerate(similar):
            print(f"   {i+1}. {result['filename']} (similarity: {result['similarity']:.4f})")
    
    # Visualize features
    print("\n4. Creating PCA visualization...")
    try:
        reduced_features = fm.visualize_features(max_files=500, method='pca', 
                                               save_path=Path(features_dir).parent / "feature_pca.png")
        if reduced_features is not None:
            print(f"✅ PCA visualization created: {reduced_features.shape}")
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")
    
    # Analyze distributions
    print("\n5. Analyzing feature distributions...")
    try:
        fm.analyze_feature_distribution(max_files=500, 
                                      save_path=Path(features_dir).parent / "feature_distribution.png")
        print("✅ Distribution analysis completed")
    except Exception as e:
        print(f"⚠️ Distribution analysis failed: {e}")
    
    print("\n🎉 Demo completed!")

def main():
    parser = argparse.ArgumentParser(description='Analyze extracted ViGNN features')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing extracted features')
    parser.add_argument('--action', type=str, default='demo',
                        choices=['demo', 'stats', 'similar', 'visualize'],
                        help='Action to perform')
    parser.add_argument('--query_image', type=str, default=None,
                        help='Query image for similarity search')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of similar images to return')
    parser.add_argument('--max_files', type=int, default=1000,
                        help='Maximum number of files to process')
    parser.add_argument('--vis_method', type=str, default='pca',
                        choices=['pca', 'tsne'],
                        help='Visualization method')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save plots instead of showing them')
    
    args = parser.parse_args()
    
    # Initialize feature manager
    fm = FeatureManager(args.features_dir)
    
    if args.action == 'demo':
        demo_feature_analysis(args.features_dir)
    
    elif args.action == 'stats':
        print("Computing feature statistics...")
        stats = fm.get_feature_stats(args.max_files)
        if stats:
            print(json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v 
                            for k, v in stats.items() if k not in ['mean', 'std', 'min', 'max']}, 
                           indent=2))
    
    elif args.action == 'similar':
        if not args.query_image:
            print("❌ Please provide --query_image for similarity search")
            return
        
        print(f"Finding images similar to {args.query_image}...")
        similar = fm.find_similar_images(args.query_image, args.top_k)
        if similar:
            print(f"Top {args.top_k} similar images:")
            for i, result in enumerate(similar):
                print(f"{i+1:2d}. {result['filename']} (similarity: {result['similarity']:.4f})")
    
    elif args.action == 'visualize':
        save_path = Path(args.features_dir).parent / f"feature_{args.vis_method}.png" if args.save_plots else None
        fm.visualize_features(args.max_files, args.vis_method, save_path)

if __name__ == '__main__':
    main()
