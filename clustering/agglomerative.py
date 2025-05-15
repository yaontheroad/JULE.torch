import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

class AgglomerativeCluster:
    """Agglomerative clustering with custom affinity."""
    
    def __init__(self, n_clusters, k_neighbors=20):
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        
    def compute_affinity(self, features):
        """
        Compute affinity matrix based on feature similarity.
        Args:
            features: torch.Tensor of shape (n_samples, n_features)
        Returns:
            affinity: torch.Tensor of shape (n_samples, n_samples)
        """
        # Convert to numpy for sklearn compatibility
        features_np = features.detach().cpu().numpy()
        
        # Compute pairwise distances
        distances = pdist(features_np, metric='euclidean')
        distances = squareform(distances)
        
        # Convert distances to affinities
        sigma = np.mean(np.sort(distances, axis=1)[:, 1:self.k_neighbors+1])
        affinity = np.exp(-distances**2 / (2 * sigma**2))
        
        return torch.from_numpy(affinity).float()
    
    def fit_predict(self, features):
        """
        Perform clustering on the features.
        Args:
            features: torch.Tensor of shape (n_samples, n_features)
        Returns:
            labels: torch.Tensor of shape (n_samples,)
        """
        affinity = self.compute_affinity(features)
        
        # Use sklearn's implementation with our affinity matrix
        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            linkage='average'
        )
        
        labels = clustering.fit_predict(affinity.cpu().numpy())
        return torch.from_numpy(labels).long()
    
    def update_clusters(self, features, current_labels):
        """
        Update cluster assignments based on current features.
        Args:
            features: torch.Tensor of shape (n_samples, n_features)
            current_labels: torch.Tensor of shape (n_samples,)
        Returns:
            new_labels: torch.Tensor of shape (n_samples,)
        """
        return self.fit_predict(features) 