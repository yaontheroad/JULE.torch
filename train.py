import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import numpy as np

from models.base_net import BaseNet
from criterions.triplet_loss import TripletLoss
from clustering.agglomerative import AgglomerativeCluster
from datasets.data_loader import get_dataloader
from evaluate.metrics import compute_nmi

def parse_args():
    parser = argparse.ArgumentParser(description='JULE Training')
    parser.add_argument('--dataset', type=str, default='USPS', help='dataset name')
    parser.add_argument('--eta', type=float, default=0.9, help='unfolding rate')
    parser.add_argument('--num_nets', type=int, default=1, help='number of networks to train')
    parser.add_argument('--use_fast', type=int, default=1, help='use fast mode')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--k_neighbors', type=int, default=20, help='number of neighbors for affinity')
    parser.add_argument('--feature_dim', type=int, default=10, help='feature dimension')
    return parser.parse_args()

class JULETrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model, loss, and optimizer
        self.model = BaseNet(feature_dim=args.feature_dim).to(self.device)
        self.criterion = TripletLoss(margin=0.2, gamma=1.0)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=5e-5
        )
        
        # Initialize clustering
        self.clustering = AgglomerativeCluster(
            n_clusters=10,  # This should be set based on dataset
            k_neighbors=args.k_neighbors
        )
        
        # Get data loader
        self.dataloader = get_dataloader(
            args.dataset,
            batch_size=args.batch_size
        )
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(self.dataloader)):
            data = data.to(self.device)
            
            # Get features
            features = self.model(data)
            
            # Update clusters
            with torch.no_grad():
                pseudo_labels = self.clustering.update_clusters(features, None)
            
            # Compute loss
            loss = self.criterion(features, pseudo_labels)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.dataloader)
    
    def evaluate(self):
        self.model.eval()
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for data, labels in self.dataloader:
                data = data.to(self.device)
                features = self.model(data)
                features_list.append(features.cpu())
                labels_list.append(labels)
        
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        # Get cluster assignments
        pseudo_labels = self.clustering.fit_predict(features)
        
        # Compute NMI
        nmi = compute_nmi(labels.numpy(), pseudo_labels.numpy())
        return nmi
    
    def train(self):
        best_nmi = 0
        
        for epoch in range(self.args.epochs):
            # Train for one epoch
            loss = self.train_epoch()
            
            # Evaluate
            nmi = self.evaluate()
            
            print(f'Epoch {epoch+1}/{self.args.epochs}:')
            print(f'Loss: {loss:.4f}, NMI: {nmi:.4f}')
            
            if nmi > best_nmi:
                best_nmi = nmi
                # Save best model
                torch.save(self.model.state_dict(), f'checkpoints/{self.args.dataset}_best.pth')

def main():
    args = parse_args()
    
    # Create trainer and train
    trainer = JULETrainer(args)
    trainer.train()

if __name__ == '__main__':
    main() 