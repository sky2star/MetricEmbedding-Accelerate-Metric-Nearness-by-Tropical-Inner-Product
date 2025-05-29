import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import dropout_adj
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse

class GCL(nn.Module):
    """Graph Contrastive Learning Model"""
    def __init__(self, input_dim, hidden_dim, output_dim, proj_dim):
        super(GCL, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            GCNConv(input_dim, hidden_dim),
            nn.ReLU(),
            GCNConv(hidden_dim, output_dim)
        )
        
        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        # Classifier for node classification
        self.classifier = nn.Linear(output_dim, output_dim)
    
    def forward(self, x, edge_index):
        # Get node representations
        h = self.encoder[0](x, edge_index)
        h = F.relu(h)
        h = self.encoder[2](h, edge_index)
        
        # Node classification
        out = self.classifier(h)
        
        return h, out
    
    def get_embedding(self, x, edge_index):
        h = self.encoder[0](x, edge_index)
        h = F.relu(h)
        h = self.encoder[2](h, edge_index)
        return h
    
    def projection(self, h):
        # Project to contrastive learning space
        z = self.projector(h)
        return F.normalize(z) # Normalize

def augment(x, edge_index, p_node=0.1, p_edge=0.1):
    """Graph augmentation function"""
    # Node feature augmentation - random masking
    x_aug = x.clone()
    node_mask = torch.FloatTensor(x.shape[0], 1).uniform_() > p_node
    x_aug = x_aug * node_mask.to(x.device)
    
    # Edge augmentation - random edge dropout
    edge_index_aug, _ = dropout_adj(edge_index, p=p_edge)
    
    return x_aug, edge_index_aug

def compute_similarity(z1, z2, sim_type='cosine', temperature=0.5):
    """Compute similarity matrix based on specified similarity type"""
    
    if sim_type == 'cosine':
        # Cosine similarity
        similarity_matrix = torch.matmul(z1, z2.T) / temperature
    elif sim_type == 'euclidean':
        # Euclidean distance (converted to similarity)
        dist = torch.cdist(z1, z2, p=2)
        similarity_matrix = -dist / temperature
    elif sim_type == 'manhattan':
        # Manhattan distance (converted to similarity)
        dist = torch.cdist(z1, z2, p=1)
        similarity_matrix = -dist / temperature
    elif sim_type == 'hamming':
        # Hamming distance (approximated for continuous values)
        z1_binary = (z1 > 0).float()
        z2_binary = (z2 > 0).float()
        dist = torch.cdist(z1_binary, z2_binary, p=1)
        similarity_matrix = -dist / temperature
    elif sim_type == 'tropical':
        z1 = F.relu(z1)
        z2 = F.relu(z2)

        N = z1.shape[0]
        K = z1.shape[1]
        M = z2.shape[0]
        expanded_x = z1.unsqueeze(1).expand(N, M, K)  # N x M x K
        expanded_w = z2.unsqueeze(0).expand(N, M, K)  # N x M x K
        
        sum_result = expanded_x + expanded_w
        result = torch.max(sum_result, dim=2)[0]
        # mask = (torch.ones(N, N) - torch.eye(N)).to(z1.device)
        dist = result
        similarity_matrix = -dist / temperature
        # print(similarity_matrix)

    else:
        # Default to cosine similarity
        similarity_matrix = torch.matmul(z1, z2.T) / temperature
    
    return F.normalize(similarity_matrix)

def contrastive_loss(z1, z2, sim_type='cosine', temperature=0.5):
    """InfoNCE contrastive loss with different similarity metrics"""
    # Compute similarity matrix
    similarity_matrix = compute_similarity(z1, z2, sim_type, temperature)
    
    # Positive pairs
    positives = torch.diag(similarity_matrix)
    
    # Negative pairs - all other samples
    negatives_1 = torch.logsumexp(similarity_matrix, dim=1) - positives
    negatives_2 = torch.logsumexp(similarity_matrix, dim=0) - positives
    
    # InfoNCE loss
    loss_1 = -torch.mean(positives - negatives_1)
    loss_2 = -torch.mean(positives - negatives_2)
    
    return (loss_1 + loss_2) / 2

def train(model, data, optimizer, device, sim_type='cosine', alpha=0.8):
    """Training function"""
    model.train()
    optimizer.zero_grad()
    
    # Original data
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    y = data.y.to(device)
    
    # Data augmentation
    x_aug1, edge_index_aug1 = augment(x, edge_index)
    x_aug2, edge_index_aug2 = augment(x, edge_index)
    
    # Original view
    h, out = model(x, edge_index)
    z = model.projection(h)
    
    # Augmented views
    h1, _ = model(x_aug1, edge_index_aug1)
    z1 = model.projection(h1)
    
    h2, _ = model(x_aug2, edge_index_aug2)
    z2 = model.projection(h2)
    
    # Compute losses
    cl_loss = contrastive_loss(z1, z2, sim_type=sim_type)
    ce_loss = F.cross_entropy(out[data.train_mask], y[data.train_mask])
    
    # Total loss = contrastive loss + classification loss
    loss = alpha * cl_loss + (1 - alpha) * ce_loss
    
    loss.backward()
    optimizer.step()
    
    return loss.item(), cl_loss.item(), ce_loss.item()

def evaluate(model, data, device, mask):
    """Evaluation function"""
    model.eval()
    with torch.no_grad():
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        _, out = model(x, edge_index)
        pred = out.argmax(dim=1)
        correct = pred[mask] == data.y[mask].to(device)
        acc = int(correct.sum()) / int(mask.sum())
    return acc

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Graph Contrastive Learning for Node Classification')
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'CiteSeer'],
                        help='Dataset name (default: Cora)')
    parser.add_argument('--similarity', type=str, default='cosine', 
                        help='Similarity function for contrastive learning (default: cosine)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension (default: 256)')
    parser.add_argument('--proj_dim', type=int, default=128, help='Projection dimension (default: 128)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default: 5e-4)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs (default: 200)')
    parser.add_argument('--patience', type=int, default=200, help='Patience for early stopping (default: 20)')
    parser.add_argument('--alpha', type=float, default=0.8, 
                        help='Weight for contrastive loss (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device (default: cuda)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Dataset list (if only one dataset is specified)
    datasets = [args.dataset]
    
    for dataset_name in datasets:
        print(f"\n{'='*50}\nProcessing dataset: {dataset_name}\n{'='*50}")
        
        # Load dataset
        dataset = Planetoid(root='./dataset', name=dataset_name, transform=NormalizeFeatures())
        data = dataset[0]
        
        # Model parameters
        input_dim = dataset.num_features
        hidden_dim = args.hidden_dim
        output_dim = dataset.num_classes
        proj_dim = args.proj_dim
        
        # Initialize model
        model = GCL(input_dim, hidden_dim, output_dim, proj_dim).to(device)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Training parameters
        epochs = args.epochs
        patience = args.patience
        best_val_acc = 0
        counter = 0
        
        # Training loop
        for epoch in tqdm(range(epochs)):
            # Train
            loss, cl_loss, ce_loss = train(model, data, optimizer, device, sim_type=args.similarity, alpha=args.alpha)
            
            # Validate
            val_acc = evaluate(model, data, device, data.val_mask)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
                # Save best model
                torch.save(model.state_dict(), f'best_model_{dataset_name}_{args.similarity}.pt')
            else:
                counter += 1
                
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, CL Loss: {cl_loss:.4f}, CE Loss: {ce_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Load best model
        model.load_state_dict(torch.load(f'best_model_{dataset_name}_{args.similarity}.pt'))
        
        # Test
        test_acc = evaluate(model, data, device, data.test_mask)
        print(f'Final test accuracy: {test_acc:.4f}')
        print(f'Best validation accuracy: {best_val_acc:.4f}')
        print(f'Similarity function: {args.similarity}')

if __name__ == '__main__':
    main()
