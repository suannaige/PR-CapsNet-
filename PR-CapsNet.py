import os
import sys
import random
import datetime
import logging
import argparse
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Coauthor
from torch_scatter import scatter_add, scatter_softmax


# ==========================================
# Global Config & Constants
# ==========================================

EPSILON = 1e-12
MAX_TANH_VAL = 10.0
CLIP_VAL = 1e4
ACOSH_MIN = 1.0 + 1e-6

# Log
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    dataset_name: str = "Cora"
    s_dim: int = 9                  # Space-like dimensions
    t_dim: int = 9                  # Time-like dimensions
    hidden_dim: int = 64            # GCN hidden dimensions
    num_routing: int = 3            # Iterations for dynamic routing
    num_perspectives: int = 4       # Number of perspectives in ACR
    dropout: float = 0.5            # Dropout rate
    lr: float = 0.005               # Learning rate
    weight_decay: float = 5e-4      # Weight decay
    epochs: int = 200               # Training epochs
    seed: int = 2903                 # Random seed
    learnable_curvature: bool = True
    model_type: str = "PR-CapsNet"  # 'PR-CapsNet' or 'Euclidean'

# ==========================================
# 1. Utils
# ==========================================

def setup_seed(seed: int) -> None:

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 保证 CuDNN 的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. Geometry Core
# ==========================================

class PseudoRiemannianManifold:
    
    @staticmethod
    def inner_product(x: torch.Tensor, y: torch.Tensor, t_dim: int) -> torch.Tensor:
        x_time, x_space = x[..., :t_dim], x[..., t_dim:]
        y_time, y_space = y[..., :t_dim], y[..., t_dim:]

        time_prod = torch.sum(x_time * y_time, dim=-1, keepdim=True)
        space_prod = torch.sum(x_space * y_space, dim=-1, keepdim=True)
        
        return -time_prod + space_prod

    @staticmethod
    def get_pole(dim: int, t_dim: int, beta: float, device: torch.device) -> torch.Tensor:
        pole = torch.zeros(1, dim, device=device)
        target_scale = np.sqrt(abs(beta))
        
        # Beta < 0: Pseudo-Hyperbolic (Pole on time axis)
        # Beta > 0: Pseudo-Spherical (Pole on space axis)
        if beta < 0:
            pole[0, 0] = target_scale
        else:
            idx = t_dim if dim > t_dim else 0
            pole[0, idx] = target_scale 
        return pole

    @staticmethod
    def diffeo_log_map(x: torch.Tensor, ref: torch.Tensor, t_dim: int) -> torch.Tensor:
        x = x.clamp(min=-CLIP_VAL, max=CLIP_VAL)
        inner = PseudoRiemannianManifold.inner_product(x, ref, t_dim)
        ref_sq = PseudoRiemannianManifold.inner_product(ref, ref, t_dim)
        x_sq = PseudoRiemannianManifold.inner_product(x, x, t_dim)
        
        denom = torch.sqrt((ref_sq * x_sq).abs().clamp(min=EPSILON))
        cos_dist = -inner / denom
        
        cos_dist = cos_dist.clamp(min=ACOSH_MIN) 
        
        dist = torch.acosh(cos_dist)
        
        sin_dist = torch.sqrt(cos_dist**2 - 1).clamp(min=EPSILON)
        coef = dist / sin_dist
        
        # v = coef * (x - cosh(dist) * ref_unit * |ref|) ... Similar to LogMap
        term1 = x
        term2 = (inner / ref_sq) * ref 
        
        tangent_vec = coef * (term1 + term2)
        return tangent_vec

    @staticmethod
    def diffeo_exp_map(v: torch.Tensor, ref: torch.Tensor, t_dim: int) -> torch.Tensor:
        """
        Exp Map: Tangent Space -> Manifold
        """
        v_norm_sq = PseudoRiemannianManifold.inner_product(v, v, t_dim)
        # Clamp v_norm
        v_norm = torch.sqrt(v_norm_sq.abs() + EPSILON).clamp(max=MAX_TANH_VAL) 
        
        coef_cosh = torch.cosh(v_norm)
        coef_sinh = torch.sinh(v_norm) / v_norm
        
        res = coef_cosh * ref + coef_sinh * v
        return res

# ==========================================
# 3. Main Layers: Routing Layer
# ==========================================

class AdaptiveCurvatureRouting(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, num_perspectives: int = 4):
        super().__init__()
        self.num_perspectives = num_perspectives
        self.out_dim = out_dim

        self.per_transform = nn.Linear(in_dim, out_dim * num_perspectives)
        
        # calculate gate weights
        self.gating_fc = nn.Linear(in_dim, num_perspectives)

    def forward(self, x_tangent: torch.Tensor) -> torch.Tensor:

        # [E, K * out_dim] -> [E, K, out_dim]
        u_hat_k = self.per_transform(x_tangent).view(-1, self.num_perspectives, self.out_dim)
        
        # calculate gate weights
        gate_logits = self.gating_fc(x_tangent) # [E, K]
        gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1) # [E, K, 1]
        
        # fused features
        u_hat_fused = torch.sum(u_hat_k * gate_weights, dim=1) # [E, out_dim]
        return u_hat_fused

class PR_RoutingLayer(nn.Module):
    """
    Pseudo-Riemannian Routing Layer
    Workflow: LogMap -> Dropout -> ACR -> Dynamic Routing -> ExpMap
    """
    def __init__(self, in_dim: int, out_dim: int, t_dim: int, 
                 num_routing: int = 3, num_perspectives: int = 4, dropout: float = 0.5):
        super().__init__()
        self.t_dim = t_dim
        self.num_routing = num_routing
        self.dropout = dropout
        
        self.acr = AdaptiveCurvatureRouting(in_dim, out_dim, num_perspectives)
        self.bias = nn.Parameter(torch.zeros(1, out_dim))
        
        self.register_buffer("ref_point", None)

    @staticmethod
    def squash(s: torch.Tensor) -> torch.Tensor:
        norm_sq = torch.sum(s**2, dim=-1, keepdim=True)
        scale = norm_sq / (1 + norm_sq)
        norm = torch.sqrt(norm_sq + 1e-9)
        return scale * (s / norm)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, beta: float = -1.0) -> torch.Tensor:

        num_nodes = x.size(0)
        row, col = edge_index
        
        # 0. Lazy Initialization
        if self.ref_point is None:
            self.ref_point = PseudoRiemannianManifold.get_pole(x.size(1), self.t_dim, beta, x.device)
            
        # 1. Log Map: 
        ref_expand = self.ref_point.expand(num_nodes, -1)
        x_tangent = PseudoRiemannianManifold.diffeo_log_map(x, ref_expand, self.t_dim)
        
        # 2. Dropout: 
        x_tangent = F.dropout(x_tangent, p=self.dropout, training=self.training)
        
        # 3. ACR
        x_child_tangent = x_tangent[row] 
        u_hat_tangent = self.acr(x_child_tangent) # [E, out_dim]
        
        # 4. Dynamic Routing 
        b_ij = torch.zeros(edge_index.size(1), device=x.device)
        s_j_tangent = torch.zeros(num_nodes, u_hat_tangent.size(1), device=x.device) # Init
        
        for r in range(self.num_routing):
            # Coupling Coefficients
            c_ij = scatter_softmax(b_ij, col, dim=0)
            
            # Weighted Aggregation
            weighted_u = c_ij.unsqueeze(-1) * (u_hat_tangent + self.bias)
            s_j_tangent = scatter_add(weighted_u, col, dim=0, dim_size=num_nodes)
            
            # Squash Activation
            s_j_tangent = self.squash(s_j_tangent)
            
            # Agreement Update 
            if r < self.num_routing - 1:
                agreement = (s_j_tangent[col] * u_hat_tangent).sum(dim=-1)
                b_ij = b_ij + agreement
        
        # 5. Exp Map
        ref_out = self.ref_point.expand(num_nodes, -1)
        v_j = PseudoRiemannianManifold.diffeo_exp_map(s_j_tangent, ref_out, self.t_dim)
        
        return v_j

# ==========================================
# 4. Model Architecture
# ==========================================

class PRCapsNet(nn.Module):
    """
    PR-CapsNet Model
    Architecture: GCN Encoder -> Linear -> PR Routing Layers -> Tangent Space Classifier
    """
    def __init__(self, config: ModelConfig, num_features: int, num_classes: int):
        super().__init__()
        self.t_dim = config.t_dim
        self.s_dim = config.s_dim
        self.capsule_dim = config.t_dim + config.s_dim 
        self.dropout = config.dropout
        
        # GNN Encoder
        self.gcn = GCNConv(num_features, config.hidden_dim)
        self.pre_cap = nn.Linear(config.hidden_dim, self.capsule_dim)
        
        # 可学习曲率参数 (beta)
        if config.learnable_curvature:
            self.beta = nn.Parameter(torch.tensor(-1.0))
        else:
            self.register_buffer("beta", torch.tensor(-1.0))

        # Stack of Routing Layers
        self.layers = nn.ModuleList([
            PR_RoutingLayer(
                in_dim=self.capsule_dim, 
                out_dim=self.capsule_dim, 
                t_dim=config.t_dim, 
                num_routing=config.num_routing, 
                num_perspectives=config.num_perspectives, 
                dropout=config.dropout
            )
            # For multiple routing layers, you can loop through the layers in the forward method.
        ])
            
        # Classifier Reference Point (Lazy Init)
        self.register_buffer("classifier_ref", None)
        
        # Final Classifier
        self.classifier = nn.Linear(self.capsule_dim, num_classes)
        self.norm = nn.LayerNorm(self.capsule_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        
        # 1. Encoder Block
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gcn(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 2. Feature Projection
        x = self.pre_cap(x)
        x = self.norm(x) # LayerNorm 
        
        # 3. Pseudo-Riemannian Routing Stack
        for layer in self.layers:
            x = layer(x, edge_index, beta=self.beta.item())
            
        # 4. PRCC: Tangent Space Classifier
        if self.classifier_ref is None:
             self.classifier_ref = PseudoRiemannianManifold.get_pole(
                 self.capsule_dim, self.t_dim, self.beta.item(), x.device
             )
        
        ref = self.classifier_ref.expand(x.size(0), -1)
        z_j = PseudoRiemannianManifold.diffeo_log_map(x, ref, self.t_dim)
        
        z_j = F.dropout(z_j, p=self.dropout, training=self.training)
        logits = self.classifier(z_j)
        
        return logits

# --- Baseline: Euclidean CapsNet ---
class EuclideanCapsNode(nn.Module):
    """
    Euclidean Baseline for ablation studies.
    """
    def __init__(self, config: ModelConfig, num_features: int, num_classes: int):
        super().__init__()
        capsule_dim = config.t_dim + config.s_dim
        self.dropout = config.dropout
        self.gcn = GCNConv(num_features, config.hidden_dim)
        self.pre_cap = nn.Linear(config.hidden_dim, capsule_dim)
        self.layer = self.EuclideanRoutingLayer(capsule_dim, capsule_dim, config.num_routing)
        self.classifier = nn.Linear(capsule_dim, num_classes)
        self.norm = nn.LayerNorm(capsule_dim)

    class EuclideanRoutingLayer(nn.Module):
        def __init__(self, in_dim, out_dim, iter_num):
            super().__init__()
            self.iter_num = iter_num
            self.W = nn.Linear(in_dim, out_dim)
        
        def squash(self, s):
            norm_sq = torch.sum(s**2, dim=-1, keepdim=True)
            scale = norm_sq / (1 + norm_sq)
            norm = torch.sqrt(norm_sq + 1e-9)
            return scale * (s / norm)

        def forward(self, x, edge_index):
            u_hat = self.W(x)
            u_hat_edges = u_hat[edge_index[0]]
            b_ij = torch.zeros(edge_index.size(1), device=x.device)
            v_j = None
            col = edge_index[1]
            for r in range(self.iter_num):
                c_ij = scatter_softmax(b_ij, col, dim=0)
                v_j = scatter_add(c_ij.unsqueeze(-1) * u_hat_edges, col, dim=0, dim_size=x.size(0))
                v_j = self.squash(v_j)
                if r < self.iter_num - 1:
                    agreement = (v_j[col] * u_hat_edges).sum(dim=-1)
                    b_ij = b_ij + agreement
            return v_j

    def forward(self, data):
        x = F.dropout(data.x, p=self.dropout, training=self.training)
        x = F.relu(self.gcn(x, data.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.norm(self.pre_cap(x))
        x = self.layer(x, data.edge_index)
        return self.classifier(x)

# ==========================================
# 5. Training
# ==========================================

class Trainer:
    
    def __init__(self, config: ModelConfig, device: torch.device):
        self.config = config
        self.device = device
        self.dataset, self.data = self._load_data()
        self.model = self._build_model()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def _load_data(self) -> Tuple[Any, Data]:
            """Split Train/Val/Test """
            path = os.path.join('data', self.config.dataset_name)
            
            # 1. Load Dataset
            if self.config.dataset_name in ['Cora', 'Citeseer', 'PubMed']:
                dataset = Planetoid(root=path, name=self.config.dataset_name)
            elif self.config.dataset_name == 'CoauthorCS':
                dataset = Coauthor(root=path, name='CS')
            else:
                raise ValueError(f"Unknown dataset: {self.config.dataset_name}")
                
            data = dataset[0].to(self.device)
            
            g_cpu = torch.Generator()
            g_cpu.manual_seed(self.config.seed)
            
            num_nodes = data.num_nodes
            indices = torch.randperm(num_nodes, generator=g_cpu) 
            
            n_train = int(0.6 * num_nodes)
            n_val = int(0.2 * num_nodes)
            
            data.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
            
            data.train_mask[indices[:n_train]] = True
            data.val_mask[indices[n_train:n_train+n_val]] = True
            data.test_mask[indices[n_train+n_val:]] = True
            
            return dataset, data

    def _build_model(self) -> nn.Module:
        if self.config.model_type == 'PR-CapsNet':
            return PRCapsNet(self.config, self.dataset.num_features, self.dataset.num_classes).to(self.device)
        elif self.config.model_type == 'Euclidean':
            return EuclideanCapsNode(self.config, self.dataset.num_features, self.dataset.num_classes).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def run(self) -> float:
        """Run"""
        best_val_acc = 0.0
        final_test_acc = 0.0
        
        patience = 50
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            loss = self._train_step()
            val_acc, test_acc = self._evaluate()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch:03d}: Loss={loss:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f} (Best Test={final_test_acc:.4f})")
                
        return final_test_acc

    def _train_step(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        try:
            out = self.model(self.data)
            loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            return loss.item()
        except RuntimeError as e:
            logger.error(f"Training error (NaN/Inf likely): {e}")
            return float('inf')

    @torch.no_grad()
    def _evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        out = self.model(self.data)
        pred = out.argmax(dim=1)
        
        val_acc = (pred[self.data.val_mask] == self.data.y[self.data.val_mask]).float().mean().item()
        test_acc = (pred[self.data.test_mask] == self.data.y[self.data.test_mask]).float().mean().item()
        
        return val_acc, test_acc

# ==========================================
# 6. Main
# ==========================================

def run_experiment_suite(device: torch.device):
    """
    run_experiment(Cora,Citeseer)
    """
    logger.info("Starting Experiment Suite...")
    
    experiments = [
        ModelConfig(
            dataset_name='Cora', model_type='PR-CapsNet', 
            s_dim=9, t_dim=9, dropout=0.0, num_routing=3, learnable_curvature=True,
            seed=42
        ),
        ModelConfig(
            dataset_name='Citeseer', model_type='PR-CapsNet',
            s_dim=9, t_dim=9, dropout=0.5, num_routing=3, learnable_curvature=True,
            seed=42
        ),
    ]
    
    results = []
    
    for i, config in enumerate(experiments):
        logger.info(f"Running Experiment [{i+1}/{len(experiments)}]: {config.dataset_name} | {config.model_type} | Dims: {config.s_dim}/{config.t_dim}")
        
        # Set Global Seed
        setup_seed(config.seed)
        
        try:
            trainer = Trainer(config, device)
            acc = trainer.run()
            
            logger.info(f"--> Final Result: {acc:.4f}")
            
            # result
            res_dict = {k:v for k,v in config.__dict__.items()}
            res_dict['final_test_acc'] = acc
            results.append(res_dict)
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)

    # save
    if results:
        df = pd.DataFrame(results)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"pr_capsnet_results_{timestamp}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"All experiments completed. Results saved to {filename}")
        print("\n=== Summary ===")
        print(df[['dataset_name', 'model_type', 's_dim', 't_dim', 'final_test_acc']])

if __name__ == "__main__":
    # hyperparameters
    parser = argparse.ArgumentParser(description="PR-CapsNet Experiment Runner")
    parser.add_argument('--mode', type=str, default='suite', choices=['suite', 'single'], help="Run full suite or single experiment")
    parser.add_argument('--dataset', type=str, default='Cora', help="Dataset name for single run")
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Using device: {device}")

    if args.mode == 'suite':
        run_experiment_suite(device)
    else:
        # example
        single_config = ModelConfig(dataset_name=args.dataset)
        setup_seed(single_config.seed)
        trainer = Trainer(single_config, device)
        acc = trainer.run()
        logger.info(f"Single experiment {args.dataset} finished with Acc: {acc:.4f}")