# m3gnet_torch/trainers/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader # 导入 DataLoader
from typing import List, Optional, Dict, Union
import numpy as np
from tqdm import tqdm
import os

from ..models import M3GNet
from ..graph import MaterialGraph, RadiusCutoffGraphConverter, M3GNetDataset, collate_fn_base # <-- 导入 M3GNetDataset 和 collate_fn_base
from ..type import StructureOrMolecule

# --- 移除这里重复的 M3GNetDataset 和 collate_fn_base 定义 ---
# class M3GNetDataset(Dataset): ...
# def collate_fn_base(batch): ...

class Trainer:
    """A universal trainer for M3GNet models."""
    def __init__(self, model: M3GNet, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device: str = "cpu"):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        self.model.to(device)
        self.loss_fn.to(device)

    def train(
        self,
        train_structures: List[StructureOrMolecule],
        train_targets: List,
        val_structures: Optional[List[StructureOrMolecule]] = None,
        val_targets: Optional[List] = None,
        epochs: int = 1000,
        batch_size: int = 32,
        is_efs_training: bool = False,
        force_loss_ratio: float = 1.0,
        stress_loss_ratio: float = 0.1,
        save_dir: str = "checkpoints",
        early_stop_patience: int = 50
    ):
        train_dataset = M3GNetDataset(train_structures, train_targets, self.model.graph_converter, is_efs_explicit=is_efs_training)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_base)

        val_loader = None
        if val_structures:
            val_dataset = M3GNetDataset(val_structures, val_targets, self.model.graph_converter, is_efs_explicit=is_efs_training)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_base)
        
        self.train_loader = train_loader # <--- 暴露 train_loader 作为属性
        self.val_loader = val_loader # <--- 暴露 val_loader 作为属性

        if not os.path.exists(save_dir): os.makedirs(save_dir)
        best_val_loss = float('inf')
        patience_counter = 0

        print("Starting training...")
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [T]")
            for batched_graph, batched_targets_list in pbar:
                batched_graph.to(self.device)
                self.optimizer.zero_grad()
                
                if train_dataset.is_efs:
                    targets_e = torch.cat([t[0].to(self.device) for t in batched_targets_list])
                    targets_f = torch.cat([t[1].to(self.device) for t in batched_targets_list])
                    targets_s = torch.cat([t[2].to(self.device) for t in batched_targets_list]) if len(batched_targets_list[0]) > 2 else None
                    
                    pred_e, pred_f, pred_s = self.model.get_efs(batched_graph)
                    loss_e = self.loss_fn(pred_e.squeeze(), targets_e.squeeze())
                    loss_f = self.loss_fn(pred_f, targets_f)
                    loss = loss_e + force_loss_ratio * loss_f
                    if targets_s is not None and pred_s is not None:
                        loss_s = self.loss_fn(pred_s, targets_s)
                        loss += stress_loss_ratio * loss_s
                else:
                    targets_on_device = torch.cat([t.to(self.device) for t in batched_targets_list])
                    predictions = self.model(batched_graph)
                    loss = self.loss_fn(predictions, targets_on_device)
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_loader)
            
            if val_loader:
                val_loss = self.evaluate(val_loader, self.loss_fn, train_dataset.is_efs, force_loss_ratio, stress_loss_ratio)
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss, patience_counter = val_loss, 0
                    self.model.save(os.path.join(save_dir, "best_model.pt"))
                    print(f"  -> New best model saved with validation loss {val_loss:.4f}")
                else:
                    patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print("Early stopping triggered."); break
            else:
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, loss_fn: nn.Module, is_efs: bool, force_loss_ratio: float, stress_loss_ratio: float) -> float:
        self.model.eval()
        total_loss = 0
        pbar = tqdm(data_loader, desc="           [V]")
        for batched_graph, batched_targets_list in pbar:
            batched_graph.to(self.device)
            if is_efs:
                targets_e = torch.cat([t[0].to(self.device) for t in batched_targets_list])
                targets_f = torch.cat([t[1].to(self.device) for t in batched_targets_list])
                targets_s = torch.cat([t[2].to(self.device) for t in batched_targets_list]) if len(batched_targets_list[0]) > 2 else None
                
                pred_e, pred_f, pred_s = self.model.get_efs(batched_graph)
                loss_e = loss_fn(pred_e.squeeze(), targets_e.squeeze())
                loss_f = loss_fn(pred_f, targets_f)
                loss = loss_e + force_loss_ratio * loss_f
                if targets_s is not None and pred_s is not None:
                    loss_s = loss_fn(pred_s, targets_s)
                    loss += stress_loss_ratio * loss_s
            else:
                targets_on_device = torch.cat([t.to(self.device) for t in batched_targets_list])
                predictions = self.model(batched_graph)
                loss = loss_fn(predictions, targets_on_device)
            total_loss += loss.item()
        return total_loss / len(data_loader)