# m3gnet_torch/trainers/tests/test_trainer.py

import unittest
import torch
import torch.nn as nn
import os
import shutil
import numpy as np
from pymatgen.core import Lattice, Structure

from torch.utils.data import DataLoader
from m3gnet.models import M3GNet
from m3gnet.trainers import Trainer
from m3gnet.graph import RadiusCutoffGraphConverter, M3GNetDataset, collate_fn_base

class TestTrainer(unittest.TestCase):

    def setUp(self):
        print("\n--- Setting up for a new test ---")
        # 使用更小的模型加速测试
        self.model = M3GNet(
            max_n=1, max_l=1, n_blocks=1, units=8,  # 减少units大小
            cutoff=4.0, threebody_cutoff=3.0
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss() 
        
        # 创建更小的测试结构
        s1 = Structure(Lattice.cubic(3.0), ["H"], [[0, 0, 0]])
        s2 = Structure(Lattice.cubic(3.5), ["O"], [[0, 0, 0]])
        s3 = Structure(Lattice.cubic(4.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        self.train_structures = [s1, s2, s3]
        self.train_targets = [-1.0, -2.0, -5.0]
        
        s_val = Structure(Lattice.cubic(3.2), ["H", "H"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        self.val_structures = [s_val]
        self.val_targets = [-2.5]

        self.save_dir = "temp_test_checkpoints"
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    def tearDown(self):
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    def test_01_property_training_and_saving(self):
        print("\n--- Running Test 1: Property Training and Saving ---")
        
        trainer = Trainer(self.model, self.optimizer, self.loss_fn) 
        
        # 减少epochs和batch_size
        trainer.train(
            train_structures=self.train_structures,
            train_targets=self.train_targets,
            val_structures=self.val_structures,
            val_targets=self.val_targets,
            epochs=1,  # 减少到1个epoch
            batch_size=1,  # 使用batch_size=1
            is_efs_training=False, 
            save_dir=self.save_dir
        )
        
        best_model_path = os.path.join(self.save_dir, "best_model.pt")
        self.assertTrue(os.path.exists(best_model_path))
        print("Model saving successful.")

    def test_02_loading_and_prediction(self):
        print("\n--- Running Test 2: Loading and Prediction ---")
        
        trainer = Trainer(self.model, self.optimizer, self.loss_fn) 
        trainer.train(
            train_structures=self.train_structures,
            train_targets=self.train_targets,
            val_structures=self.val_structures,
            val_targets=self.val_targets,
            epochs=1, batch_size=1, is_efs_training=False, save_dir=self.save_dir
        )
        
        model_path = os.path.join(self.save_dir, "best_model.pt")
        try:
            loaded_model = M3GNet.load(model_path)
        except Exception as e:
            self.fail(f"M3GNet.load() failed with an error: {e}")
            
        self.assertIsInstance(loaded_model, M3GNet)
        print("Model loading successful.")
        
        # 使用更小的结构进行预测
        small_struct = Structure(Lattice.cubic(3.0), ["H"], [[0, 0, 0]])
        predictions = loaded_model.predict([small_struct])
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape, (1, 1))
        print(f"Prediction successful. Shape: {predictions.shape}")

    def test_03_evaluation(self):
        print("\n--- Running Test 3: Evaluation Method ---")
        
        # 使用更小的模型
        eval_model = M3GNet(n_blocks=1, units=8)  # 减少units大小
        eval_optimizer = torch.optim.Adam(eval_model.parameters(), lr=1e-3)
        eval_loss_fn = nn.L1Loss()
        
        eval_trainer = Trainer(eval_model, eval_optimizer, eval_loss_fn)
        
        # 使用更小的验证集
        val_dataset_for_eval = M3GNetDataset(
            self.val_structures, self.val_targets, 
            eval_model.graph_converter, is_efs_explicit=False
        )
        val_loader_for_eval = DataLoader(
            val_dataset_for_eval, batch_size=1, shuffle=False, 
            collate_fn=collate_fn_base
        )

        mae_loss = eval_trainer.evaluate(
            data_loader=val_loader_for_eval,
            loss_fn=eval_loss_fn, 
            is_efs=False, 
            force_loss_ratio=0.0, stress_loss_ratio=0.0
        )
        
        self.assertIsInstance(mae_loss, float)
        print(f"Evaluation successful. MAE loss: {mae_loss:.4f}")

if __name__ == '__main__':
    unittest.main()