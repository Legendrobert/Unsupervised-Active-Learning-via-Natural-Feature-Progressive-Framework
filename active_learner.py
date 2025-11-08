# -*- coding: utf-8 -*-
"""
主动学习模块 - 封装主动学习流程
"""
import numpy as np
import torch
from tqdm import tqdm
from feature_extractor import SampleSelector
from model_trainer import ELMTrainer
from data_loader import CIFAR100Loader


class ActiveLearner:
    """主动学习器"""
    
    def __init__(self, config, data_loader, feature_extractor, model_trainer):
        """
        初始化主动学习器
        Args:
            config: 配置对象
            data_loader: 数据加载器
            feature_extractor: 特征提取器
            model_trainer: 模型训练器
        """
        self.config = config
        self.data_loader = data_loader
        self.feature_extractor = feature_extractor
        self.model_trainer = model_trainer
        self.sample_selector = SampleSelector(
            n_clusters=config.n_clusters,
            n_init=config.kmeans_n_init,
            max_iter=config.kmeans_max_iter,
            random_state=config.seed
        )
    
    def run(self, all_train_data, all_train_labels, test_data):
        """
        运行主动学习流程
        Args:
            all_train_data: 所有训练数据
            all_train_labels: 所有训练标签
            test_data: 测试数据
        Returns:
            current_indices: 最终选择的样本索引
            current_data: 最终选择的样本数据
            results: 实验结果字典
        """
        print("="*60)
        print("创建并训练参考ELM模型...")
        ref_model = self.model_trainer.create_model(
            hidden_dim=self.config.ref_hidden_dim,
            activation='sigmoid',
            C=1e-2,
            type_=1
        )
        
        ref_final_mse, ref_final_sample_mse = self.model_trainer.train_model(
            ref_model, all_train_data, "参考ELM模型"
        )
        
        print(f"参考模型每个样本MSE统计:")
        print(f"  样本数量: {len(ref_final_sample_mse)}")
        print(f"  最小值: {ref_final_sample_mse.min():.6f}")
        print(f"  最大值: {ref_final_sample_mse.max():.6f}")
        print(f"  均值: {ref_final_sample_mse.mean():.6f}")
        print(f"  标准差: {ref_final_sample_mse.std():.6f}")
        print(f"  中位数: {np.median(ref_final_sample_mse):.6f}")
        
        print("="*60)
        print("初始化：从全部训练数据中均匀抽样样本...")
        
        initial_samples_per_class = int(self.config.initial_samples_ratio * self.config.target_size * 0.01)
        current_indices, current_data, remaining_indices, remaining_data = \
            self.sample_selector.select_samples(
                all_train_data, all_train_labels, 
                initial_samples_per_class, 
                self.feature_extractor
            )
        
        self.data_loader.print_class_distribution(current_indices, all_train_labels, "初始")
        
        print("="*60)
        print("开始ELM主动学习迭代...")
        
        pbar = tqdm(
            total=self.config.target_size - len(current_data),
            desc="ELM主动学习进度",
            unit="样本"
        )
        
        iteration = 0
        
        while len(current_data) < self.config.target_size:
            iteration += 1
            
            current_model = self.model_trainer.create_model(
                hidden_dim=self.config.current_hidden_dim_base + iteration,
                activation='sine',
                C=self.config.C,
                type_=0
            )
            
            current_model.fit(current_data)
            
            if len(remaining_data) == 0:
                break
            
            pool_size = min(self.config.pool_size, len(remaining_data))
            pool_indices, pool_data, pool_remaining_indices = \
                self.sample_selector.create_candidate_pool(
                    remaining_data, remaining_indices,                     pool_size=pool_size
                )
            
            pool_data_tensor = torch.from_numpy(pool_data) if isinstance(pool_data, np.ndarray) else pool_data
            loss_differences, ref_losses, current_losses = \
                self.model_trainer.compute_loss_differences(
                    ref_final_sample_mse, current_model, pool_data_tensor,
                    pool_indices, loss_type=self.config.loss_type
                )
            
            num_to_select = min(
                self.config.samples_per_iteration,
                len(remaining_data),
                self.config.target_size - len(current_data)
            )
            
            top_diff_indices = np.argsort(loss_differences)[-num_to_select:][::-1]
            selected_pool_indices = pool_remaining_indices[top_diff_indices]
            selected_original_indices = remaining_indices[selected_pool_indices]
            
            selected_data = remaining_data[selected_pool_indices]
            current_data = np.vstack([current_data, selected_data])
            current_indices = np.concatenate([current_indices, selected_original_indices])
            
            remaining_mask = np.ones(len(remaining_data), dtype=bool)
            remaining_mask[selected_pool_indices] = False
            remaining_data = remaining_data[remaining_mask]
            remaining_indices = remaining_indices[remaining_mask]
            
            selected_labels = self.data_loader.get_labels_from_data(all_train_labels, selected_original_indices)
            class_counts = {}
            for label in selected_labels:
                class_num = int(label)
                class_counts[class_num] = class_counts.get(class_num, 0) + 1
            
            class_info = ", ".join([f"{cls}:{cnt}" for cls, cnt in sorted(class_counts.items()) if cnt > 0])
            
            pbar.update(num_to_select)
            pbar.set_postfix({
                'iter': iteration,
                'current': len(current_data),
                'selected': class_info,
                'loss': self.config.loss_type.upper()
            })
            
            if iteration % self.config.print_interval == 0:
                pool_labels = self.data_loader.get_labels_from_data(all_train_labels, pool_indices)
                self.model_trainer.print_loss_results(
                    pool_indices, loss_differences, ref_losses, current_losses,
                    pool_labels, loss_type=self.config.loss_type, top_k=self.config.top_k
                )
        
        pbar.close()
        
        print("\n" + "="*60)
        print("ELM主动学习完成，进行最终评估...")
        
        final_current_model = self.model_trainer.create_model(
            hidden_dim=self.config.final_hidden_dim,
            activation='sigmoid',
            C=1e-2,
            type_=1
        )
        
        final_current_mse, final_current_sample_mse = self.model_trainer.train_model(
            final_current_model, current_data, "最终ELM模型"
        )
        
        print("在测试集上评估ELM模型...")
        test_subset = test_data[:self.config.test_subset_size]
        
        ref_test_mse = self.model_trainer.evaluate_model(ref_model, test_subset)
        final_current_test_mse = self.model_trainer.evaluate_model(final_current_model, test_subset)
        
        print("="*60)
        print("ELM主动学习最终结果总结:")
        print(f"最终当前模型训练数据量: {len(current_data)} 样本")
        print(f"参考模型训练集 - MSE: {ref_final_mse:.6f}")
        print(f"最终当前模型训练集 - MSE: {final_current_mse:.6f}")
        print(f"参考模型测试集 - MSE: {ref_test_mse:.6f}")
        print(f"最终当前模型测试集 - MSE: {final_current_test_mse:.6f}")
        print(f"总迭代次数: {iteration}")
        print(f"目标达成: {len(current_data) >= self.config.target_size}")
        print(f"特征提取方法: {self.config.feature_method}")
        print(f"损失计算方法: {self.config.loss_type.upper()}")
        
        self.data_loader.print_class_distribution(current_indices, all_train_labels, "最终选择的")
        
        results = {
            'ref_model_results': {
                'train_mse': float(ref_final_mse),
                'test_mse': float(ref_test_mse),
                'data_size': int(len(all_train_data))
            },
            'final_model_results': {
                'train_mse': float(final_current_mse),
                'test_mse': float(final_current_test_mse),
                'data_size': int(len(current_data))
            },
            'experiment_config': {
                'target_size': int(self.config.target_size),
                'feature_method': self.config.feature_method,
                'loss_type': self.config.loss_type,
                'iterations': int(iteration),
                'samples_per_iteration': int(self.config.samples_per_iteration)
            }
        }
        
        return current_indices, current_data, results

