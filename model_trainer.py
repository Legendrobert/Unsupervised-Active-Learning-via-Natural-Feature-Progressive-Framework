# -*- coding: utf-8 -*-
"""
模型训练模块 - 封装ELM模型训练和评估
"""
import numpy as np
import torch
import time
from SFLM.SFLM import ELMAutoEncoder
from scipy.stats import pearsonr


class ELMTrainer:
    """ELM模型训练器"""
    
    def __init__(self, config):
        """
        初始化训练器
        Args:
            config: 配置对象
        """
        self.config = config
    
    def create_model(self, hidden_dim, activation=None, C=None, type_=None, epochs=None):
        """
        创建ELM模型
        Args:
            hidden_dim: 隐藏层维度
            activation: 激活函数（如果为None，使用config中的值）
            C: 正则化系数（如果为None，使用config中的值）
            type_: 模型类型（如果为None，使用config中的值）
            epochs: 训练轮数（如果为None，使用config中的值）
        Returns:
            model: ELM模型
        """
        return ELMAutoEncoder(
            input_dim=self.config.input_dim,
            hidden_dim=hidden_dim,
            activation=activation or self.config.activation,
            C=C or self.config.C,
            epochs=epochs or self.config.epochs,
            type_=type_ if type_ is not None else self.config.type_,
            seed=self.config.seed
        )
    
    def train_model(self, model, data, model_name="模型"):
        """
        训练模型
        Args:
            model: ELM模型
            data: 训练数据
            model_name: 模型名称
        Returns:
            final_mse: 最终MSE
            sample_mse: 每个样本的MSE
        """
        print(f"正在训练 {model_name}...")
        start_time = time.time()
        model.fit(data)
        training_time = time.time() - start_time
        
        reconstructed = model.predict(data)
        sample_mse = np.mean((data - reconstructed) ** 2, axis=1)
        final_mse = np.mean((data - reconstructed) ** 2)
        
        print(f"{model_name} 训练完成，耗时: {training_time:.2f}秒，MSE: {final_mse:.6f}")
        return final_mse, sample_mse
    
    def compute_loss_differences(self, ref_sample_mse, current_model, pool_data, pool_indices, loss_type='mse'):
        """
        计算损失差值
        Args:
            ref_sample_mse: 参考模型在每个样本上的MSE
            current_model: 当前模型
            pool_data: 候选池数据
            pool_indices: 候选池索引
            loss_type: 损失类型 ('mse', 'pearson')
        Returns:
            loss_differences: 损失差值数组
            ref_losses: 参考模型的损失数组
            current_losses: 当前模型的损失数组
        """
        print(f"正在计算ELM {loss_type.upper()}差值...")
        
        if isinstance(pool_data, torch.Tensor):
            pool_data_np = pool_data.cpu().numpy()
        else:
            pool_data_np = pool_data
        
        if loss_type.lower() == 'mse':
            ref_losses = ref_sample_mse[pool_indices]
            
            current_reconstructed = current_model.predict(pool_data_np)
            current_losses = np.mean((pool_data_np - current_reconstructed) ** 2, axis=1)
            
            loss_differences = current_losses - ref_losses
            
        elif loss_type.lower() == 'pearson':
            current_reconstructed = current_model.predict(pool_data_np)
            
            num_samples = len(pool_data_np)
            pearson_scores = np.zeros(num_samples)
            
            for i in range(num_samples):
                original = pool_data_np[i].flatten()
                reconstructed = current_reconstructed[i].flatten()
                
                try:
                    correlation, _ = pearsonr(original, reconstructed)
                    if np.isnan(correlation):
                        correlation = 0.0
                    pearson_scores[i] = correlation
                except:
                    pearson_scores[i] = 0.0
            
            current_losses = 1.0 - pearson_scores
            
            ref_mse_normalized = ref_sample_mse[pool_indices]
            if ref_mse_normalized.max() > ref_mse_normalized.min():
                ref_losses = (ref_mse_normalized - ref_mse_normalized.min()) / (ref_mse_normalized.max() - ref_mse_normalized.min())
            else:
                ref_losses = np.zeros_like(ref_mse_normalized)
            
            loss_differences = current_losses - ref_losses
            
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}. 支持的类型: 'mse', 'pearson'")
        
        print(f"ELM {loss_type.upper()}差值计算完成，共处理 {len(pool_data_np)} 个样本")
        
        if loss_type.lower() == 'pearson':
            print(f"参考损失范围: [{ref_losses.min():.6f}, {ref_losses.max():.6f}]")
            print(f"当前模型Pearson损失范围: [{current_losses.min():.6f}, {current_losses.max():.6f}]")
            print(f"Pearson损失差值范围: [{loss_differences.min():.6f}, {loss_differences.max():.6f}]")
        else:
            print(f"参考模型MSE范围: [{ref_losses.min():.6f}, {ref_losses.max():.6f}]")
            print(f"当前模型MSE范围: [{current_losses.min():.6f}, {current_losses.max():.6f}]")
            print(f"MSE差值范围: [{loss_differences.min():.6f}, {loss_differences.max():.6f}]")
        
        return loss_differences, ref_losses, current_losses
    
    def evaluate_model(self, model, test_data):
        """
        评估模型
        Args:
            model: ELM模型
            test_data: 测试数据
        Returns:
            test_mse: 测试集MSE
        """
        test_reconstructed = model.predict(test_data)
        test_mse = np.mean((test_data - test_reconstructed) ** 2)
        return test_mse
    
    @staticmethod
    def print_loss_results(pool_indices, loss_differences, ref_losses, current_losses, 
                          pool_labels, loss_type='mse', top_k=10):
        """
        打印损失差值结果
        Args:
            pool_indices: 候选池索引
            loss_differences: 损失差值数组
            ref_losses: 参考模型损失数组
            current_losses: 当前模型损失数组
            pool_labels: 候选池标签
            loss_type: 损失类型
            top_k: 显示前k个结果
        """
        results = []
        for i, (idx, diff, ref_loss, curr_loss, label) in enumerate(
            zip(pool_indices, loss_differences, ref_losses, current_losses, pool_labels)):
            results.append({
                'index': idx,
                'difference': diff,
                'ref_loss': ref_loss,
                'current_loss': curr_loss,
                'label': label,
                'class_num': int(label)
            })
        
        results.sort(key=lambda x: x['difference'], reverse=True)
        
        loss_name = loss_type.upper()
        print(f"\nTop {top_k} Largest {loss_name} Differences ({loss_name}_current - {loss_name}_ref):")
        print("Index\t\tClass\t\t{}_ref\t\t{}_current\t\tDifference".format(loss_name, loss_name))
        
        print("-" * 80)
        for i in range(min(top_k, len(results))):
            result = results[i]
            print(f"{result['index']}\t\t{result['class_num']:<6}\t\t{result['ref_loss']:.8f}\t{result['current_loss']:.8f}\t{result['difference']:.8f}")
        
        print(f"\nELM {loss_name} Difference Statistics:")
        print(f"Mean difference: {np.mean(loss_differences):.6f}")
        print(f"Std difference: {np.std(loss_differences):.6f}")
        print(f"Max difference: {np.max(loss_differences):.6f}")
        print(f"Min difference: {np.min(loss_differences):.6f}")
        
        print(f"Positive differences (current > ref): {np.sum(loss_differences > 0)}/{len(loss_differences)}")
        print(f"Negative differences (current < ref): {np.sum(loss_differences < 0)}/{len(loss_differences)}")
        
        print(f"\nClass Distribution in Top {top_k}:")
        class_counts = {}
        for i in range(min(top_k, len(results))):
            class_num = results[i]['class_num']
            class_counts[class_num] = class_counts.get(class_num, 0) + 1
        
        for class_num, count in sorted(class_counts.items()):
            print(f"类别{class_num}: {count}")

