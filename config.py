# -*- coding: utf-8 -*-
"""
配置管理模块 - 使用argparse处理命令行参数
"""
import argparse
import os
import torch


class Config:
    """配置类，管理所有实验参数"""
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='ELM主动学习在CIFAR100上的实验')
        self._add_arguments()
        self.args = self.parser.parse_args()
        self._validate_args()
    
    def _add_arguments(self):
        """添加所有命令行参数"""
        self.parser.add_argument('--data_root', type=str, 
                                default='/home/lsk/01-Project/Main_data/',
                                help='CIFAR100数据集根目录')
        self.parser.add_argument('--test_data_root', type=str,
                                default='/home/lsk/01-Project/08-Attention_AL/Attention_AL/data',
                                help='测试集数据根目录')
        
        self.parser.add_argument('--input_dim', type=int, default=3072,
                                help='输入维度 (CIFAR100: 3*32*32=3072)')
        self.parser.add_argument('--ref_hidden_dim', type=int, default=1800,
                                help='参考模型隐藏层维度')
        self.parser.add_argument('--current_hidden_dim_base', type=int, default=10,
                                help='当前模型隐藏层基础维度 (实际为 base + iteration)')
        self.parser.add_argument('--final_hidden_dim', type=int, default=1000,
                                help='最终模型隐藏层维度')
        self.parser.add_argument('--activation', type=str, default='sigmoid',
                                choices=['sigmoid', 'sine'],
                                help='激活函数类型')
        self.parser.add_argument('--C', type=float, default=1e-2,
                                help='正则化系数')
        self.parser.add_argument('--epochs', type=int, default=1,
                                help='训练轮数')
        self.parser.add_argument('--type_', type=int, default=1,
                                choices=[0, 1],
                                help='模型类型 (0: 使用激活函数, 1: 不使用)')
        
        self.parser.add_argument('--target_size', type=int, default=1000,
                                help='目标数据集大小')
        self.parser.add_argument('--initial_samples_ratio', type=float, default=0.3,
                                help='初始样本比例 (相对于target_size)')
        self.parser.add_argument('--samples_per_iteration', type=int, default=10,
                                help='每次迭代选择的样本数')
        self.parser.add_argument('--pool_size', type=int, default=1500,
                                help='候选池大小')
        
        self.parser.add_argument('--feature_method', type=str, default='resnet',
                                choices=['resnet', 'pca', 'tsne', 'raw'],
                                help='特征提取方法')
        self.parser.add_argument('--pca_components', type=int, default=512,
                                help='PCA主成分数量')
        self.parser.add_argument('--tsne_components', type=int, default=50,
                                help='t-SNE降维维度')
        
        self.parser.add_argument('--loss_type', type=str, default='pearson',
                                choices=['mse', 'pearson'],
                                help='损失类型')
        
        self.parser.add_argument('--n_clusters', type=int, default=100,
                                help='K-means聚类数量 (CIFAR100类别数)')
        self.parser.add_argument('--kmeans_n_init', type=int, default=10,
                                help='K-means初始化次数')
        self.parser.add_argument('--kmeans_max_iter', type=int, default=300,
                                help='K-means最大迭代次数')
        
        self.parser.add_argument('--seed', type=int, default=42,
                                help='随机种子')
        self.parser.add_argument('--device', type=str, default='auto',
                                choices=['auto', 'cuda', 'cpu'],
                                help='计算设备')
        
        self.parser.add_argument('--output_dir', type=str, default='Results',
                                help='结果输出目录')
        self.parser.add_argument('--save_indices', action='store_true', default=True,
                                help='是否保存选择的样本索引')
        self.parser.add_argument('--save_results', action='store_true', default=True,
                                help='是否保存实验结果')
        self.parser.add_argument('--print_interval', type=int, default=10,
                                help='打印详细结果的迭代间隔')
        self.parser.add_argument('--top_k', type=int, default=3,
                                help='打印前k个结果')
        
        self.parser.add_argument('--test_subset_size', type=int, default=200,
                                help='测试集子集大小（用于评估）')
    
    def _validate_args(self):
        """验证参数有效性"""
        if self.args.target_size <= 0:
            raise ValueError("target_size必须大于0")
        if self.args.initial_samples_ratio <= 0 or self.args.initial_samples_ratio >= 1:
            raise ValueError("initial_samples_ratio必须在(0,1)之间")
        if self.args.samples_per_iteration <= 0:
            raise ValueError("samples_per_iteration必须大于0")
        if self.args.pool_size <= 0:
            raise ValueError("pool_size必须大于0")
        
        os.makedirs(self.args.output_dir, exist_ok=True)
    
    def get_device(self):
        """获取计算设备"""
        if self.args.device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.args.device
    
    def to_dict(self):
        """将配置转换为字典"""
        return vars(self.args)
    
    def save_config(self, filepath):
        """保存配置到文件"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __getattr__(self, name):
        """允许通过config.param_name访问参数"""
        return getattr(self.args, name)

