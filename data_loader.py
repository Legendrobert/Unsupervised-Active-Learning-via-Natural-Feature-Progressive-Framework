# -*- coding: utf-8 -*-
"""
数据加载模块 - 封装CIFAR100数据加载
"""
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


class CIFAR100Loader:
    """CIFAR100数据加载器"""
    
    def __init__(self, data_root, test_data_root=None):
        """
        初始化数据加载器
        Args:
            data_root: 训练数据根目录
            test_data_root: 测试数据根目录（如果为None，使用data_root）
        """
        self.data_root = data_root
        self.test_data_root = test_data_root or data_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def load_data(self):
        """
        加载CIFAR100数据集
        Returns:
            train_data: 训练数据 [50000, 3072]
            train_labels: 训练标签 [50000]
            test_data: 测试数据 [10000, 3072]
            test_labels: 测试标签 [10000]
        """
        print("正在加载CIFAR100数据集...")
        
        train_set = torchvision.datasets.CIFAR100(
            root=self.data_root,
            train=True,
            download=True,
            transform=self.transform
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=len(train_set),
            shuffle=False
        )
        
        full_data, full_labels = next(iter(train_loader))
        train_data = full_data.numpy().reshape(len(train_set), -1)
        train_labels = full_labels.numpy()
        
        test_set = torchvision.datasets.CIFAR100(
            root=self.test_data_root,
            train=False,
            download=True,
            transform=self.transform
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=len(test_set),
            shuffle=False
        )
        
        test_data, test_labels = next(iter(test_loader))
        test_data = test_data.numpy().reshape(len(test_set), -1)
        test_labels = test_labels.numpy()
        
        print(f"训练数据形状: {train_data.shape}")
        print(f"测试数据形状: {test_data.shape}")
        print(f"CIFAR100总类别数: 100")
        
        return train_data, train_labels, test_data, test_labels
    
    @staticmethod
    def get_labels_from_data(all_labels, indices):
        """
        从标签数组中获取指定索引的标签
        Args:
            all_labels: 所有标签
            indices: 样本索引数组
        Returns:
            labels: 对应的标签数组
        """
        return all_labels[indices]
    
    @staticmethod
    def print_class_distribution(data_indices, all_labels, title="数据集"):
        """
        打印数据集的类别分布
        Args:
            data_indices: 数据索引
            all_labels: 所有标签
            title: 标题
        """
        labels = CIFAR100Loader.get_labels_from_data(all_labels, data_indices)
        class_counts = {}
        
        for label in labels:
            class_num = int(label)
            class_counts[class_num] = class_counts.get(class_num, 0) + 1
        
        print(f"\n{title}类别分布 (总计 {len(data_indices)} 个样本):")
        print("-" * 50)
        
        for class_num in sorted(class_counts.keys()):
            count = class_counts[class_num]
            percentage = (count / len(data_indices)) * 100 if len(data_indices) > 0 else 0
            print(f"类别{class_num:<3}: {count:>4} 个样本 ({percentage:>5.1f}%)")

