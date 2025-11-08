# -*- coding: utf-8 -*-
"""
主程序 - ELM主动学习在CIFAR100上的实验
使用方式:
    python main.py --target_size 1000 --feature_method resnet --loss_type pearson
"""
import os
import json
import numpy as np
from config import Config
from data_loader import CIFAR100Loader
from feature_extractor import FeatureExtractor
from model_trainer import ELMTrainer
from active_learner import ActiveLearner
from utils import set_seed


def main():

    config = Config()
    

    set_seed(config.seed)
    

    data_loader = CIFAR100Loader(
        data_root=config.data_root,
        test_data_root=config.test_data_root
    )
    all_train_data, all_train_labels, test_data, test_labels = data_loader.load_data()
    
    feature_kwargs = {}
    if config.feature_method == 'pca':
        feature_kwargs['n_components'] = config.pca_components
        feature_kwargs['random_state'] = config.seed
    elif config.feature_method == 'tsne':
        feature_kwargs['n_components'] = config.tsne_components
        feature_kwargs['random_state'] = config.seed
    elif config.feature_method == 'resnet':
        feature_kwargs['batch_size'] = 256
    
    feature_extractor = FeatureExtractor(
        method=config.feature_method,
        **feature_kwargs
    )
    
    model_trainer = ELMTrainer(config)
    
    active_learner = ActiveLearner(
        config=config,
        data_loader=data_loader,
        feature_extractor=feature_extractor,
        model_trainer=model_trainer
    )
    
    current_indices, current_data, results = active_learner.run(
        all_train_data, all_train_labels, test_data
    )
    
    if config.save_indices:
        indices_file = os.path.join(
            config.output_dir,
            f'elm_selected_indices_cifar100_{config.target_size}.npy'
        )
        np.save(indices_file, current_indices)
        print(f"\n已保存选择的样本索引到 '{indices_file}'")
        print(f"保存的索引数量: {len(current_indices)}")
    
    if config.save_results:
        results_file = os.path.join(config.output_dir, 'elm_al_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"已保存实验结果到 '{results_file}'")
        
        # 同时保存配置
        config_file = os.path.join(config.output_dir, 'config.json')
        config.save_config(config_file)
        print(f"已保存配置到 '{config_file}'")
    
    print("ELM主动学习测试完成！")


if __name__ == "__main__":
    main()

