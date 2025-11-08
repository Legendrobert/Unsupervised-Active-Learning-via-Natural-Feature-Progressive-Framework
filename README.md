# Unsupervised-Active-Learning-via-Natural-Feature-Progressive-Framework

Official implementation of our paper "Unsupervised Active Learning via Natural Feature Progressive Framework", currently under review at IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

## 文件结构

程序已重构为模块化结构，包含以下文件：

```
├── main.py                 # 主程序入口
├── config.py              # 配置管理模块（argparse参数处理）
├── data_loader.py         # 数据加载模块
├── feature_extractor.py   # 特征提取模块（ResNet/PCA/t-SNE）
├── model_trainer.py       # 模型训练模块（ELM训练和评估）
├── active_learner.py      # 主动学习模块（主动学习流程）
├── utils.py               # 工具函数模块
└── README.md              # 本说明文件
```

## 使用方法

### 基本使用

```bash
python main.py
```

### 使用命令行参数

```bash
# 设置目标数据集大小
python main.py --target_size 1000

# 选择特征提取方法
python main.py --feature_method resnet  # 或 pca, tsne, raw

# 选择损失类型
python main.py --loss_type pearson  # 或 mse

# 组合多个参数
python main.py --target_size 2000 --feature_method pca --loss_type mse --pool_size 2000
```

## 使用方法
设置好config, 数据集路径. 开选.
选完根据保存的索引数据使用下游分类器测试数据性能.

(无监督学习数据均衡性是一个提升创新点)
