import torch
import numpy as np
from scipy.linalg import orth
import random
from SFLM.mapminmax import MapMinMax
import torch.nn.functional as F

def set_seed(seed=42):
    """设置所有随机数生成器的种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print(f"随机种子已设置为: {seed}")


class ELMAutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation='sigmoid', C=1e6, epochs=3, type_=1,seed =42):
        super().__init__()
        set_seed(seed)
        # 参数重命名（保持MATLAB参数语义）
        self.epochs = epochs          # 原kkkk参数
        self.C = C                    # 正则化系数
        self.type = type_             # 类型标记
        self.a = 1e-7                 # MATLAB中的小量偏移
        self.activation = activation
        # 编码器组件
        self.encoder = _ELMEncoder(input_dim, hidden_dim, activation, type_)
        
        # 解码器组件
        self.decoder = _ELMDecoder(hidden_dim, input_dim,type_,activation)
        #归一化组件
        self.scaler = MapMinMax(y_min=-1.0, y_max=1.0)
        self.ps = None
    def fit(self, X):
        X = torch.FloatTensor(X)
        H = None 
        
        # 主训练循环（对应MATLAB的kkkk循环）
        for epoch in range(self.epochs):
            # print(f"Epoch {epoch+1}/{self.epochs}")
            
            # 如果不是第一个epoch，将上一轮的输出权重转移到输入权重
            if epoch > 0:
                # 获取上一轮的输出权重
                output_weights = self.decoder.YYM  # 解码器权重
                
                # 确保权重维度匹配
                # 输入权重IW形状为 [hidden_dim, input_dim]
                # 输出权重YYM形状为 [hidden_dim, input_dim]或需要调整
                print(f"解码器权重形状: {output_weights.shape}, 编码器权重形状: {self.encoder.IW.shape}")
                
                # 确保形状匹配
                with torch.no_grad():
                    if output_weights.shape == self.encoder.IW.shape:
                        # 形状相同，直接复制
                        self.encoder.IW.copy_(output_weights)
                    elif output_weights.shape[1] == self.encoder.IW.shape[0] and output_weights.shape[0] == self.encoder.IW.shape[1]:
                        # 需要转置
                        self.encoder.IW.copy_(output_weights.T)

        
                    # 使用decoder.BB更新编码器偏置
                    if hasattr(self.decoder, 'BB') and self.decoder.BB is not None:
                        # 如果BB是标量，扩展为与Bias相同形状
                        if isinstance(self.decoder.BB, torch.Tensor):
                            if self.decoder.BB.dim() == 0:  # 标量张量
                                bias_value = self.decoder.BB.item()
                            else:
                                # 如果是多维张量，可能需要取平均或其他处理
                                bias_value = self.decoder.BB.mean().item()
                        else:
                            # 非张量情况
                            bias_value = float(self.decoder.BB)
                        
                        # 创建新的偏置张量
                        new_bias = torch.full_like(self.encoder.Bias, bias_value)
                        self.encoder.Bias.copy_(new_bias)
                        print(f"已更新编码器偏置，值: {bias_value}")
            
            # === 编码阶段 ===
            H = self.encoder(X)  # [batch_size, hidden]
            
            # === 目标构建===
            Y = X.T + self.a

            # 使用mapminmax替代MapMinMaxScalerTorch
            Y_normalized = self.scaler.fit_transform(Y)
            self.ps = self.scaler.get_params()
            if self.type == 0:
                Y4 = self.compute_inverse_avtivation_function(self.activation,Y_normalized)
            else:
                Y4= Y_normalized

            # === 解码器权重计算）===
            self.decoder.compute_weights(H.T, Y4, self.C)
            
            # === 隐层更新）===
            # if epoch < self.epochs-1:  # 最后一次迭代不需要更新
            #     H = self.decoder(H)     # 使用当前权重重建隐层
            #     H = self.encoder.normalize(H.T).T  # 重新归一化

    def predict(self, X):
        """预测过程"""
        with torch.no_grad():
            H = self.encoder(torch.FloatTensor(X))
            normalized_data = self.decoder(H)  # 这里直接使用解码器
            original_data = self.scaler.inverse_transform(normalized_data).T

            # return (self.decoder(H) - self.a).numpy()
            return original_data.numpy()

    def _mapminmax(self, x, y_min, y_max):
        """MATLAB的mapminmax精确实现"""
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        return (x - x_min) * (y_max - y_min) / (x_max - x_min + 1e-8) + y_min

    def compute_inverse_avtivation_function(self,ActivationFunction,Y2):
        """
        根据给定的激活函数类型，计算其逆函数值。
        
        Args:
            ActivationFunction (str): 激活函数类型，如'sigmoid'或'sine'。
            Y2 (float or array): 输入值。
            
        Returns:
            np.ndarray: 逆函数值。
        """
        if isinstance(ActivationFunction, str) and type(ActivationFunction) == 'str':
            ActivationFunction = str.lower(ActivationFunction)
        
       
        # 这里的 `type` 应该是参数，需要明确其含义
        # 假设 `type` 是一个标志位，表示是否需要计算逆函数
        if ActivationFunction in ['sig', 'sigmoid']:
            Y4 = -np.log((1. / Y2) - 1)
        elif ActivationFunction in ['sin', 'sine']:
            # Y4 = np.asin(Y2)
            Y4 = torch.asin(Y2)
        else:
            raise ValueError("未支持的激活函数类型：{}".format(ActivationFunction))
        
        # 确保结果是实数
        Y4 = np.real(Y4)
        
        return Y4


    
class _ELMEncoder(torch.nn.Module):
    """编码器模块"""
    def __init__(self, input_dim, hidden_dim, activation, type_):
        super().__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.type = type_
        
        # 权重矩阵维度 [hidden, input]
        self.IW = torch.nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.Bias = torch.nn.Parameter(torch.Tensor(hidden_dim, 1))
        self._orth_init()

    def _orth_init(self):
        with torch.no_grad():
            iw_np = np.random.rand(self.hidden_dim, self.input_dim) * 2 - 1
            
             # 正交化逻辑
            if self.hidden_dim > self.input_dim:
                # MATLAB: IW = orth(IW')'
                Q = orth(iw_np).astype(np.float32)
            else:
                # MATLAB: IW = orth(IW)
                Q = orth(iw_np.T).T.astype(np.float32)
            
            self.IW.data = torch.from_numpy(Q)
              
            # Bias初始化
            bias_np = orth(np.random.rand(self.hidden_dim, 1)).astype(np.float32)
            self.Bias.data = torch.from_numpy(bias_np)

    def forward(self, x):
        """前向传播"""
        # x shape: [batch_size, input_dim]
        H = torch.mm(self.IW, x.T)  # [hidden, batch]
        H = H - self.Bias  # [hidden, batch]
        
        # 激活函数、    
        if self.type == 1:
            if self.activation.lower() in ['sig', 'sigmoid']:
                H = 1 / (1 + torch.exp(-H))
            elif self.activation.lower() in ['sin', 'sine']:
                H = torch.sin(H)
         
        # 归一化（MATLAB第109行）
        H = self.normalize(H.T)
        return H# [batch, hidden]

    def normalize(self, H):
        # 假设H的维度为[特征数, 样本数]，与Matlab一致
        min_val = H.min(dim=0, keepdim=True)[0]  # 按列取最小值（对应Matlab行）
        max_val = H.max(dim=0, keepdim=True)[0]
        return 2 * (H - min_val) / (max_val - min_val + 1e-8) - 1

    


class _ELMDecoder(torch.nn.Module):
    """解码器模块"""
    def __init__(self, hidden_dim, output_dim,type,activefunction):
        super().__init__()
        self.YYM = None  # 输出权重
        self.BB = None    # 偏置项
        self.type = type
        self.act_f = activefunction

    def compute_weights(self, H, Y, C):
        """权重解析解计算"""
        # H shape: [hidden, batch]
        # Y shape: [input_dim, batch]
        m = H.size(0)  # 获取 H 的行数

        # 构造矩阵 A = (I / C) + H @ H^T
        I = torch.eye(m, device=H.device, dtype=H.dtype)
        A = I / C + H @ H.T

        # 构造右侧矩阵 B = H @ Y^T
        B = H @ Y.T

        # 解线性方程组 A * YYM = B
        try:
            self.YYM = torch.linalg.solve(A, B)  # 直接求解
        except:
            # 若矩阵 A 不可逆，使用伪逆
            self.YYM = torch.linalg.lstsq(A, B).solution

        # 计算重建隐层与目标隐层的差异，作为偏置
        yjx = H.T @ self.YYM

        bb1 = Y.shape[1]
        difference = yjx - Y.T
        bb2 = torch.sum(difference, dim=0)  # 沿第一个维度求和，与MATLAB的sum行为一致

        self.BB = bb2 / bb1
        if self.BB.numel() > 1:  # 如果BB是多元素tensor
            self.BB = self.BB[0]  # 取第一个元素，对应MATLAB中的BB = BB(1)
        
        # 打印调试信息
        # print(f"解码器权重YYM形状: {self.YYM.shape}, 偏置BB: {self.BB.item() if isinstance(self.BB, torch.Tensor) else self.BB}")

    def inverse_mapminmax(self, y, y_min, y_max, x_min, x_max):
        """反归一化函数，将归一化后的数据还原"""
        return (y - y_min) * (x_max - x_min) / (y_max - y_min + 1e-8) + x_min

    def forward(self, H):
        """重建过程"""
        # 将self.BB从标量转换为适合广播的形状
        if isinstance(self.BB, torch.Tensor) and self.BB.dim() == 0:
            # 如果self.BB是标量张量，将其扩展为适合广播的形状
            BB_reshaped = self.BB.expand(self.YYM.shape[1])
        else:
            # 确保BB是一个行向量，以便进行广播
            BB_reshaped = self.BB.view(1, -1)
        
        # 实现等效于MATLAB中的bsxfun(@minus,A',BB.')'
        # 在PyTorch中，这相当于 (H @ self.YYM - BB_reshaped)
        GXZ111 = H @ self.YYM - BB_reshaped

        # 激活函数处理
        if self.type == 0:  # 使用激活函数
            if self.act_f.lower() in ['sig', 'sigmoid']:
                GXZ2 = 1 / (1 + torch.exp(-GXZ111.T))
            elif self.act_f.lower() in ['sin', 'sine']:
                GXZ2 = torch.sin(GXZ111.T)
        else:  # 无激活函数
            GXZ2 = GXZ111.T

        return GXZ2
  




# class MapMinMaxScalerTorch:
#     def __init__(self, y_min=-1, y_max=1):
#         if y_max <= y_min:
#             raise ValueError("y_max must be greater than y_min.")
#         self.y_min = y_min
#         self.y_max = y_max
#         self.x_min = None
#         self.x_max = None

#     def fit(self, X: torch.Tensor):
#         if not isinstance(X, torch.Tensor):
#             X = torch.tensor(X)
#         self.x_min = X.min(dim=0).values  # 沿第0维度（列）取最小值
#         self.x_max = X.max(dim=0).values  # 沿第0维度（列）取最大值
#         return self

#     def transform(self, X: torch.Tensor) -> torch.Tensor:
#         if self.x_min is None or self.x_max is None:
#             raise ValueError("Scaler has not been fitted yet.")
#         if not isinstance(X, torch.Tensor):
#             X = torch.tensor(X)
#         # 处理除零情况（最大值等于最小值时缩放因子为0）
#         scale = (self.y_max - self.y_min) / (self.x_max - self.x_min + 1e-8)
#         # 归一化计算（自动广播维度）
#         normalized = self.y_min + (X - self.x_min) * scale
#         # 对常数列特殊处理（保持 y_min）
#         normalized[:, self.x_max == self.x_min] = self.y_min
#         return normalized

#     def inverse_transform(self, X_normalized: torch.Tensor) -> torch.Tensor:
#         if self.x_min is None or self.x_max is None:
#             raise ValueError("Scaler has not been fitted yet.")
#         if not isinstance(X_normalized, torch.Tensor):
#             X_normalized = torch.tensor(X_normalized)
#         # 计算逆缩放因子 n  
#         scale = (self.x_max - self.x_min) / (self.y_max - self.y_min + 1e-8)
#         # 反归一化计算（自动广播维度）
#         original = self.x_min + (X_normalized - self.y_min) * scale
#         return original

#     def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
#         return self.fit(X).transform(X)

#     def to(self, device: str): 
#         """将参数移动到指定设备"""
#         self.x_min = self.x_min.to(device)
#         self.x_max = self.x_max.to(device) 
#         return self
