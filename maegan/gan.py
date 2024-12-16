import numpy as np
import torch.nn as nn


"""
The code is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


class Generator(nn.Module):
    def __init__(self, input_dim, latent_dim, num_blocks=5):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 计算每个 block 的维度，从 latent_dim 到 input_dim 的线性过渡
        block_dims = self._calculate_block_dims(latent_dim, input_dim, num_blocks)
        print(block_dims)

        layers = []
        # 构建第一个 block，设置 normalize=False
        layers.extend(self._block(block_dims[0], block_dims[1], normalize=False))
        
        # 构建剩余的 blocks，使用 normalize=True
        for i in range(1, num_blocks):
            in_feat = block_dims[i]
            out_feat = block_dims[i + 1]
            layers.extend(self._block(in_feat, out_feat, normalize=True))

        # 最终输出层
        layers.append(nn.Linear(block_dims[-1], self.input_dim))
        layers.append(nn.Tanh())  # 输出范围在 [-1, 1]

        # 将所有层合并为一个 Sequential 模型
        self.model = nn.Sequential(*layers)

    def _block(self, in_feat, out_feat, normalize=True):
        """单个 block，包含 Linear、BatchNorm、LeakyReLU"""
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def _calculate_block_dims(self, latent_dim, input_dim, num_blocks):
        """计算每个 block 的输入和输出维度"""
        # 使用线性插值生成从 latent_dim 到 input_dim 的均匀间隔，并确保它们为整数
        step = (input_dim - latent_dim) / num_blocks
        return [int(latent_dim + i * step) for i in range(num_blocks + 1)]
    
    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self, input_dim, num_blocks=3, min_dim=1):
        super().__init__()
        self.input_dim = input_dim
        
        # 计算每个 block 的维度，从 input_dim 逐渐减小到 min_dim
        block_dims = self._calculate_block_dims(input_dim, min_dim, num_blocks)
        print(block_dims)

        layers = []
        # 构建每个 block
        for i in range(num_blocks):
            in_feat = block_dims[i]
            out_feat = block_dims[i + 1]
            layers.extend(self._block(in_feat, out_feat))

        # 将所有层合并为一个 Sequential 模型
        self.features = nn.Sequential(*layers)

        # 最终线性层，输出维度为 1
        self.last_layer = nn.Sequential(
            nn.Linear(block_dims[-1], 1)
        )

    def _block(self, in_feat, out_feat):
        """单个 block，包含 Linear 和 LeakyReLU"""
        layers = [nn.Linear(in_feat, out_feat), nn.LeakyReLU(0.2, inplace=True)]
        return layers

    def _calculate_block_dims(self, input_dim, min_dim, num_blocks):
        """计算每个 block 的输入和输出维度"""
        # 生成一个从 input_dim 到 min_dim 的均匀间隔
        return [input_dim] + [int(input_dim - (i + 1) * (input_dim - min_dim) / num_blocks) for i in range(num_blocks)]

    def forward(self, img):
        features = self.forward_features(img)
        validity = self.last_layer(features)
        return validity

    def forward_features(self, img):
        features = self.features(img)
        return features


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_blocks=5):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 计算每个 block 的维度，从 input_dim 逐渐减小到 latent_dim
        block_dims = self._calculate_block_dims(input_dim, latent_dim, num_blocks)
        print("Block dimensions:", block_dims)
        
        layers = []
        # 构建每个 block
        for i in range(num_blocks):
            in_feat = block_dims[i]
            out_feat = block_dims[i + 1]
            layers.extend(self._block(in_feat, out_feat))

        # 最后一层，映射到 latent_dim 并使用 Tanh 激活
        layers.append(nn.Linear(block_dims[-1], self.latent_dim))
        layers.append(nn.Tanh())  # 输出范围在 [-1, 1]

        # 将所有层合并为一个 Sequential 模型
        self.model = nn.Sequential(*layers)

    def _block(self, in_feat, out_feat):
        """单个 block，包含 Linear 和 LeakyReLU"""
        layers = [nn.Linear(in_feat, out_feat), nn.LeakyReLU(0.2, inplace=True)]
        return layers

    def _calculate_block_dims(self, input_dim, latent_dim, num_blocks):
        """计算每个 block 的输入和输出维度"""
        # 确保维度递减到 latent_dim，避免浮点数精度误差
        step = (input_dim - latent_dim) / num_blocks
        return [int(input_dim - i * step) for i in range(num_blocks + 1)] + [latent_dim]

    def forward(self, img):
        validity = self.model(img)
        return validity
