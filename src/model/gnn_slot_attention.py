"""
GNN-in-the-Loop Slot Attention

核心创新：在 Slot Attention 的每次迭代中插入 GNN 消息传递，
让 slot 之间交换信息，实现协作式部件发现。

原版 Slot Attention 的问题：
- 每个 slot 独立竞争像素，不知道其他 slot 在干嘛
- 可能重叠（两个 slot 都抢"头"）或遗漏（没人管"尾巴"）
- 忽略部件间的结构约束

我们的解法：
- 每次迭代后，基于 slot 的空间位置构建动态图
- 通过 GNN 消息传递让 slot 感知彼此
- 部件发现变成协作过程而非独立竞争
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.utils.pos_embed import build_position_encoding


class SlotGNNLayer(nn.Module):
    """
    轻量级 GNN 层，用于 slot 之间的消息传递
    使用 GAT 风格的注意力机制
    """
    def __init__(self, slot_dim, edge_dim=8, num_heads=4):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_heads = num_heads
        self.head_dim = slot_dim // num_heads
        
        # 多头注意力
        self.W_q = nn.Linear(slot_dim, slot_dim)
        self.W_k = nn.Linear(slot_dim, slot_dim)
        self.W_v = nn.Linear(slot_dim, slot_dim)
        
        # 边特征影响注意力权重
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_heads)
        )
        
        # 输出投影
        self.out_proj = nn.Linear(slot_dim, slot_dim)
        self.norm = nn.LayerNorm(slot_dim)
        
    def forward(self, slots, edge_attr):
        """
        slots: (B, K, D) - slot 表征
        edge_attr: (B, K, K, E) - 边特征（空间关系）
        
        返回: (B, K, D) - 更新后的 slot
        """
        B, K, D = slots.shape
        
        # 多头注意力
        Q = self.W_q(slots).view(B, K, self.num_heads, self.head_dim)
        K_mat = self.W_k(slots).view(B, K, self.num_heads, self.head_dim)
        V = self.W_v(slots).view(B, K, self.num_heads, self.head_dim)
        
        # (B, num_heads, K, K)
        attn = torch.einsum('bqhd,bkhd->bhqk', Q, K_mat) / math.sqrt(self.head_dim)
        
        # 加入边特征的偏置
        edge_bias = self.edge_encoder(edge_attr)  # (B, K, K, num_heads)
        edge_bias = edge_bias.permute(0, 3, 1, 2)  # (B, num_heads, K, K)
        attn = attn + edge_bias
        
        # 自环 mask（可选：不让 slot 给自己发消息）
        # mask = torch.eye(K, device=slots.device).bool()
        # attn = attn.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        # 聚合
        out = torch.einsum('bhqk,bkhd->bqhd', attn, V)
        out = out.reshape(B, K, D)
        out = self.out_proj(out)
        
        # 残差 + LayerNorm
        return self.norm(slots + out)


class GNNInLoopSlotAttention(nn.Module):
    """
    GNN-in-the-Loop Slot Attention
    
    在每次 Slot Attention 迭代中插入 GNN 消息传递
    """
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128, 
                 gnn_start_iter=1, edge_dim=8):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.gnn_start_iter = gnn_start_iter  # 从第几次迭代开始用 GNN
        
        # Slot 初始化（可学习，每个 slot 不同）
        self.slots_mu = nn.Parameter(torch.zeros(1, num_slots, dim))
        self.slots_sigma = nn.Parameter(torch.zeros(1, num_slots, dim))
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_sigma)
        self.slots_sigma.data = self.slots_sigma.data.abs() + 0.1

        # Slot Attention 组件
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        
        hidden_dim = max(dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        
        # GNN 层（核心创新）
        self.gnn_layer = SlotGNNLayer(dim, edge_dim=edge_dim)
        
        # 空间关系编码器
        self.edge_dim = edge_dim

    def compute_slot_positions(self, attn, h, w):
        """
        从 attention map 计算每个 slot 的空间位置（质心）
        
        attn: (B, K, N) where N = H*W
        返回: (B, K, 2) - 归一化的 (y, x) 坐标
        """
        B, K, N = attn.shape
        device = attn.device
        
        # 创建坐标网格
        y_coords = torch.linspace(0, 1, h, device=device)
        x_coords = torch.linspace(0, 1, w, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1)  # (N, 2)
        
        # 加权平均得到质心
        # attn: (B, K, N), coords: (N, 2) -> positions: (B, K, 2)
        attn_normalized = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        positions = torch.einsum('bkn,nc->bkc', attn_normalized, coords)
        
        return positions

    def compute_edge_features(self, positions, slots):
        """
        计算 slot 之间的边特征
        
        positions: (B, K, 2) - slot 位置
        slots: (B, K, D) - slot 表征
        
        返回: (B, K, K, edge_dim) - 边特征
        """
        B, K, _ = positions.shape
        
        # 空间关系
        # (B, K, 1, 2) - (B, 1, K, 2) = (B, K, K, 2)
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        
        delta = pos_j - pos_i  # 相对位置 (dx, dy)
        dist = torch.norm(delta, dim=-1, keepdim=True)  # 距离
        angle = torch.atan2(delta[..., 1:2], delta[..., 0:1])  # 角度
        
        # 语义关系（slot 表征的相似度）
        slots_norm = F.normalize(slots, dim=-1)
        similarity = torch.bmm(slots_norm, slots_norm.transpose(1, 2))  # (B, K, K)
        
        # 拼接边特征 (共 8 维)
        edge_attr = torch.cat([
            delta,                          # (B, K, K, 2) 相对位置
            dist,                           # (B, K, K, 1) 距离
            angle,                          # (B, K, K, 1) 角度
            similarity.unsqueeze(-1),       # (B, K, K, 1) 语义相似度
            (dist < 0.3).float(),           # (B, K, K, 1) 是否邻近
            (dist < 0.15).float(),          # (B, K, K, 1) 是否很近
            (dist < 0.5).float(),           # (B, K, K, 1) 是否中等距离
        ], dim=-1)  # (B, K, K, 8)
        
        return edge_attr

    def forward(self, inputs, num_slots=None, spatial_shape=None):
        """
        inputs: (B, N, D) where N = H*W
        spatial_shape: (H, W) 用于计算空间位置
        """
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        # 推断空间尺寸
        if spatial_shape is None:
            h = w = int(math.sqrt(n))
        else:
            h, w = spatial_shape
        
        # 初始化 slots
        mu = self.slots_mu.expand(b, -1, -1)
        sigma = self.slots_sigma.expand(b, -1, -1)
        noise = torch.randn(b, n_s, d, device=inputs.device) * 0.1
        slots = mu + sigma * noise

        inputs = self.norm_input(inputs)        
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        for iter_idx in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            # Slot Attention
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
            
            # GNN 消息传递（核心创新！）
            # 从第 gnn_start_iter 次迭代开始，让 slot 之间交互
            if iter_idx >= self.gnn_start_iter:
                # 计算当前 slot 的空间位置
                positions = self.compute_slot_positions(attn, h, w)
                
                # 计算边特征
                edge_attr = self.compute_edge_features(positions, slots)
                
                # GNN 消息传递
                slots = self.gnn_layer(slots, edge_attr)

        return slots, attn


class PartAwareGNNSlotAttention(nn.Module):
    """
    完整的 Part-Aware GNN-in-the-Loop Slot Attention 模块
    处理 2D 特征图，包含位置编码
    """
    def __init__(self, num_slots, in_channels, slot_dim, iters=3, 
                 resolution=(14, 14), gnn_start_iter=1):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        
        self.slot_attn = GNNInLoopSlotAttention(
            num_slots=num_slots,
            dim=slot_dim,
            iters=iters,
            hidden_dim=slot_dim * 2,
            gnn_start_iter=gnn_start_iter
        )
        
        # 通道映射
        if in_channels != slot_dim:
            self.conv1x1 = nn.Linear(in_channels, slot_dim)
        else:
            self.conv1x1 = nn.Identity()
            
        # 位置编码
        self.register_buffer(
            "pos_embed", 
            build_position_encoding(resolution[0], slot_dim).unsqueeze(0)
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        返回: slots (B, K, D), attn (B, K, H, W)
        """
        b, c, h, w = x.shape
        
        # Flatten: (B, C, H, W) -> (B, H*W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(b, h*w, c)
        
        # 投影到 slot_dim
        x = self.conv1x1(x)
        
        # 加位置编码
        if x.shape[1] == self.pos_embed.shape[1]:
            x = x + self.pos_embed
        
        # GNN-in-the-Loop Slot Attention
        slots, attn = self.slot_attn(x, spatial_shape=(h, w))
        
        # Reshape attention: (B, K, H*W) -> (B, K, H, W)
        attn = attn.view(b, self.num_slots, h, w)
        
        return slots, attn
