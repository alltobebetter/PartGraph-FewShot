import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.pos_embed import build_position_encoding

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        # 改进的初始化：每个 slot 有独立的初始化
        self.slots_mu = nn.Parameter(torch.zeros(1, num_slots, dim))
        self.slots_sigma = nn.Parameter(torch.zeros(1, num_slots, dim))
        
        # Xavier 初始化，让不同 slot 有不同的起点
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_sigma)
        self.slots_sigma.data = self.slots_sigma.data.abs() + 0.1  # 确保正值

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

    def forward(self, inputs, num_slots=None):
        """
        inputs: (B, N, D)  where N = H*W
        """
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        # 改进：使用学习到的初始化，加小噪声
        mu = self.slots_mu.expand(b, -1, -1)
        sigma = self.slots_sigma.expand(b, -1, -1)
        noise = torch.randn(b, n_s, d, device=inputs.device) * 0.1
        slots = mu + sigma * noise

        inputs = self.norm_input(inputs)        
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)

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

        return slots, attn

class PartAwareSlotAttention(nn.Module):
    """
    Wrapper that handles 2D feature maps and positional encoding.
    """
    def __init__(self, num_slots, in_channels, slot_dim, iters=3, resolution=(14, 14)):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.slot_dim = slot_dim
        
        self.slot_attn = SlotAttention(
            num_slots=num_slots,
            dim=slot_dim,
            iters=iters,
            hidden_dim=slot_dim * 2
        )
        
        # 1x1 conv to map input channels to slot dim if necessary
        if in_channels != slot_dim:
            self.conv1x1 = nn.Linear(in_channels, slot_dim)
        else:
            self.conv1x1 = nn.Identity()
            
        # Positional encoding
        self.register_buffer(
            "pos_embed", 
            build_position_encoding(resolution[0], slot_dim).unsqueeze(0) # (1, H*W, D)
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        b, c, h, w = x.shape
        
        # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(b, h*w, c)
        
        # Project to slot_dim
        x = self.conv1x1(x)
        
        # Add positional encoding
        # Note: We assume input resolution matches the pre-computed pos_embed
        # If not, we might need to interpolate, but for MVP let's assume fixed size.
        if x.shape[1] == self.pos_embed.shape[1]:
             x = x + self.pos_embed
        
        slots, attn = self.slot_attn(x)
        
        # Reshape attention back to 2D: (B, K, H*W) -> (B, K, H, W)
        attn = attn.view(b, self.slot_attn.num_slots, h, w)
        
        return slots, attn
