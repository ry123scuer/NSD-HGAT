import torch.nn as nn
import torch
from torchinfo import summary
import torch.nn.functional as F
import math
import pickle
from math import sqrt
# from torch_geometric.nn import GATv2Conv

normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"


# 修正后的混合池化层（自动适配kernel_size，保证输出维度=输入维度）
class MixedPool1d(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        # 自动计算padding，保证输出维度=输入维度（核心修正）
        self.padding = padding if padding is not None else (kernel_size - 1) // 2
        self.avg_pool = nn.AvgPool1d(kernel_size, stride, self.padding)
        self.max_pool = nn.MaxPool1d(kernel_size, stride, self.padding)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习融合权重

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        # 强制截断/填充到输入维度（兜底，避免偶发维度偏差）
        if avg_out.shape[-1] != x.shape[-1]:
            avg_out = F.pad(avg_out, (0, x.shape[-1] - avg_out.shape[-1])) if avg_out.shape[-1] < x.shape[-1] else avg_out[..., :x.shape[-1]]
            max_out = F.pad(max_out, (0, x.shape[-1] - max_out.shape[-1])) if max_out.shape[-1] < x.shape[-1] else max_out[..., :x.shape[-1]]
        # 加权融合（sigmoid保证权重在0~1）
        out = torch.sigmoid(self.alpha) * avg_out + (1 - torch.sigmoid(self.alpha)) * max_out
        return out




class TemporalDualBranch(nn.Module):
    def __init__(self, d_model, window_size=3):
        super().__init__()
        self.d_model = d_model
        self.local_window = window_size
        self.num_heads = 4  # 注意力头数（确保d_model能被其整除）
        self.head_dim = d_model // self.num_heads
        assert d_model % self.num_heads == 0, f"d_model={d_model}必须能被num_heads={self.num_heads}整除"

        # ========== 分支1：增强版局部去趋势（自定义带掩码的局部注意力） ==========
        self.window_weight = nn.Parameter(torch.ones(window_size) / window_size)
        self.local_linear = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        # 自定义注意力的线性投影层
        self.qkv = nn.Linear(d_model, d_model * 3)  # 合并Q/K/V投影
        self.out_proj = nn.Linear(d_model, d_model)

        # ========== 分支2：增强版全局趋势（卷积池化） ==========
        # self.global_conv_pool = nn.Sequential(
        #     nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, groups=d_model),
        #     nn.AdaptiveAvgPool1d(1)
        # )
        # # 修改后（滑动窗口平均池化，窗口=3，和卷积核尺寸一致）
        # self.global_conv_pool = nn.Sequential(
        #     nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, groups=d_model),
        #     nn.AvgPool1d(kernel_size=3, stride=1, padding=1)  # 滑动窗口=3，步长=1，padding=1
        # )
        # ========== 全局分支：修正混合池化的初始化 ==========
        self.global_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, groups=d_model)
        # 关键：kernel_size=6，padding=2（精准计算，保证输出步长=12）
        self.mixed_pool = MixedPool1d(kernel_size=6, stride=1, padding=2)

        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, 1, d_model))
        self.global_linear = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # ========== 增强版门控融合 ==========
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        # ========== 动态残差权重（适配任意N） ==========
        self.res_scale = nn.Parameter(torch.randn(1, 1, 1, d_model))
        self.scale_norm = nn.Softplus()

    # 自定义带邻域掩码的多头注意力（彻底移除k参数，硬编码邻域大小）
    def local_multihead_attn(self, x):
        """
        x: [total_batch, T, D]  # total_batch = B*N
        固定邻域k=1：仅关注前后1个时间步（直接硬编码，无参数类型问题）
        return: [total_batch, T, D]
        """
        total_batch, T, D = x.shape
        # 1. 投影Q/K/V：[total_batch, T, D] → [total_batch, T, 3*D]
        qkv = self.qkv(x).reshape(total_batch, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [total_batch, num_heads, T, head_dim]

        # 2. 计算注意力分数：[total_batch, num_heads, T, T]
        attn_scores = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=x.device))

        # 3. 生成邻域掩码（核心：硬编码k=1为整数，彻底解决类型问题）
        k_int = 1  # 直接硬编码为整数，不依赖任何参数
        # 生成基础掩码：[T, T]，仅保留前后1个时间步，其余屏蔽
        mask = torch.ones(T, T, device=x.device)
        mask = torch.triu(mask, diagonal=k_int + 1) + torch.tril(mask, diagonal=-(k_int + 1))  # 非邻域设为1
        mask = mask.bool()  # True=屏蔽，False=允许
        # 扩展到[total_batch, num_heads, T, T]
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(total_batch, self.num_heads, 1, 1)

        # 4. 应用掩码：屏蔽区域设为极小值（-1e9），softmax后权重为0
        attn_scores = attn_scores.masked_fill(mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 5. 注意力加权求和 + 输出投影
        out = (attn_weights @ v).transpose(1, 2).reshape(total_batch, T, D)
        out = self.out_proj(out)
        return out

    def forward(self, x):
        B, T, N, D = x.shape
        device = x.device
        total_batch = B * N  # B*N，如8*307=2456

        # ========== 分支1：增强版局部去趋势（带自定义掩码注意力） ==========
        # 1. 滑动窗口去趋势
        padding = torch.zeros(B, self.local_window // 2, N, D).to(device)
        x_padded = torch.cat([padding, x, padding], dim=1)
        windows = x_padded.unfold(1, self.local_window, 1)          # B，T，N，D，3winsize
        window_weight = F.softmax(self.window_weight, dim=0).to(device)         # 3winsize
        local_trend = (windows * window_weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        x_local = x - local_trend

        # 2. 自定义带邻域掩码的局部注意力（无参数传入，彻底规避类型问题）
        x_local_reshaped = x_local.permute(0, 2, 1, 3).reshape(total_batch, T, D)  # [2456,12,152]
        x_local_attn = self.local_multihead_attn(x_local_reshaped)  # 无k参数，直接调用
        x_local = x_local_attn.reshape(B, N, T, D).permute(0, 2, 1, 3)  # 恢复[B,T,N,D]
        x_local = self.local_linear(x_local)

        # ========== 全局分支：修正池化逻辑 ==========
        pos_enc = self.pos_encoding[:, :T, :, :].repeat(B, 1, N, 1).to(device)
        x_pos = x + pos_enc
        x_reshaped = x_pos.permute(0, 2, 3, 1).reshape(total_batch, D, T)  # [2720, 152, 12]

        # 卷积 + 混合池化（保证输出维度=12）
        x_conv = self.global_conv(x_reshaped)  # [2720, 152, 12]
        x_pooled = self.mixed_pool(x_conv)  # [2720, 152, 12]（核心修正）

        # 维度恢复（现在总元素数匹配，不会报错）
        x_pooled = x_pooled.reshape(B, N, D, T)  # [16, 170, 152, 12]
        x_global = x_pooled.permute(0, 3, 1, 2)  # [16, 12, 170, 152]

        # 后续门控融合无需修改，直接调用global_linear
        x_global = self.global_linear(x_global)

        # ========== 融合 + 动态残差 ==========
        gate = self.fusion_gate(torch.cat([x_local, x_global], dim=-1))  # [B,T,N,2D]→[B,T,N,D]
        branch_out = gate * x_local + (1 - gate) * x_global  # [B,T,N,D]
        adaptive_scale = self.scale_norm(self.res_scale).repeat(B, T, N, 1)  # [1,1,1,D]→[B,T,N,D]
        output = x + adaptive_scale * branch_out  # 残差连接

        return output




class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, dropout = 0.1,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)


        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(dropout)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DynamicFilter(nn.Module):
    def __init__(self, model_dim, in_steps, num_nodes,  expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, dropout = 0.1 , feed_forward_dim =2048,
                 **kwargs):
        super().__init__()

        self.size = num_nodes
        self.filter_size = (in_steps//2)+1
        self.num_filters = num_filters
        # self.dim = model_dim
        self.med_channels = int(expansion_ratio * model_dim)        #152*2 = 304
        self.pwconv1 = nn.Linear(model_dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(model_dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.complex_weights_t = nn.Parameter(
            torch.randn(num_nodes, (in_steps//2)+1, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.complex_weights_s = nn.Parameter(
            torch.randn(in_steps, (num_nodes // 2) + 1, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, model_dim, bias=bias)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        # self.mlp = Mlp(model_dim, 152)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )

    def forward(self, x, dim):
        # x = x.permute(0, 2, 1, 3)
        x = x.transpose(dim,2)
        B, H, W, _ = x.shape
        residual = x

        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters,
                                                          -1).softmax(dim=1)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        if(dim == 1):
            complex_weights = torch.view_as_complex(self.complex_weights_t)
        else:
            complex_weights = torch.view_as_complex(self.complex_weights_s)

        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)

        weight = weight.view(-1, H, W//2 +1, self.med_channels)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = self.act2(x)
        x = self.pwconv2(x)
        out = self.dropout1(x)
        # out = self.ln1(residual + out)
        alpha = nn.Parameter(torch.tensor(0.1))  # 初始值很小
        out = self.ln1(alpha * out + residual)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = self.ln2(alpha * out + residual)
        out = out.transpose(dim, 2)
        return out

class CrossAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.emb_1 = nn.Linear(456, 152)
        self.emb_2 = nn.Linear(304, 152)
        self.weight_generator = nn.Sequential(
            nn.Linear(456, 304),
            nn.ReLU(),
            nn.Linear(304, 152)
        )

    def forward(self, xl,xh, dim=-2):
        # 生成动态权重矩阵 W，形状为 [B, T, N, D]

        x_concat = torch.cat([xl, xh], dim=-1)  # Shape: (B, N, D1 + D2)
        x = self.emb_2(x_concat)

        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(xl, xh, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out

class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SelfAttention, self).__init__()

        # Linear layers for Query, Key, and Value
        self.query = nn.Linear(in_channels, out_channels)
        self.key = nn.Linear(in_channels, out_channels)
        self.value = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        """
        Args:
            x: Input node features of shape (B, N, D_in)
        Returns:
            out: Output node features of shape (B, N, D_out)
        """
        batch_size, num_nodes, _ = x.shape

        # Compute Query, Key, and Value
        Q = self.query(x)  # Shape: (B, N, D_out)
        K = self.key(x)  # Shape: (B, N, D_out)
        V = self.value(x)  # Shape: (B, N, D_out)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)  # Shape: (B, N, N)
        attn_weights = F.softmax(scores, dim=-1)  # Shape: (B, N, N)
        out = torch.matmul(attn_weights, V)  # Shape: (B, N, D_out)

        return out

class SpectralGatingNetwork(nn.Module):
    def __init__(self, model_dim, in_steps, num_nodes):
        super().__init__()
        self.complex_weight_s = nn.Parameter(torch.randn(in_steps, (num_nodes//2)+1, model_dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_t = nn.Parameter(torch.randn(num_nodes, (in_steps//2)+1, model_dim, 2, dtype=torch.float32) * 0.02)


    def forward(self, x, flag):
        # B, T, N, C = x.shape


        # print('wno',x.shape)
        if(flag=='T'):
            weight = torch.view_as_complex(self.complex_weight_t)
        else:
            weight = torch.view_as_complex(self.complex_weight_s)
            # x = x.permute(0,2,1,3)
        B, T, N, C = x.shape
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x = x * weight
        x = torch.fft.irfft2(x, s=(T, N), dim=(1, 2), norm='ortho')

        return x


class SpectralBlock(nn.Module):
    def __init__(self, model_dim, num_nodes, in_steps, dropout = 0.1 , feed_forward_dim =2048):
        super().__init__()
        self.filter = SpectralGatingNetwork(model_dim, in_steps, num_nodes)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        # self.mlp = Mlp(model_dim, 152)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )

    def forward(self, x, flag):

        if (flag == 'T'):
            x = x.transpose(1, 2)
        residual = x
        out = self.filter(x, flag)  # (batch_size, ..., length, model_dim)  btnf  bntf
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        if (flag == 'T'):
            out = out.transpose(1, 2)

        return out

class MoE_GAT(nn.Module):
    def __init__(self, in_features, out_features, experts=4, heads=1):
        super().__init__()
        self.experts = nn.ModuleList([
            CustomGATLayer(in_features, out_features, heads)
            for _ in range(experts)
        ])
        self.gate = nn.Linear(in_features, experts)

    def forward(self, x, edge_index):
        B, N, _ = x.shape
        gate = F.softmax(self.gate(x.mean(dim=1)), dim=-1)  # [B, experts]

        outputs = []
        for i, expert in enumerate(self.experts):
            out = expert(x, edge_index) * gate[:, i].view(B, 1, 1)
            outputs.append(out)

        return sum(outputs)  # [B, N, D']

class CustomGATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=1):
        super().__init__()
        self.heads = heads
        self.W = nn.Linear(in_features, out_features * heads, bias=False)
        self.attn = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        # x: [B, N, in_features], edge_index: [2, E]
        B, N, _ = x.shape
        h = self.W(x).view(B * N, self.heads, -1)  # [B * N, heads, out_features]

        # 计算注意力分数
        src, dst = edge_index
        h_src = h[src]  # [E, heads, out_features]
        h_dst = h[dst]  # [E, heads, out_features]
        cat_result = torch.cat([h_src, h_dst], dim=-1)  # [E, heads, 2 * out_features]

        # 展平多头维度
        cat_result = cat_result.view(-1, cat_result.size(-1))  # [E * heads, 2 * out_features]
        alpha = self.attn(cat_result)  # [E * heads, 1]
        alpha = alpha.view(-1, self.heads, 1)  # [E, heads, 1]
        alpha = self.leaky_relu(alpha)

        # 注意力权重归一化
        alpha = F.softmax(alpha, dim=0)  # 按边归一化

        # 消息聚合
        out = torch.zeros(B * N, self.heads, h.size(-1), device=x.device)
        out.scatter_add_(0, dst.view(-1, 1, 1).expand_as(alpha * h_src), alpha * h_src)

        return out.mean(dim=1).view(B, N, -1)  # 多头输出取平均

class DynamicThreshold(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid())  # 输出 δ ∈ (0,1)

    def forward(self, x):
        delta = self.mlp(x.mean(dim=1)).squeeze(-1)  # [B]
        return delta  # 直接输出 λ ∈ (0,1)

class ExternalAttention(nn.Module):
    """交通模式相似性提取的外部注意力"""

    def __init__(self, d_model, S=64):
        super().__init__()
        self.S = S
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        nn.init.xavier_uniform_(self.mk.weight)
        nn.init.xavier_uniform_(self.mv.weight)

    def forward(self, x):
        """输入: [B, N, D] 输出: [B, N, D]"""
        B, N, D = x.shape
        attn = self.mk(x)  # [B, N, S]
        attn = F.softmax(attn, dim=1)
        out = self.mv(attn)  # [B, N, D]
        return out




# --------------------------- 基础模块 ---------------------------
class DynamicThreshold(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x_agg = torch.mean(x, dim=1)  # [B, N, D] → [B, D]
        threshold = self.fc(x_agg).squeeze(-1)  # [B]
        return torch.sigmoid(threshold)


class MoE_GAT(nn.Module):
    def __init__(self, in_features, out_features, experts, heads):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features * heads, out_features)
        self.heads = heads

    def forward(self, x, edge_index):
        B, N, D = x.shape
        return self.fc(x.repeat(1, 1, self.heads)).reshape(B, N, self.out_features)


# --------------------------- 低影响版LEEA（核心改进） ---------------------------
class LowImpactLEEA(nn.Module):
    """
    低影响版局部增强外部注意力：
    1. 简化计算逻辑，仅保留核心创新点（局部邻域+距离加权）
    2. 引入门控系数，让模型自适应控制LEEA的贡献度
    3. 减少参数规模，降低对主模型的干扰
    """

    def __init__(self, d_model, S=32, num_neighbors=8):  # S从64→32，K从10→8，减少计算
        super().__init__()
        self.d_model = d_model
        self.S = S
        self.num_neighbors = num_neighbors

        # 简化参数：仅保留1组轻量级线性层（原EA的简化版）
        self.mk = nn.Linear(d_model, S, bias=True)  # 增加bias，减少对特征的过度依赖
        self.mv = nn.Linear(S, d_model, bias=True)

        # 可学习的门控系数（初始值0.1，让LEEA初始贡献极低）
        self.gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        # 距离衰减系数（固定值，避免学习带来的波动）
        self.beta = 0.05  # 固定小值，弱化距离加权的影响

        # 初始化：让线性层接近恒等映射，减少特征扭曲
        nn.init.eye_(self.mk.weight[:d_model, :d_model])  # 仅对角线初始化
        nn.init.zeros_(self.mk.bias)
        nn.init.eye_(self.mv.weight[:d_model, :d_model])
        nn.init.zeros_(self.mv.bias)

    def forward(self, x, mask, distances):
        """
        极简计算逻辑，低影响输出
        Args:
            x: [B, N, D] 输入特征
            mask: [B, N, N] 近距离掩码
            distances: [N, N] 归一化距离矩阵
        Returns:
            out: [B, N, D] 增强特征（贡献度由gate控制）
        """
        B, N, D = x.shape
        K = self.num_neighbors

        # 1. 极简邻域筛选（仅1行向量化操作）
        dist_masked = distances.unsqueeze(0).repeat(B, 1, 1) * mask + (1 - mask) * 1e9
        _, neighbor_indices = torch.topk(-dist_masked, k=K, dim=2)

        # 2. 提取邻域特征（仅必要操作）
        neighbor_feat = torch.gather(
            x.unsqueeze(2).expand(B, N, N, D),
            dim=2,
            index=neighbor_indices.unsqueeze(-1).expand(B, N, K, D)
        )

        # 3. 极简注意力计算（无复杂加权，仅基础操作）
        attn = self.mk(neighbor_feat)  # [B, N, K, S]
        dist_weight = torch.exp(-self.beta * torch.gather(dist_masked, dim=2, index=neighbor_indices)).unsqueeze(-1)
        attn = F.softmax(attn * dist_weight, dim=2)

        # 4. 聚合+门控融合（核心：gate控制贡献度）
        attn_agg = torch.sum(attn, dim=2)  # [B, N, S]
        leea_out = self.mv(attn_agg)  # [B, N, D]

        # 门控融合：原特征为主，LEEA特征为辅（gate∈[0,1]）
        out = x + torch.sigmoid(self.gate) * leea_out
        return out


# --------------------------- 改进后的STGAFormer（修复融合维度） ---------------------------
class STGAFormer(nn.Module):
    def __init__(self, model_dim=152, feed_forward_dim=256, num_heads=4, dropout=0.1):
        super(STGAFormer, self).__init__()
        self.model_dim = model_dim
        self.threshold_predictor = DynamicThreshold(model_dim)

        # 重要性分支：低影响LEEA + 简化结构（减少LEEA的权重）
        self.importance_net = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            LowImpactLEEA(model_dim // 2),
            LowImpactLEEA(model_dim // 2),
            LowImpactLEEA(model_dim // 2),
            LowImpactLEEA(model_dim // 2)
        )

        # 相似性分支：增强为主（让模型主要依赖该分支，降低LEEA影响）
        self.similarity_net = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            # 增强MoE-GAT的表达能力，让模型更依赖该分支
            MoE_GAT(in_features=model_dim // 2,
                    out_features=model_dim // 2,
                    experts=4,  # 专家数从3→4，增强表达
                    heads=num_heads),
            nn.ReLU(),
            # 增加1层轻量级线性层，强化相似性分支
            nn.Linear(model_dim // 2, model_dim // 2)
        )

        # 特征融合：修复维度匹配问题，且保持低影响逻辑
        self.fusion = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),  # 输入152→输出76，匹配imp_feat/sim_feat维度
            nn.LayerNorm(model_dim // 2),  # 归一化维度改为76
            nn.Sigmoid()
        )
        # 最终投影回原维度（152）
        self.fusion_proj = nn.Linear(model_dim // 2, model_dim)
        # 融合权重参数（初始让相似性分支占主导）
        self.fusion_weight = nn.Parameter(torch.tensor([0.2, 0.8], dtype=torch.float32))

    def _load_distance_matrix(self):
        with open('./data/PEMS03/adj_PEMS03_distance.pkl', 'rb') as f:
            adj = pickle.load(f)
        distances = torch.tensor(adj, dtype=torch.float32)
        d_min, d_max = distances.min(), distances.max()
        distances = (distances - d_min) / (d_max - d_min + 1e-8)
        return distances

    def forward(self, x):
        B, T, N, D = x.shape
        distances = self._load_distance_matrix().to(x.device)
        outputs = []

        for t in range(T):
            x_t = x[:, t, :, :]  # [B, N, 152]

            # 1. 动态阈值计算
            threshold = self.threshold_predictor(x_t)  # [B]

            # 2. 重要性分支（低影响LEEA）
            imp_mask = (distances < threshold.view(B, 1, 1)).float()
            imp_feat = self.importance_net[0](x_t)  # [B, N, 76]
            imp_feat = self.importance_net[1](imp_feat)  # ReLU
            imp_feat = self.importance_net[2](imp_feat, imp_mask, distances)  # 低影响LEEA

            # 3. 相似性分支（增强为主）
            sim_mask = (distances >= threshold.view(B, 1, 1)).float()
            sim_feat = self.similarity_net[0](x_t)  # [B, N, 76]
            sim_feat = sim_feat * sim_mask.mean(dim=2, keepdim=True)
            if sim_mask[0].sum() > 0:
                edge_index = torch.nonzero(sim_mask[0], as_tuple=False).t()
                sim_feat = self.similarity_net[1](sim_feat, edge_index)
            sim_feat = self.similarity_net[2](sim_feat)  # ReLU
            sim_feat = self.similarity_net[3](sim_feat)  # 增强线性层

            # 4. 自适应融合（修复维度匹配，核心：降低LEEA分支影响）
            # 4.1 加权融合（76维度，保持低影响逻辑）
            alpha = torch.sigmoid(self.fusion_weight[0])  # 约束在0-1
            beta = torch.sigmoid(self.fusion_weight[1])
            alpha = alpha / (alpha + beta)  # 归一化，保证权重和为1
            beta = 1 - alpha
            combined = alpha * imp_feat + beta * sim_feat  # [B, N, 76]

            # 4.2 融合门控（维度匹配，无冲突）
            fusion_gate = self.fusion[0](torch.cat([imp_feat, sim_feat], dim=-1))  # [B, N, 76]
            fusion_gate = self.fusion[1](fusion_gate)  # LayerNorm
            fusion_gate = self.fusion[2](fusion_gate)  # Sigmoid，门控系数

            # 4.3 最终融合（维度一致，76→152）
            z_s = fusion_gate * combined + (1 - fusion_gate) * fusion_gate  # [B, N, 76]
            z_s = self.fusion_proj(z_s)  # 投影回152维度 [B, N, 152]

            outputs.append(z_s.unsqueeze(1))

        return torch.cat(outputs, dim=1)





class GatedNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedNetwork, self).__init__()
        # 定义卷积层
        self.conv_filter = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv_gate = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, X):
        # 输入形状: (B, T, N, D)
        B, T, N, D = X.shape

        # 将输入重塑为适合卷积的形状: (B * N, T, D)
        X_reshaped = X.view(B * N, T, D).permute(0, 2, 1)  # 形状变为 (B * N, D, T)

        # 计算滤波器
        filter = torch.tanh(self.conv_filter(X_reshaped))  # 形状不变 (B * N, D, T)

        # 计算门控信号
        gate = torch.sigmoid(self.conv_gate(X_reshaped))  # 形状不变 (B * N, D, T)

        # 应用门控机制
        gated_X = filter * gate  # 形状不变 (B * N, D, T)

        # 重塑回原始形状
        gated_X = gated_X.permute(0, 2, 1)
        gated_X = gated_X.reshape(B, T, N, D)  # 形状变为 (B, T, N, D)

        return gated_X


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class STAEformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_spatial = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(2)
            ]
        )
        self.gated = GatedNetwork(self.model_dim, self.model_dim)

        self.spatial_attn = STGAFormer(self.model_dim, feed_forward_dim, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(self.model_dim)
        self.fea_reduce = nn.Linear(304, 152)

        self.spect_wave = SpectralBlock(self.model_dim, num_nodes, in_steps, dropout)
        self.attn_layers_fusion = nn.ModuleList(
            [
                CrossAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.spect = DynamicFilter(self.model_dim, in_steps , num_nodes)
        self.temporal_dep = TemporalDualBranch(self.model_dim, 3)

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]            #64,12,307,3

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim) #64,12,307,24
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim) 64，12，307，152


        x_s = x
        x_t = x

        ''' for spatial dependency'''
        x_s = self.spatial_attn(x_s)
        # for attn in self.attn_layers_s:
        #     x_s = attn(x_s, dim=2)

        ''' for temporal dependency'''
        x_t = self.temporal_dep(x_t)
        # for attn in self.attn_layers_t:
        #     x_t = attn(x_t, dim=1)

        x_concat = torch.cat([x_t, x_s], dim=-1)  # Shape: (B, N, D1 + D2)
        x = self.fea_reduce(x_concat)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        


        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)  8，307，12，152
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out


if __name__ == "__main__":
    model = STAEformer(207, 12, 12)
    summary(model, [64, 12, 207, 3])
