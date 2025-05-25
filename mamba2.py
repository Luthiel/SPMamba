import torch
import torch.nn as nn
import math

from einops import rearrange

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError:
    print("Mamba module not found, make sure 'mamba_ssm' package is installed")


class Mamba2Block(nn.Module):
    def __init__(self, 
                 d_model, 
                 d_state, 
                 d_conv=4, 
                 expand_factor=2, 
                 head_dim=128, 
                 ngroups=1,
                 chunk_size=256,
                 dt_min=0.001, 
                 dt_max=0.1, 
                 dt_init="random", 
                 dt_init_floor=1e-4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.d_inner = int(self.expand_factor * self.d_model)
        self.head_dim = head_dim
        self.ngroups = ngroups
        print(self.d_inner, self.head_dim, self.ngroups)
        assert self.d_inner % self.head_dim == 0, "d_inner must be divisible by head_dim"
        self.heads = self.d_inner // self.head_dim
        self.chunk_size = chunk_size
        
        # SSM参数
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_init_floor = dt_init_floor
        
        # 投影层 (一个是 shortcut，一个是 input)
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.heads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)  # dimension concate for [z, x, B, C, dt]
        
        # 卷积层用于局部信息聚合
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            padding=d_conv - 1, # 一维卷积核大小 - 1 即为需要填充的大小
            groups=conv_dim,
            bias=True
        )
        
        self.act = nn.SiLU()
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        # 初始化 SSM 参数
        if self.dt_init == "random":
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min))
        elif self.dt_init == "constant":
            dt = torch.exp(torch.ones(self.d_inner) * math.log(self.dt_min))
        else:
            raise ValueError(f"Unknown dt_init: {self.dt_init}")
        dt = torch.clamp(dt, min=self.dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        
        # S4D参数
        A = torch.empty(self.heads).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A).float())
        self.D = nn.Parameter(torch.ones(self.heads))
        self.D._no_weight_decay = True
        nn.init.uniform_(self.A_log, a=math.log(0.001), b=math.log(0.999))
        nn.init.uniform_(self.D, a=-0.5, b=0.5)
        
        # 初始化投影层
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
        # nn.init.zeros_(self.in_proj.bias)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        # nn.init.zeros_(self.out_proj.bias)
        
        # 初始化卷积
        nn.init.zeros_(self.conv1d.bias)
        
    def forward(self, u):
        """
        参数:
            x: 形状为 (batch, seq_len, d_model) 的输入特征
            cache: 可选缓存，用于增量解码
        返回:
            y: 形状为 (batch, seq_len, d_model) 的输出特征
        """
        batch, seq_len, _ = u.shape
        
        A = -torch.exp(self.A_log)
        zxbcdt = self.in_proj(u)  
        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.heads], dim=-1)
        print(z.shape, xBC.shape, dt.shape, self.dt_bias.shape)
        dt = dt + self.dt_bias
        # 在我们的任务中建议尽可能使用 Causal Conv1d，因为我们的动态图数据是遵循严格时间序的
        if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
            xBC = self.act(
                # first transposed: (B, L, self.d_inner + 2 * ngroups * d_state) -> (B, self.d_inner + 2 * ngroups * d_state, L)
                self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
            )  # (B, L, self.d_inner + 2 * ngroups * d_state)
            xBC = xBC[:, :seq_len, :]
        else:
            xBC = causal_conv1d_fn(
                x=xBC.transpose(1, 2),
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            ).transpose(1, 2)
        
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim), # 划分最后一个维度，p 是 head 维度
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None
        )
        y = rearrange(y, "b l h p -> b l (h p)")

        # Multiply "gate" branch and apply extra normalization layer
        y = self.norm(y, z)
        
        # 输出投影
        y = self.out_proj(y)
        
        return y