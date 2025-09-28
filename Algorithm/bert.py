import torch
import torch.nn as nn


class Bert(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim=128,
                 num_layers=4, ff_dim=512, num_heads=4, dropout=0.1, CLS=False, TANH=False):
        super().__init__()
        self.CLS = CLS
        self.projection = nn.Linear(input_dim, embed_dim)
        if self.CLS:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.randn(1, input_dim + 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, input_dim, embed_dim))

        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        if TANH:
            self.classifier = nn.Sequential(nn.Linear(embed_dim, output_dim), nn.Tanh())
        else:
            self.classifier = nn.Linear(embed_dim, output_dim)

        self.layers.train()
        self.classifier.train()

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, input_dim)
        # 线性投影
        x = self.projection(x)  # (batch_size, input_dim, embed_dim)

        batch_size = x.size(0)
        if self.CLS:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, 29, embed_dim)

        # 添加位置编码
        x = x + self.pos_embed

        # 转置为(seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)

        for layer in self.layers:
            x = layer(x, mask=mask)

        if self.CLS:
            return self.classifier(x[0, :, :])
        else:
            pooled = x.mean(dim=0)  # (batch_size, embed_dim)
            return self.classifier(pooled)



class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        # 使用GELU激活函数
        self.activation = nn.GELU()

    def forward(self, x, mask=None):
        # Post-LN 结构 (残差连接后归一化)
        # 注意力部分
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # FFN部分
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x