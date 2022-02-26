import pdb
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# Attention, MLP 이전에 수행되는 Layer Normalization
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# 각 쿼리(패치)가 다른 패치와 어느정도 연관성을 가지는지 구하는것이 바로 attention의 목적.
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads  # 512
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads  # multi head attention (시퀀스를 병렬로 분할함으로써 다르게 주의를 기울이고 다양한 특징을 얻을 수 있다고 함)
        self.scale = dim_head ** -0.5  # 큰값을 softmax에 올리면 gradient vanishing이 일어나기 때문에 downscale에 사용될 값 (softmax 함수 그래프를 보면 이해가능)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # query, key, value로 분할하기 위해 3을 곱해줌

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # embed dim 기준으로 3분할 (튜플로 감싸져 있음)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # q = k = v (b, heads, num_patches, dim)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # query와 key간의 dot product를 위한 차원변경 + scaling
        # dots = (b, heads, num_patches, dim) * (b, heads, dim, num_patches) = (b, heads, num_patches, num_patches)

        attn = self.attend(dots)  # self attention (각 패치간의 연관성을 softmax 확률값으로 나타냄)

        out = torch.matmul(attn, v)  # 구한 확률값을 실제 값(value)에 dot product 시킴 (원래 차원으로 복구) (b, heads, num_patches, dim)
        # out = (b, heads, num_patches, num_patches) * (b, heads, num_patches, dim) = (b, heads, num_patches, dim)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)  # 원래 dim으로 복귀


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # skip connection
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0.):
        super(ViT, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, '이미지 사이즈를 패치 사이즈로 나눌 수 없음 (Must be divisible)'

        num_patches = (image_height // patch_height) * (image_width // patch_width)  # 이미지의 패치 수
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # class token이 패치 순서 첫번째에 추가되니까 1을 더해줌
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )




    def forward(self, img):
        x = self.to_patch_embedding(img)  # Rearrange (b, num_patches, patch_dim) -> Linear (b, num_patches, dim)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # 각 이미지(배치) 마다 클래스 토큰 보유
        x = torch.cat((cls_tokens, x), dim=1)  # 클래스 토큰이 첫번째로 오도록 하고 패치개수의 차원 dim=1로 concat 시킨다.
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # mean은 classification을 전체 패치의 평균값을 사용한다는 것이고 cls는 class token의 값만 사용한다는 것.
        # 논문은 class token이 이미지 전체의 embedding을 표현하고 있음을 가정하기 때문에 class token만 사용하였음.
        x = self.to_latent(x)  # make more compact
        return self.mlp_head(x)  # classification


if __name__ == '__main__':

    model = ViT(image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048
    )

    input_data = torch.randn(4, 3, 256, 256)
    out = model(input_data)
    print(f'INPUT SIZE: {input_data.shape}   |   OUTPUT SIZE: {out.shape}')
