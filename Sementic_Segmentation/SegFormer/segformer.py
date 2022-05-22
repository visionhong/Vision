import torch
from torch import nn, einsum
from torchsummary import summary
from einops import rearrange
from math import sqrt

class DsConv2d(nn.Module):
    '''
    Mix-FFN에 Positional encoding을 대신할 3x3 Depthwise separable convolution
    '''
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride=1, bias=True):
        super(DsConv2d, self).__init__()
        self.net = nn.Sequential(
            # Depthwise Convolution
            nn.Conv2d(in_channels=dim_in,
                      out_channels=dim_in,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=dim_in,
                      bias=bias
                      ),
            # Pointwise Convolution
            nn.Conv2d(in_channels=dim_in,
                      out_channels=dim_out,
                      kernel_size=1,
                      bias=bias
                      )
        )

    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    '''
    Efficient Self-Attn block과 Mix-FFN block 전에 사용될 Layer Normalization
    '''
    def __init__(self, dim, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()  # std = root(variation)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class EfficientSelfAttention(nn.Module):
    def __init__(self, *, dim, heads, reduction_ratio):
        super(EfficientSelfAttention, self).__init__()
        self.scale = (dim // heads) ** -0.5  # root(dim head)
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, dim*2, reduction_ratio, stride=reduction_ratio, bias=False)  # ESA
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))  # chunk: dim=1을 기준으로 2개로 쪼갬
        # q,k,v의 차원을 모두 아래와 같이 변경
        # batch, (head*channel), w, h -> (batch*head), (w*h), channel
        # 첫 stage에서 16384 -> 256 으로 행렬곱 연산을 64배 감소시킴(Efficient multi head self attention)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=heads), (q, k, v))

        # einsum을 아래와 같이 행렬곱으로 사용 가능, q@k로도 가능
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale  # Q*K / root(dim head)
        attn = sim.softmax(dim=-1)  # dim=-1 -> channel

        out = einsum('b i j, b j d -> b i d', attn, v)  # 최종 attention 계산
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h=heads, x=h, y=w)  # 차원 복구
        return self.to_out(out)


class MixFeedForward(nn.Module):
    def __init__(self, *, dim, expansion_factor):
        super(MixFeedForward, self).__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),  # positional embedding 대체
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class MiT(nn.Module):
    '''
    Mix Transformer encoders
    '''
    def __init__(self,
                 *,
                 channels,
                 dims,
                 heads,
                 ff_expansion,
                 reduction_ratio,
                 num_layers
                 ):
        super(MiT, self).__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)  # *: take off tuple  (3, 32, 64, 160, 256)
        dim_pairs = list(zip(dims[:-1], dims[1:]))  # [(3, 32), (32, 64), (64, 160), (160, 256)]

        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel, stride=stride, padding=padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            layers = nn.ModuleList([])

            # Layer Norm -> ESA -> Layer Norm -> MFFN
            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim=dim_out, heads=heads, reduction_ratio=reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim=dim_out, expansion_factor=ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
               get_overlap_patches,
               overlap_patch_embed,
               layers
            ]))

    def forward(self, x, return_layer_outputs=False):
        h, w = x.shape[-2:]

        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)  # (b, c x kernel x kernel, num_patches)

            num_patches = x.shape[-1]
            ratio = int(sqrt(h * w / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)  # (b, c(cxkernelxkernel), h, w)
            x = overlap_embed(x)  # (b c(embed dim) h w)
            for (attn, ff) in layers:  # attention, feed forward
                x = attn(x) + x  # skip connection
                x = ff(x) + x

            layer_outputs.append(x)  # multi scale features

        return x if not return_layer_outputs else layer_outputs

class SegFormer(nn.Module):
    '''
    Default values from Mix Transformer B0
    '''
    def __init__(self,
                 *,
                 dims=(32, 64, 160, 256),
                 heads=(1, 2, 5, 8),
                 ff_expansion=(8, 8, 4, 4),
                 reduction_ratio=(8, 4, 2, 1),
                 num_layers=(2, 2, 2, 2),
                 channels=3,
                 decoder_dim=256,
                 num_classes=4):
        super(SegFormer, self).__init__()
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels=channels,
            dims=dims,
            heads=heads,
            ff_expansion=ff_expansion,
            reduction_ratio=reduction_ratio,
            num_layers=num_layers
        )
        # 논문 모델 그림에서 MLP Layer
        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),  # First, multi-level features Fi from the MiT encoder go through an MLP layer to unify the channel dimension.
            nn.Upsample(scale_factor=2**i)  # second step, features are up-sampled to 1/4th and concatenated together.
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),  # Third, a MLP layer is adopted to fuse the concatenated features
            nn.Conv2d(decoder_dim, num_classes, 1),  # Finally, another MLP layer takes the fused feature to predict the segmentation mask M with a H4 × W4 × Ncls resolution
        )

    def forward(self, x):
        layer_outputs = self.mit(x, return_layer_outputs=True)

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim=1)
        return self.to_segmentation(fused)


if __name__ == '__main__':
    model = SegFormer()
    img = torch.rand((1, 3, 512, 512))

    summary(model, input_size=(3, 512, 512), device='cpu')
    print(f'OUTPUT Shape: {model(img).shape}')










































