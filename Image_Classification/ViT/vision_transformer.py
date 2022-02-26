import torch
import torch.nn as nn
import pdb

class PatchEmbed(nn.Module):
    """
    이미지를 패치로 나누고 1차원 벡터로 변환

    Parameters
    ----------
    img_size : image size(square)
    patch_size: patch_size(square)
    in_chans : 3(color) or 1(gray)
    embed_dim : emmbedding dimension (basic use 768)

    Attributes
    ----------
    n_patches : number of patches inside image (int)
    proj : nn.Conv2d(이미지를 패치로 나누고 임베딩으로 변환하기 위한 convolution
    -> 일반적인 convolution과 다름)


    """
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )  # Embedding dim으로 변환하며 패치크기의 커널로 패치크기만큼 이동하여 이미지를 패치로 분할 할 수 있음.

    def forward(self, x):
        """
        Parameters
        __________
        x = torch.Tensor
            shape (batch_size, in_chans, img_size, img_size)

        Returns
        _______
        torch.Tensor
        shape(batch_size, n_patches, embed_dim)
        """
        x = self.proj(x)  # (batch_size, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # 세번째 차원부터 끝까지 flatten (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x

class Attention(nn.Module):
    """
    Attention mechanism

    Parameters
    __________
    dim : int
        The input and out dimension of per token features.

    n_heads : int
        Number of attention heads.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    attn_p : float
        Dropout probability applied to the query, key, and value tensors.

    proj_p : float
        Dropout probability applied to the output tensor.

    Attrubutes
    __________
    scale : float
        Normalizing consant for the dot product.

    qkv : nn.Linear
        Linear projection for the query, key, and value

    proj : nn.Linear
        Linear mappin that takes in the concatenated output of all attention heads and maps it into a new space.

    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5  # 1 / root(self.head_dim)
        '''
        # 나중에 query와 key를 곱하고 softmax를 취하기전에 scale factor로 나눠주는데 이 scale factor의 역할은 
        query @ key 의 값이 커지게 되면 softmax 함수에서의 기울기 변화가 거의 없는 부분으로 가기때문에 gradient vanishing
        문제를 해결하려면 scaling을 해주어야 한다고 Attention is all you need 논문에서 주장
         
        '''
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """
        Parameters
        __________
        x : torch.Tensor
            shape (batch_size, n_patches + 1, dim)  # sequence 첫번쩨 자리는 class token이 있기 때문에 1 추가
        
        Returns
        _______
        torch.Tensor   
            shape (batch_size, n_patches + 1, dim)     
        """

        n_samples, n_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches+1, dim*3)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_samples, n_patches+1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, n_samples, n_heads, n_patches+1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 각각의 n_heads끼리 query, key, value로 나눔
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches+1)  dot product를 위한 transpose
        # dot product를 통해 query와 key사이의 유사도를 구함
        dp = (q @ k_t) * self.scale  # (n_samples, n_heads, n_patches+1, n_patches+1)  @: dot product (2x1)@(1x2)=(2x2)
        attn = dp.softmax(dim=-1)  # attention (n_samples, n_heads, n_patches+1, n_patches+1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches+1, head_dim)
        # 원래 차원으로 되돌림.
        weighted_avg = weighted_avg.transpose(1, 2)  # (n_samples, n_patches+1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # concat (n_samples, n_patches+1, dim)

        x = self.proj(weighted_avg)  # linear projection (n_samples, n_patches+1, dim)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """
        Parameters
        __________
        x: torch.Tensor
            shape (n_samples, n_patches + 1, in_features)

        Returns
        _______
        torch.Tensor
            shape (n_sample, n_patches + 1, out_features)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    Transformer Encoder.

    Parameters
    __________
    dim : int
        Embedding dimension

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension size of the 'MLP' module with respect to 'dim'.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    __________
    norm1, norm2 : LayerNorm
        Layer normalization

    attn : Attentopm
        Attention module.

    mlp : MLP
        MLP module.
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)  # 3072(MLP size)
        self.mlp = MLP(
            in_features=dim,
            hidden_features= hidden_features,
            out_features=dim,
        )

    def forward(self, x):
        """
         Parameters
         __________
         x : torch.Tensor
            shape (n_samples, n_patches + 1, dim)

        Returns
        _______
        torch.Tensor
            shape (n_samples, n_patches + 1, dim)
        """

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Simplified implementation of the vision transformer.

    Parameters
    __________
    img_size : int
        Both height and the width of the image (square)

    patch_size : int
        Both heicht and the width of the patch (square)

    in_chans : int
        Number of input channels

    n_classes : int
        Number of classes.
    embed_dim : int
        Dimensionality of the token/patch embeddings.

    depth : int
        Number of blocks.

    n_heads : int
        Number of attention heads

    mlp_ratio : float
        Determines the hidden dimension of the 'MLP' module.

    qkv_bias : bool
        If True then we include vias to the query, key, and value projections.

    p, attn_p : float
        Dropout probability

    Attributes
    __________
    patch_embed : PatchEmbed
        Instance of 'PatchEmbed' layer.

    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence
        It has 'embed_dim' elements.

    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has '(n_patches + 1) * embed_dim' elements.

    pos_drop : nn.Dropout
        Dropout layer

    blocks : nn.ModuleList
        List of 'Block' modules.

    norm : nn.LayerNorm
        Layer normalization.
    """

    def __init__(
            self,
            img_size=384,
            patch_size=16,
            in_chans=3,
            n_classes=1000,
            embed_dim=768,
            depth=12,
            n_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.):
        super(VisionTransformer, self).__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)  # 12개의 block
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """
        Parameters
        __________
        x : torch.Tensor
            shape (n_samples, in_chans, img_size, img_size)

        Returns
        _______
        logits : torch.Tensor
            Logits over all the classes - '(n_samples, n_samples)'
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)  # (n_samples, n_patches, embed_dim)
        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1+n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, 1+n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)  # (n_samples, 577, 768)

        x = self.norm(x)

        cls_token_final = x[:, 0]  # just tje CLS token
        x = self.head(cls_token_final)

        return x




if __name__ == '__main__':
    from torchsummary import summary

    custom_config = {
        "img_size": 384,
        "in_chans": 3,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "n_heads": 12,
        "qkv_bias": True,
        "mlp_ratio": 4
    }
    model_custom = VisionTransformer(**custom_config)

    inp = torch.rand(2, 3, 384, 384)
    res_c = model_custom(inp)
    print(res_c.shape)

    # summary(model_custom, input_size=(3, 384, 384), device='cpu')
    print(len(model_custom.state_dict().keys()))










