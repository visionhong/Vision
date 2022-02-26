import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers import trunc_normal_
import timm

class DistillationVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super(DistillationVisionTransformer, self).__init__()
        # new : distillation token
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))  # (n_samples, token_dim, embed_dim)
        num_patches = self.patch_embed.num_patches  # 패치의 개수 ex) 384x384 with 16 patch_size -> 576개
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        # new : distillation classifier(head)
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        # initialize
        trunc_normal_(self.dist_token, std=.02)  # mean = 0 / std = 0.02
        trunc_normal_(self.pos_embed, std=.02)

        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)  # 배치사이즈만큼 확장
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_token, dist_token, x), dim=1)  # concatenate cls_token, dist_token, x 순

        x = x + self.pos_embed  # positional embedding
        x = self.pos_drop(x)  # use dropout if we need

        for block in self.blocks:  # encoder (multi-head-attention + mlp)
            x = block(x)

        x = self.norm(x)  # Layer Normalization
        return x[:, 0], x[:, 1]  # cls_embedding, dist_embedding

    def forward(self, x):
        x, x_dist = self.forward_features(x)  # cls, dist
        x = self.head(x)  # cls classifier
        x_dist = self.head_dist(x_dist)  # dist classifier

        if self.training:
            return x, x_dist

        else:
            return (x + x_dist / 2)  # mean



def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistillationVisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.defualt_cfg = _cfg()

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

if __name__ == '__main__':
    from torchsummary import summary
    model = deit_base_distilled_patch16_224()
    # print(model)
    summary(model, input_size=(3, 224, 224), device='cpu')

