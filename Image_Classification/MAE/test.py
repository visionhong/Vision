import torch
from vit import ViT
from mae import MAE
import timm

if __name__ == '__main__':
    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048
    )

    mae = MAE(
        encoder=v,
        masking_ratio=0.75,  # the paper recommended 75% masked patches
        decoder_dim=512,  # paper showed good results with just 512
        decoder_depth=6  # anywhere from 1 to 8
    )

    images = torch.randn(8, 3, 256, 256)

    loss = mae(images)
    print(f"Masked Autoencoders MSE Loss: {loss:.5f}")

