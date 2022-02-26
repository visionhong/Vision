import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from vit import Transformer


class MAE(nn.Module):
    def __init__(self, *, encoder, decoder_dim, masking_ratio=0.75, decoder_depth=1, decoder_heads=8, decoder_dim_head=64):
        super(MAE, self).__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # some encoder parameters extract
        self.encoder = encoder  # vit encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]  # patch_size * patch_size * 3 (패치당 픽셀 개수(rgb))
        # decoder parameters
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head, mlp_dim=decoder_dim*4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape  # (b, 64, 3072)  not in class token

        tokens = self.patch_to_emb(patches)  # shape (b, 64, 1024)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches+1)]  # not in class token

        # mask, unmask의 랜덤인덱스를 생성
        num_masked = int(num_patches * self.masking_ratio)  # int(64 * 0.75) = 48
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)  # 배치별로 패치에 uniform distribution으로 랜덤 index 부여(논문에서 uniform distribution 사용)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]  # shape (b, 48) (b, 16)

        # unmasked 위치의 토큰값만 인덱싱
        batch_range = torch.arange(batch, device=device)[:, None]  # shape (b, 1)
        tokens = tokens[batch_range, unmasked_indices]  # 마스크가 아닌 위치의 embed값만 인덱싱함  shape (b, 16, 1024)

        # reconstruction loss를 계산하기 위한 정답 masked_patches
        masked_patches = patches[batch_range, masked_indices]  # shape (b, 48, 3072)

        # encoding
        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # repeat mask token
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)  # (b 48 512)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)  # mask_token도 positional embedding 추가

        # concat tokens and decoding
        # position embedding을 둘다 주었기 때문에 원래 sequence로 돌려놓지 않고 바로 concat시킴
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)  # shape (b, 64, 512)
        decoded_tokens = self.decoder(decoder_tokens)

        mask_tokens = decoded_tokens[:, :num_masked]  # 위에서 concat을 mask먼저 했으므로 이렇게 mask 정보만 인덱싱
        pred_pixel_values = self.to_pixels(mask_tokens)  # (b, 48, 3072)

        # calculate reconstruction loss

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss

