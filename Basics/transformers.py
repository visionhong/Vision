import torch
import torch.nn as nn
import pdb

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size  # 512
        self.heads = heads  # 8
        self.head_dim = embed_size // heads  # 64

        assert (self.head_dim * heads == self.embed_size), 'Embed size needs to be div by heads'

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        '''
        주석에 달린 차원 :  encoder self-attention   or   encoder-decoder attention
        Masked decoder self-attention의 차원은 주석에 달지 않았음
        '''
        N = query.shape[0]  # 2
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]  # 9, 9, 9 or 9, 9, 7

        # Split embedding into self.heads pieces
        # h 개로 구분된 서로다른 value, key, query 쌍을 만들어서 좀더 다양하게 학습되도록 함
        values = values.reshape(N, value_len, self.heads, self.head_dim)  # (2, 9, 8, 64)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)  # (2, 9, 8, 64)  or (2, 7, 8, 64)

        values = self.values(values)  # (N, value_len, heads, head_dim) (2, 9, 8, 64)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim) (2, 9, 8, 64) or (2, 7, 8, 64)
        # keys shape: (N, keys_len, heads, heads_dim) (2, 9, 8, 64)
        # energy shape: (N, heads, query_len, key_len)  query_len : target sentence / key_len : src sentence
        # endoder-decoder attention에 대해서 : decoder의 query(문장)이 encoder의 out key(각각의 단어)와 얼마나 유사한지 알아냄
        # 즉 '나는 선생님 이다'라는 쿼리에서 '나는' '선생님' '이다' 각각이 'i' 'am' 'a' 'teacher' 에서 어떤 단어와 연관성이 있는지, 즉 가중치를 학습
        # (2, 8, 9, 9) or (2, 8, 7, 9) : 각각의 헤드(8)에 대해서 query(9 or 7)와 key(9)간의 attention 에너지를 구함
        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))  # mask에 걸리는 값은 아주 작은수(-무한)로 (softmax에서 출력이 0에 아주 가까운 값이 됨) -> 특정 단어를 무시하기위해 사용
            # energy = QK
        # softmax(QK/root(512))
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)  # src끼리의 attention (2, 8, 9, 9) or (2, 8, 7, 9)

        # attetion과 실제 value값의 matmul
        out = torch.einsum('nhql,nlhd->nqhd', [attention, values]).reshape(N, query_len, self.heads*self.head_dim)  # 다시 원래 차원으로 concat
        # attention shape: (N, heads, query_len, key_len) (2, 8, 9, 9) or (2, 8, 7, 9)
        # values shape: (N, value_len, heads, heads_dim) (2, 9, 8, 64)
        # (N, query_len, heads, heads_dim) (2, 9, 8, 64) -> (2, 9, 512) or (2, 7, 8, 64) -> (2, 7, 512)
        # 항상 key_len와 value_len는 같다.

        out = self.fc_out(out)  # (2, 9, 512) or (2, 7, 512)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):

        attention = self.attention(value, key, query, mask)  # (2, 9, 512)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))  # (2, 9, 512)
        forward = self.feed_forward(x)  # (2, 9, 512)
        out = self.dropout(self.norm2(forward + x))  # (2, 9, 512)
        return out



class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 device,
                 forward_expansion,
                 dropout,
                 max_length,):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        '''
        모든 단어를 정수 인코딩(ex) great -> 30, good-> 1450) 한 뒤에 각각 임베딩 벡터로 표현할 수있다.(여기선 512 embedd dim)
        이 각각의 단어가 가지는 임베딩 벡터를 학습시키는 것이다.
        '''
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)

        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        N, seq_length = x.shape  # 2, 9
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)  # (2, 9)

        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))  # skip connection  (2, 9, 512)
        )

        # 인코더에서 query, key, value는 전부 같다. (디코더에선 달라짐)
        for layer in self.layers:
            out = layer(out, out, out, mask)  # (2, 9, 512)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)  # (2, 7, 512)  하삼각행렬 마스크
        query = self.dropout(self.norm(attention + x))  # (2, 7, 512)
        out = self.transformer_block(value, key, query, src_mask)  # (2, 9, 512),(2, 9, 512),(2, 7, 512), 일반 마스크
        return out


class Decoder(nn.Module):
    def __init__(self,
                 trg_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_length,):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):

        N, seq_length = x.shape  # (2, 7)
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)  # (2, 7)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))  # (2, 7, 512)

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)  # (2, 7, 512)
        out = self.fc_out(x)  # (2, 7, 10)

        return out


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size=512,
                 num_layers=6,
                 forward_expansion=4,
                 heads=8,
                 dropout=0,
                 device='cpu',
                 max_length=100,
                 ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)  # tril: 들어온 행렬을 하삼각행렬로 변환
        '''
        Masked Decoder Self-Attention에서 하 삼각행렬을 마스크로 사용하는 이유 
         - 만약 target문장이 '나는 게임을 했다' 일때 '게임을' 이라는 단어는 '나는' 이라는 단어를 참고해도 되지만 
         '했다' 라는 단어를 참고하면 일종의 cheat로 볼 수 있기 때문이라고 한다.
        '''
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)  # src 요소 각가에 대해 0이 아니면 True 0이면 False인 마스크
        trg_mask = self.make_trg_mask(trg)  # 7 x 7 하 삼각행렬
        enc_src = self.encoder(src, src_mask)  # (2, 9, 512)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)  # (2, 7, 10)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)  # (2, 7, 10)
    print(out.max(2))














