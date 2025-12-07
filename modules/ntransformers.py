import torch
import torch.nn as nn

class AttnBlock(nn.Module):
    """Batch-first pre-normalized **Multi Head Attention** block.

    Args:
        embeddim (int): Embedding dimension
        num_heads (int, optional): Number of heads. Defaults to 8.
        dropout (float, optional): Defaults to 0.1.
        mlp_ratio (float, optional): Feed forward projection ratio. Defaults to 4.0.
    """
    def __init__(self, 
                 embeddim, 
                 num_heads:int=8,
                 dropout:float=0.1,
                 mlp_ratio:float=4.0):
        
        super().__init__()

        self.q_norm = nn.LayerNorm(embeddim)
        self.kv_norm  = nn.LayerNorm(embeddim)
        self.attn  = nn.MultiheadAttention(embeddim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(embeddim)
        self.dropout = nn.Dropout(dropout)

        self.mlp   = nn.Sequential(
            nn.Linear(embeddim, int(embeddim * mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(embeddim * mlp_ratio), embeddim),
        )

    def forward(self, q, k, v, key_padding_mask=None): 
        # Layer Normalization
        nq, nk, nv = self.q_norm(q), self.kv_norm(k), self.kv_norm(v)

        # MultiHead Attention - Add
        attn_out, _ = self.attn.forward(nq,nk,nv,key_padding_mask,need_weights=False)
        attn_out = self.dropout(attn_out)
        q = q + attn_out

        # Layer Normalization
        nq = self.norm(q) 
        
        # Feed Forward - Add
        mlp_out = self.mlp(nq)
        mlp_out = self.dropout(mlp_out)
        q = q + mlp_out

        return q
    
class AudioSeqEncoder(nn.Module):
    """Audio Sequence Encoder

    Process audio data shaped like [batch_size, seqnum, embeddim], after AudioCNNFeats.\\
    Uses Multi Head Self-Attention.

    Args:
        seqnum (int): Sequence number
        embeddim (int): Embedding dimension
        num_layers (int, optional): Number of layers. Defaults to 4.
        num_heads (int, optional): Number of heads. Defaults to 4.
        dropout (float, optional): Defaults to 0.1.
        mlp_ratio (float, optional): Feed forward projection ratio. Defaults to 2.0.
    """
    def __init__(self, 
                 embeddim:int,
                 audio_seq:int,
                 num_layers:int=4, 
                 num_heads:int=4, 
                 dropout:float=0.1, 
                 mlp_ratio:float=2.0):
        super().__init__()

        self.AttnBlocks = nn.ModuleList([AttnBlock(embeddim, num_heads, dropout, mlp_ratio) for _ in range(num_layers)])
        self.PE = nn.Parameter(torch.rand(audio_seq, embeddim))

    def forward(self, audio_data:torch.Tensor):
        q = audio_data + self.PE[None, :, :]
        for layer in self.AttnBlocks:
            q = layer(q, q, q)
        return q
    
class AudioImageDecoder(nn.Module):
    """
    This modules takes `AudioSeqEncoder`'s output as Query, `ImageCNNEncoder`'s output as Key and Value.\\
    Learns to relationship between audio and image. 
    Args:
        embeddim (int): Embedding dimension
        num_layers (int, optional): Number of layers. Defaults to 4.
        num_heads (int, optional): Number of heads. Defaults to 4.
        dropout (float, optional): Defaults to 0.1.
        mlp_ratio (float, optional): Feed forward projection ratio. Defaults to 2.0.
    """
    def __init__(self,
                 embeddim:int,
                 image_seq:int,
                 num_layers:int=4,
                 num_heads:int=4,
                 dropout:float=0.1,
                 mlp_ratio:float=2.0):
        super().__init__()

        self.AttnBlocks = nn.ModuleList([AttnBlock(embeddim, num_heads, dropout, mlp_ratio) for _ in range(num_layers)])
        self.PE = nn.Parameter(torch.rand(image_seq, embeddim))

    def forward(self, audio_data:torch.Tensor, image_data:torch.Tensor):
        q = audio_data
        image_data = image_data + self.PE[None, :, :]
        for layer in self.AttnBlocks:
            q = layer(q, image_data, image_data)
        return q
    
class BboxContextDecoder(nn.Module):
    """
    This modules takes `BboxCNNEncoder`'s output as Query, `AudioImageDecoder`'s output as Key and Value.\\
    Aimed to learn relationship between bbox and image-audio. 
    Args:
        embeddim (int): Embedding dimension
        num_layers (int, optional): Number of layers. Defaults to 4.
        num_heads (int, optional): Number of heads. Defaults to 4.
        dropout (float, optional): Defaults to 0.1.
        mlp_ratio (float, optional): Feed forward projection ratio. Defaults to 2.0.
    """
    def __init__(self,
                 embeddim:int,
                 num_layers:int=4,
                 num_heads:int=4,
                 dropout:float=0.1,
                 mlp_ratio:float=2.0):
        super().__init__()

        self.AttnBlocks = nn.ModuleList([AttnBlock(embeddim, num_heads, dropout, mlp_ratio) for _ in range(num_layers)])

    def forward(self, bbox_data:torch.Tensor, image_data:torch.Tensor):
        q = bbox_data
        for layer in self.AttnBlocks:
            q = layer(q, image_data, image_data)
        return q
