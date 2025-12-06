import torch
import torch.nn as nn
from feats import AudioCNNFeats, ImageCNNEncoder, BboxCNNEncoder

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
                 num_layers:int=4, 
                 num_heads:int=4, 
                 dropout:float=0.1, 
                 mlp_ratio:float=2.0):
        super().__init__()

        self.AttnBlocks = nn.ModuleList([AttnBlock(embeddim, num_heads, dropout, mlp_ratio) for _ in range(num_layers)])

    def forward(self, audio_data:torch.Tensor):
        q = audio_data
        for layer in self.AttnBlocks:
            q = layer(q, q, q)
        return q
    
class AudioImageDecoder(nn.Module):

    def __init__(self,
                 embeddim:int,
                 num_layers:int=4,
                 num_heads:int=4,
                 dropout:float=0.1,
                 mlp_ratio:float=2.0):
        super().__init__()

        self.AttnBlocks = nn.ModuleList([AttnBlock(embeddim, num_heads, dropout, mlp_ratio) for _ in range(num_layers)])

    def forward(self, audio_data:torch.Tensor, image_data:torch.Tensor):
        q = audio_data
        for layer in self.AttnBlocks:
            q = layer(q, image_data, image_data)
        return q
    
class BboxImageDecoder(nn.Module):

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


class Model(nn.Module):
    def __init__(self,
                 embeddim:int,
                 num_layers:int=4,
                 num_heads:int=4,
                 dropout:float=0.1,
                 mlp_ratio:float=2.0,
                 n_mels:int = 256,
                 sr:int = 25500,
                 ):
        super().__init__()

        self.sr = sr
        self.n_mels = n_mels
        self.embeddim = embeddim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        self.audio_cnn_feats = AudioCNNFeats(embeddim, n_mels, sr)
        self.image_encoder = ImageCNNEncoder(embeddim)
        self.bbox_encoder = BboxCNNEncoder(embeddim)

        self.audio_encoder = AudioSeqEncoder(embeddim, num_layers, num_heads, dropout, mlp_ratio)
        self.audio_image_decoder = AudioImageDecoder(embeddim, num_layers, num_heads, dropout, mlp_ratio)
        self.bbox_image_decoder = BboxImageDecoder(embeddim, num_layers, num_heads, dropout, mlp_ratio)

        self.lineer1 = nn.Linear(embeddim, 1)

    def forward(self, audio_data:torch.Tensor, image_data:torch.Tensor, bbox_data:torch.Tensor, bbox_mask:torch.Tensor):
        """

        Args:
            audio_data (torch.Tensor): Raw wave form data shapded like [batch_size, wave_data]. Padded to wave_data with zeros. 
            image_data (torch.Tensor): [batch_size, 3, 640, 640]
            bbox_data (torch.Tensor): CXCYWH Bounding boxes shaped like [batch_size, S_B, 4]. S_B is the number/sequence of bounding boxes.
            bbox_mask (torch.Tensor): [batch_size, S_B]. This boolean or binary tensor indicates which bounding boxes are ignored. Since we want to batched inference, we need to same shape as bbox_data that is why we need this mask.

        Returns:
            pass
        """

        #Audio_data.shape = [B, wave_data]
        #Image_data.shape = [B, 3, 640, 640]
        #Bbox_data.shape = [B, S_B, 4]
        #Bbox_mask.shape = [B, S_B]

        audio_data = self.audio_cnn_feats(audio_data) # Audio_data.shape = [B, S_A, embeddim]
        image_data = self.image_encoder(image_data)   # Image_data.shape = [B, S_I, embeddim]
        bbox_data = self.bbox_encoder(bbox_data)      # Bbox_data.shape  = [B, S_B, embeddim]

        audio_data = self.audio_encoder(audio_data)
        audio_image_data = self.audio_image_decoder(audio_data, image_data) 
        audio_image_bbox_data = self.bbox_image_decoder(bbox_data, audio_image_data)

        # Audio_data.shape = [B, S_A, embeddim]
        # audio_image_data.shape = [B, S_A, embeddim]
        # audio_image_bbox_data.shape = [B, S_B, embeddim]

        selected_bboxes = self.lineer1(audio_image_bbox_data).squeeze(-1)
        # selected_bboxes.shape = [B, S_B] -> Softmax -> CalcLoss

        return selected_bboxes


        