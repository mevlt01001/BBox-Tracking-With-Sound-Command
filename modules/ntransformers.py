import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable

class VisualGrounding(nn.Module):
    """
    VisualGrounding is used to align the visual features with the audio features.

    This module uses Multi Head Self-Attention, Cross-Attention.\\
    Select the moost relavent visual features according to the audio features.

    `Encoder`: Firstly apply Multi Head Self-Attention to the audio features and visual features.\\
    `Decoder`: Secondly apply Cross-Attention to the audio features and visual features as mmemory.\\
    `Head`: Finally apply grid based sigmoid classification to the visual features.

    Args:
        imgsz (int): Image size
        embeddim (int): Embedding dimension
        audio_seq (int): Audio sequence number
        image_seq (int): Image sequence number
        num_layers (int): Number of layers. Defaults to 4.
        num_heads (int): Number of heads. Defaults to 4.
        dropout (float): Defaults to 0.1.
        mlp_ratio (float): Feed forward projection ratio. Defaults to 2.0.
    """

    def __init__(self, 
                 imgsz:int,
                 embeddim:int, 
                 audio_seq:int, 
                 image_seq:int, 
                 num_layers:int=4, 
                 num_heads:int=4, 
                 dropout:float=0.1, 
                 mlp_ratio:float=2.0
                 ):
        super().__init__()

        self.encoder = AudioEncoder(
            embeddim=embeddim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            mlp_ratio=mlp_ratio
        )
        self.decoder = AudioEncoder(
            embeddim=embeddim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            mlp_ratio=mlp_ratio
        )
        self.head = Head(
            imgsz=imgsz,
            embeddim=embeddim
        )

        self.image_PE = nn.Parameter(torch.rand(image_seq, embeddim)*0.2)
        self.audio_PE = nn.Parameter(torch.rand(audio_seq, embeddim)*0.2)
    
    def forward(self, audio_data:Tensor, image_data:Tensor, audio_mask:Tensor):
        audio_data = self.encoder(audio_data, audio_mask)
        image_data = self.decoder(image_data, audio_data)
        

class AudioEncoder(nn.Module):
    """Audio Sequence Encoder

    This is a basic Pre-Norm Multi Head Self-Attention Encoder.\\
    Process audio data shaped like [batch_size, seqnum, embeddim], after AudioCNNFeats.\\
    Learns the relation between audio features.

    Args:
        seqnum (int): Sequence number
        embeddim (int): Embedding dimension
        num_layers (int): Number of layers. Defaults to 4.
        num_heads (int): Number of heads. Defaults to 4.
        dropout (float): Defaults to 0.1.
        mlp_ratio (float): Feed forward projection ratio. Defaults to 2.0.
    """
    def __init__(self, 
                 embeddim:int,
                 num_heads:int=8,
                 num_layers:int=4,
                 dropout:float=0.1,
                 mlp_ratio:float=4.0):
        super().__init__()

        self.dim = embeddim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio

        self.sa_blocks = nn.ModuleList(
            [
            nn.ModuleDict({
                "ln": nn.LayerNorm(embeddim),
                "mha": nn.MultiheadAttention(self.dim, self.num_heads, self.dropout, batch_first=True),
                "dropout": nn.Dropout(self.dropout)})
                for _ in range(self.num_layers)
            ] 
        )
        """Multi Head Self-Attention Blocks"""

        self.linear_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, int(self.dim * self.mlp_ratio)),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(int(self.dim * self.mlp_ratio), self.dim),
                nn.Dropout(self.dropout))
            for _ in range(num_layers)
        ])

    def forward(self, audio_data:Tensor, key_padding_mask:Tensor):
        x = audio_data
        for sa, linear in zip(self.sa_blocks, self.linear_blocks):
            lnx = sa["ln"](x)
            x = sa["mha"](lnx, lnx, lnx, key_padding_mask)[0]
            x = x + sa["dropout"](x)
            x = x + linear(x)
        return x
    
class ImageDecoder(nn.Module):
    """Image Sequence Decoder

    This is a basic Pre-Norm Multi Head Cross-Attention Decoder.\\
    Process image data shaped like [batch_size, seqnum, embeddim], after ImageCNNFeats.\\
    Learns the relation between image and audio features.

    Args:
        seqnum (int): Sequence number
        embeddim (int): Embedding dimension
        num_layers (int): Number of layers. Defaults to 4.
        num_heads (int): Number of heads. Defaults to 4.
        dropout (float): Defaults to 0.1.
        mlp_ratio (float): Feed forward projection ratio. Defaults to 2.0.
    """
    def __init__(self, 
                 embeddim:int,
                 num_heads:int=8,
                 num_layers:int=4,
                 dropout:float=0.1,
                 mlp_ratio:float=4.0):
        super().__init__()

        self.dim = embeddim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio

        self.sa_blocks = nn.ModuleList(
            [
            nn.ModuleDict({
                "ln": nn.LayerNorm(embeddim),
                "mha": nn.MultiheadAttention(self.dim, self.num_heads, self.dropout, batch_first=True),
                "dropout": nn.Dropout(self.dropout)})
                for _ in range(self.num_layers)
            ] 
        )
        """Multi Head Self-Attention Blocks"""

        self.ca_blocks = nn.ModuleList(
            [
            nn.ModuleDict({
                "ln": nn.LayerNorm(embeddim),
                "mha": nn.MultiheadAttention(self.dim, self.num_heads, self.dropout, batch_first=True),
                "dropout": nn.Dropout(self.dropout)})
                for _ in range(self.num_layers)
            ] 
        )
        """Multi Head Cross-Attention Blocks"""

        self.linear_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, int(self.dim * self.mlp_ratio)),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(int(self.dim * self.mlp_ratio), self.dim),
                nn.Dropout(self.dropout))
            for _ in range(num_layers)
        ])

    def forward(self, image_data:Tensor, audio_data:Tensor):
        x = image_data
        for sa, ca, linear in zip(self.sa_blocks, self.ca_blocks, self.linear_blocks):

            lnx = sa["ln"](x)
            x = sa["mha"](lnx, lnx, lnx)[0]
            x = x + sa["dropout"](x)
            
            lnx = ca["ln"](x)
            x = ca["mha"](lnx, audio_data, audio_data)[0]
            x = x + ca["dropout"](x)
            x = x + linear(x)

        return x

class Head(nn.Module):
    """Grid Based Classification Head"""
    def __init__(self, 
                 imgsz:int,
                 embeddim:int):
        super().__init__()

        self.imgsz = imgsz
        self.embeddim = embeddim
        self.linear = nn.Linear(embeddim, 1)

    def forward(self, image_data:Tensor, threshold:float=0.5):
        # image_data.shape = [B, 400, embeddim]
        image_data = self.linear(image_data)
        if self.training:
            return image_data
        pixel_data = torch.sigmoid(image_data)
        pixel_data = (pixel_data > threshold)
        return pixel_data




    
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
        self.linear = nn.Linear(num_layers, 1)

    def forward(self, audio_data:torch.Tensor, image_data:torch.Tensor):
        q = audio_data
        image_data = image_data + self.PE[None, :, :]
        outs = []
        for layer in self.AttnBlocks:
            outs.append(layer(q, image_data, image_data))
        q = self.linear(torch.cat(outs, dim=0))
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
