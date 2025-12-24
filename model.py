import torch
import torchaudio
import torch.nn as nn
from torch import Tensor
from .preprocess import Preprocess
from .trainer import Trainer

class Model(nn.Module):
    def __init__(self,
                 sr,
                 win_length_second,
                 stride_second,
                 n_mels,
                 embeddim,
                 patch_size,
                 patch_stride,
                 max_second,
                 num_heads,
                 num_layers,
                 dropout,
                 mlp_ratio,
                 device=torch.device('cpu'),
                 ):
        super().__init__()

        self.sr = sr
        self.win_length_second = win_length_second
        self.stride_second = stride_second
        self.n_mels = n_mels
        self.embeddim = embeddim
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.max_second = max_second
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio
        self.device = device
        
        # TODO: Preprocess
        self.preprocess = Preprocess(
            self.sr,
            self.win_length_second,
            self.stride_second,
            self.n_mels,
            self.embeddim,
            self.patch_size,
            self.patch_stride,
            self.max_second,
            device=self.device
        )
        # TODO: AudioEncoder: UPDATE: Use nn.TransformerEncoderLayer
        self.encoder = AudioEncoder(
            self.embeddim,
            self.num_heads,
            self.num_layers,
            self.dropout,
            self.mlp_ratio,
            self.patch_size,
            self.patch_stride,
            self.preprocess.s2_max,
            self.n_mels,
            device=self.device
        )

        # TODO: Head
        self.color_linear = nn.Linear(self.embeddim, 4).to(self.device)
        self.geo_linear = nn.Linear(self.embeddim, 4).to(self.device)

    def forward(self, x:Tensor,         # x.shape = [B, Waveform]
                lenghts:Tensor=None):   # lenghts.shape = [B]
        
        x, masks = self.preprocess(x, lenghts)  # x.shape = [B, s2_max, embeddim], masks.shape = [B, s2_max]
        x = self.encoder(x, masks)              # x.shape = [B, 2+s2_max, embeddim]
        clr = self.color_linear(x[:, 0])        # clr.shape = [B, 4]
        geo = self.geo_linear(x[:, 1])          # geo.shape = [B, 4]
        if self.training:
            return clr, geo
        clr = torch.softmax(clr, dim=-1)
        geo = torch.softmax(geo, dim=-1)
        return clr, geo
    
    def train(self, mode=True,**kwargs):
        """Set the module in training mode or train a model.

        To train please provide the following kwargs:
        - audios_path (str)
        - labels_path (str)
        - epochs (int)
        - batch_size (int)
        - lr (float)

        This has an effect only on certain modules. See the documentation of
        particular modules for details of their behaviors in training/evaluation
        mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation mode (``False``). Default: ``True``.
            images_path (str): the path of images.
            audios_path (str): the path of audios.
            labels_path (str): the path of labels.
            epochs (int): number of epochs
            batch_size (int): batch size
            lr (float): learning rate

        Returns:
            Module: self
        """
        if "labels_dir" and "epochs" in kwargs:

            labels_dir = kwargs.get("labels_dir")
            audios_dir = kwargs.get("audios_dir")
            epochs = kwargs.get("epochs")
            valid = kwargs.get("valid", True)
            batch_size = kwargs.get("batch_size", 12)
            valid_ratio = kwargs.get("valid_ratio", 0.2)
            lr = kwargs.get("lr", 0.001)
            min_lr = kwargs.get("min_lr", 0.0001)
            log_dir = kwargs.get("log_dir", "runs/sound_control")

            trainer = Trainer(
                model=self,
                labels_dir=labels_dir,
                audios_dir=audios_dir,
                batch_size=batch_size,
                lr=lr,
                min_lr=min_lr,
                epochs=epochs,
                test_ratio=valid_ratio,
                valid=valid,
                log_dir=log_dir,
                device=self.device
            )
            
            self = trainer.train()
        else:
            return super().train(mode)

    @classmethod
    def load(self, path:str, device:torch.device=torch.device('cpu')):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        """
        torch.save({
            'state_dict': self.model.state_dict(),
            'sr': self.model.sr,
            'win_length_second': self.model.win_length_second,
            'stride_second': self.model.stride_second,
            'n_mels': self.model.n_mels,
            'embeddim': self.model.embeddim,
            'patch_size': self.model.patch_size,
            'patch_stride': self.model.patch_stride,
            'max_second': self.model.max_second,
            'num_heads': self.model.num_heads,
            'num_layers': self.model.num_layers,
            'dropout': self.model.dropout,
            'mlp_ratio': self.model.mlp_ratio,
        }, f"checkpoints/model_epoch_{epoch}.pt")
        """
        state = ckpt['state_dict']
        model = self(
            sr=ckpt['sr'],
            win_length_second=ckpt['win_length_second'],
            stride_second=ckpt['stride_second'],
            n_mels=ckpt['n_mels'],
            embeddim=ckpt['embeddim'],
            patch_size=ckpt['patch_size'],
            patch_stride=ckpt['patch_stride'],
            max_second=ckpt['max_second'],
            num_heads=ckpt['num_heads'],
            num_layers=ckpt['num_layers'],
            dropout=ckpt['dropout'],
            mlp_ratio=ckpt['mlp_ratio'],
            device=device
        )
        model.load_state_dict(state)
        return model

class AudioEncoder(nn.Module):
    """Audio Sequence Encoder

    This is a basic Pre-Norm Multi Head Self-Attention Encoder.\\
    Process audio data shaped like [batch_size, s2_max, embeddim], with CLR and GEO Token.\\
    CLS and GEO Token Learns the relation between audio features due to self-attention mechanism.

    Args:
        embeddim (int): Embedding dimension
        num_layers (int): Number of layers. Defaults to 4.
        num_heads (int): Number of heads. Defaults to 4.
        dropout (float): Defaults to 0.1.
        mlp_ratio (float): Feed forward projection ratio. Defaults to 2.0.
        patch_size (int): Patch size. Defaults to 16 accorting to Audio Spectrogram Transformer.
        patch_stride (int): Patch stride. Defaults to 10 accorting to Audio Spectrogram Transformer.
        s2_max (int): Conv1d sequence number calculated after Mel Spectrogram Sequence Number. Defaults to 1000. Obtained from Preprocess.
        nmels (int): Number of Mel Bands. Defaults to 128. Obtained from Preprocess.
    """
    def __init__(self, 
                 embeddim:int,
                 num_heads:int=8,
                 num_layers:int=4,
                 dropout:float=0.1,
                 mlp_ratio:float=4.0,
                 patch_size:int=16,
                 patch_stride:int=10,
                 s2_max:int=1000,
                 nmels:int=128,
                 device:torch.device=torch.device('cpu')):
        super().__init__()

        self.dim = embeddim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.s2_max = s2_max
        self.nmels = nmels
        self.device = device

        self.audio_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.dim,
                nhead=self.num_heads,
                dim_feedforward=int(self.dim * self.mlp_ratio),
                dropout=self.dropout,
                batch_first=True,
                norm_first=True
            ),
            num_layers=self.num_layers
        ).to(device)

        self.CLR_TOKEN = nn.Parameter(torch.randn(1, 1, self.dim, requires_grad=True, device=device)*0.01)
        self.GEO_TOKEN = nn.Parameter(torch.randn(1, 1, self.dim, requires_grad=True, device=device)*0.01)
        self.PE = nn.Parameter(torch.randn(1,self.s2_max+2, self.dim, requires_grad=True, device=device)*0.01)

    def forward(self, x:Tensor,             # x.shape = [B, s2_max, dim]
                key_padding_mask:Tensor):   # key_padding_mask.shape = [B, s2_max]
        
        B, *_ = x.shape     # [B, s2_max, dim]
        CLR_TOKEN = self.CLR_TOKEN.expand(x.shape[0], -1, -1)     # [B, 1, dim]
        GEO_TOKEN = self.GEO_TOKEN.expand(x.shape[0], -1, -1)     # [B, 1, dim]
        key_padding_mask = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=self.device), key_padding_mask], dim=1).to(self.device)
        x = torch.cat([CLR_TOKEN, GEO_TOKEN, x], dim=1) + self.PE
        x = self.audio_encoder(x, src_key_padding_mask=key_padding_mask)  # [B, 2+s2_max, dim]
        return x
    
