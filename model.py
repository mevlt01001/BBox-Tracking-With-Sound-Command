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
            self.preprocess.num_mel_seq,
            self.n_mels,
            device=self.device
        )

        # TODO: Head
        self.color_linear = nn.Linear(self.embeddim, 4).to(self.device)
        self.geo_linear = nn.Linear(self.embeddim, 4).to(self.device)

    def forward(self, x:Tensor):
        x = self.preprocess(x)
        x = self.encoder(x)
        clr = self.color_linear(x)
        geo = self.geo_linear(x)
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
    Process audio data shaped like [batch_size, seqnum, embeddim], with CLS Token.\\
    CLS Token Learns the relation between audio features due to self-attention mechanism.

    Args:
        seqnum (int): Sequence number
        embeddim (int): Embedding dimension
        num_layers (int): Number of layers. Defaults to 4.
        num_heads (int): Number of heads. Defaults to 4.
        dropout (float): Defaults to 0.1.
        mlp_ratio (float): Feed forward projection ratio. Defaults to 2.0.
        patch_size (int): Patch size. Defaults to 16 accorting to Audio Spectrogram Transformer.
        patch_stride (int): Patch stride. Defaults to 10 accorting to Audio Spectrogram Transformer.
        melSeqNum (int): Mel Spectrogram Sequence Number. Defaults to 1000. Obtained from Preprocess.
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
                 melSeqNum:int=1000,
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
        self.num_mel_seq = melSeqNum
        self.nmels = nmels
        self.device = device

        # self.sa_blocks = nn.ModuleList(
        #     [
        #     nn.ModuleDict({
        #         "ln": nn.LayerNorm(embeddim),
        #         "mha": nn.MultiheadAttention(self.dim, self.num_heads, self.dropout, batch_first=True),
        #         "dropout": nn.Dropout(self.dropout)})
        #         for _ in range(self.num_layers)
        #     ] 
        # ).to(device)
        # """Multi Head Self-Attention Blocks"""

        # self.linear_blocks = nn.ModuleList([
        #     nn.Sequential(
        #         nn.LayerNorm(self.dim),
        #         nn.Linear(self.dim, int(self.dim * self.mlp_ratio)),
        #         nn.SiLU(),
        #         nn.Dropout(self.dropout),
        #         nn.Linear(int(self.dim * self.mlp_ratio), self.dim),
        #         nn.Dropout(self.dropout))
        #     for _ in range(num_layers)
        # ]).to(device)

        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=self.dim,
            stride=self.patch_stride,
            kernel_size=self.patch_size,
            bias=False
        ).to(device)

        self.encoder = nn.TransformerEncoder(
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

        self.set_seq()  
        self.CLS_TOKEN = nn.Parameter(torch.randn(1, 1, self.dim, requires_grad=True, device=device)*0.01)
        self.PE = nn.Parameter(torch.randn(1,self.num_seq+1, self.dim, requires_grad=True, device=device)*0.01)

    def forward(self, mel_spec_data:Tensor):
        # x.shape = (# [B, n_mels, 1000]
        B, *_ = mel_spec_data.shape
        CLS_TOKEN = self.CLS_TOKEN.repeat(B, 1, 1)     # [B, 1, dim]

        x = self.proj(mel_spec_data)                   # [B, dim, y_patchs, x_patchs]
        x = torch.flatten(x, start_dim=2, end_dim=-1)  # [B, dim, Seq]
        x = x.transpose(1,2)                           # [B, Seq, dim]
        x = torch.cat([CLS_TOKEN, x], dim=1) + self.PE      # [B, 1+Seq, dim]

        # for sa, linear in zip(self.sa_blocks, self.linear_blocks):
        #     res = x
        #     lnx = sa["ln"](x)
        #     x = sa["mha"](lnx, lnx, lnx)[0]
        #     x = res + sa["dropout"](x)
        #     x = x + linear(x)

        x = self.encoder(x)
        # print(f"AudioEncoder.shape: {x[:, 0].shape}, CLS_TOKEN.shape: {CLS_TOKEN.squeeze(1).shape}")
        return x[:, 0]
    
    @torch.no_grad()
    def set_seq(self):
        dummy = torch.rand(1,1,self.nmels,self.num_mel_seq, device=self.device)
        x = self.proj(dummy)                                # [B, dim, y_patchs, x_patchs]
        x = torch.flatten(x, start_dim=2, end_dim=-1)       # [B, dim, Seq]
        x = x.transpose(1,2)                                # [B, Seq, dim]
        self.num_seq = x.shape[1]
        del dummy, x
