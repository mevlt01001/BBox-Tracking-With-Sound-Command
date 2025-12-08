import torch, os
import torch.nn as nn
from .feats import AudioCNNFeats, ImageCNNEncoder, BboxCNNEncoder
from .ntransformers import AudioSeqEncoder, AudioImageDecoder, BboxContextDecoder
from .trainer import Trainer

class Model(nn.Module):
    def __init__(self,
                 imgsz:int,
                 embeddim:int,
                 num_layers:int=4,
                 num_heads:int=4,
                 dropout:float=0.1,
                 mlp_ratio:float=2.0,
                 n_mels:int = 256,
                 max_seconds:int = 10,
                 sr:int = 25500,
                 bbox_size:int = 20,
                 device:torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 ):
        super().__init__()

        self.imgsz = imgsz
        self.sr = sr
        self.max_seconds = max_seconds
        self.n_mels = n_mels
        self.embeddim = embeddim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.bbox_size = bbox_size
        self.device = device

        self.audio_cnn_feats = AudioCNNFeats(embeddim, n_mels, sr)
        self.image_encoder = ImageCNNEncoder(embeddim)
        self.bbox_encoder = BboxCNNEncoder(embeddim)
        self.set_seqs()

        self.audio_encoder = AudioSeqEncoder(embeddim, self.audio_seq, num_layers, num_heads, dropout, mlp_ratio)
        self.audio_image_decoder = AudioImageDecoder(embeddim, self.image_seq, num_layers, num_heads, dropout, mlp_ratio)
        self.bbox_image_decoder = BboxContextDecoder(embeddim, num_layers, num_heads, dropout, mlp_ratio)

        self.lineer1 = nn.Linear(embeddim, 1)


    def forward(self, audio_data:torch.Tensor, image_data:torch.Tensor, bbox_data:torch.Tensor, bbox_mask:torch.Tensor):
        """

        Args:
            audio_data (torch.Tensor): Raw wave form data shapded like [batch_size, wave_data]. Padded to wave_data with zeros. 
            image_data (torch.Tensor): [batch_size, 3, 640, 640]
            bbox_data (torch.Tensor): CXCYWH Bounding boxes shaped like [batch_size, S_B, 4]. S_B is the number/sequence of bounding boxes.
            bbox_mask (torch.Tensor): [batch_size, S_B]. This boolean or binary tensor indicates which bounding boxes are ignored. Since we want to batched inference, we need to same shape as bbox_data that is why we need this mask.

        Returns:
            selected_bboxes_idx (torch.Tensor): Selected bounding boxes idx shaped like [batch_size, S_B].
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

        selected_bboxes_idx = self.lineer1(audio_image_bbox_data).squeeze(-1)
        # selected_bboxes_idx.shape = [B, S_B]

        bbox_logits = selected_bboxes_idx.masked_fill(bbox_mask, float('-inf'))
        # selected_bboxes_idx.shape = [B, S_B]
        if self.training:
            return bbox_logits

        probs = torch.sigmoid(bbox_logits)

        return probs
    
    @torch.no_grad()
    def set_seqs(self):
        audio_dummy = torch.rand(1, self.sr*self.max_seconds)
        image_dummy = torch.rand(1, 3, self.imgsz, self.imgsz)
        out = self.audio_cnn_feats.forward(audio_dummy)
        self.audio_seq = out.shape[1]
        out = self.image_encoder.forward(image_dummy)
        self.image_seq = out.shape[1]
        del audio_dummy, image_dummy, out

    
    def train(self, mode=True,**kwargs):
        """Set the module in training mode or train a model.

        To train please provide the following kwargs:
        - images_path (str)
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
        if "images_path" and "audios_path" and "labels_path" and "epochs" and "batch_size" and "lr" in kwargs:
            images_path = kwargs.get("images_path")
            audios_path = kwargs.get("audios_path")
            labels_path = kwargs.get("labels_path")
            epochs = kwargs.get("epochs")
            batch_size = kwargs.get("batch_size")
            valid_ratio = kwargs.get("valid_ratio")
            valid = kwargs.get("valid")
            lr = kwargs.get("lr")
            log_dir = kwargs.get("log_dir", "runs/sound_control")

            trainer = Trainer(
                model=self.to(self.device), 
                images_path=images_path, 
                audios_path=audios_path, 
                labels_path=labels_path, 
                device=self.device, 
                valid_ratio=valid_ratio, 
                valid=valid, 
                log_dir=log_dir)
            
            self = trainer.train(epochs, batch_size, lr)
        else:
            return super().train(mode)

    @classmethod
    def load(self, checkpoint_path:os.PathLike, device="cpu"):
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        cls = self(
            imgsz=ckpt["imgsz"],
            embeddim=ckpt["embeddim"],
            num_layers=ckpt["num_layers"],
            num_heads=ckpt["num_heads"],
            dropout=ckpt["dropout"],
            mlp_ratio=ckpt["mlp_ratio"],
            n_mels=ckpt["n_mels"],
            max_seconds=ckpt["max_seconds"],
            sr=ckpt["sr"],
            bbox_size=ckpt["bbox_size"],
            device=device
        )
        cls.load_state_dict(ckpt["state_dict"])
        return cls
        