import torch
import torch.nn as nn
from torchvision.models import resnet
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import AmplitudeToDB

class AudioCNNFeats(nn.Module):
    """
    Audio Feature Extractor with mel-spectrogram.
    Args:
        out_channel (int, optional): Output channel/embedding dimension. Defaults to 512.
        n_mels (int, optional): Number of mel-bands. Defaults to 256.
        sr (int, optional): Sampling rate. Defaults to 25500.
    """
    def __init__(self, out_channel=512, n_mels=256, sr=25500):
        super().__init__()
        
        k1 = max(n_mels, 256)
        k2 = max(k1, 256)

        self.audio_feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=n_mels, out_channels=k1, kernel_size=13, stride=3),
            nn.BatchNorm1d(k1),
            nn.SiLU(),
            nn.Conv1d(in_channels=k1, out_channels=k2, kernel_size=7, stride=3),
            nn.BatchNorm1d(k2),
            nn.SiLU(),
            nn.Conv1d(in_channels=k1, out_channels=out_channel, kernel_size=3, stride=1),
            nn.BatchNorm1d(out_channel),
            nn.SiLU(),
        )

        self.mel_spectrogram = MelSpectrogram(
            sample_rate=sr,
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512,
            center=True,
            normalized=True,
        )

        self.amplitude_to_db = AmplitudeToDB(top_db=80.0)

    def forward(self, x):
        # X.shape = (B, 1, 459000)
        x = self.mel_spectrogram(x) # X.shape = (B, n_mels, time)
        x = self.amplitude_to_db(x)
        x = self.audio_feature_extractor(x) # X.shape = (B, embeddim, time)
        x = x.permute(0, 2, 1)
        # x.shape = (B, 128, embeddim)
        print(f"AudioCNNFeats.shape: {x.shape}")
        return x
    
class ImageCNNEncoder(nn.Module):
    """
    Image Feature Extractor with ResNet18. This extractor is pretrained on ImageNet.
    Args:
        out_channel (int, optional): Output channel/embedding dimension. Defaults to 512.
    """
    def __init__(self, out_channel=512):
        super().__init__()
        
        self.resnet18 = nn.Sequential(
            *list(resnet.resnet18(weights=resnet.ResNet18_Weights.IMAGENET1K_V1).children())[:-2]
            )
        
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Conv2d(in_channels=512, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.SiLU(),
        )

    def forward(self, x:torch.Tensor):
        # X.shape = (B, 3, 640, 640)
        x = self.resnet18(x)
        x = self.cnn_encoder(x)
        x = x.flatten(-2, -1).permute(0, 2, 1)
        # x.shape = (B, 400, embeddim)
        print(f"ImageCNNEncoder.shape: {x.shape}")
        return x

class BboxCNNEncoder(nn.Module):
    """
    Bbox Feature Extractor with Conv1d. Used to extract bbox features and give them embeddings.
    Args:
        out_channel (int, optional): Output channel/embedding dimension. Defaults to 512.
    """
    def __init__(self, out_channel=512):
        super().__init__()
        
        self.bbox_encoder = nn.Sequential(
            nn.Linear(in_features=4, out_features=64),
            nn.SiLU(),
            nn.Linear(in_features=64, out_features=out_channel),
            nn.SiLU()
        )

    def forward(self, x:torch.Tensor):
        # X.shape = (B, N, 4)
        x = self.bbox_encoder(x)
        print(f"BboxCNNEncoder.shape: {x.shape}")
        return x
