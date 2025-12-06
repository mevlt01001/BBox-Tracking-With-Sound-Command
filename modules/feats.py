import torch, torchaudio
import torch.nn as nn
from torchvision.models import resnet
from nnAudio.features import MelSpectrogram

class AudioCNNFeats(nn.Module):
    def __init__(self, out_channel=512, n_mels=256, sr=25500):
        super().__init__()
        
        k1 = max(n_mels, 128)
        k2 = max(k1, 256)

        self.audio_feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=n_mels, out_channels=k1, kernel_size=7, stride=3),
            nn.GroupNorm(1,k1),
            nn.SiLU(),
            nn.Conv1d(in_channels=k1, out_channels=k2, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(1, k2),
            nn.SiLU(),
            nn.Conv1d(in_channels=k2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, out_channel),
            nn.SiLU(),
        )

        self.mel_spectrogram = MelSpectrogram(
            sr=sr,
            n_fft=2048,
            n_mels=n_mels,
            hop_length=512,
            center=True,
            verbose=False
        )

    def forward(self, x):
        # X.shape = (B, 1, 459000)
        x = self.mel_spectrogram.forward(x)
        x = torch.log(x + 1e-10)
        print(x.shape)
        x = self.audio_feature_extractor(x)
        x = x.permute(0, 2, 1)
        # x.shape = (B, 128, embeddim)
        return x
    
class ImageCNNEncoder(nn.Module):
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
        return x

class BboxCNNEncoder(nn.Module):
    def __init__(self, out_channel=512):
        super().__init__()
        
        self.bbox_encoder = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, 256),
            nn.SiLU(),
            nn.Conv1d(in_channels=256, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, out_channel),
            nn.SiLU(),
        )

    def forward(self, x:torch.Tensor):
        # X.shape = (B, N, 4)
        x = x.permute(0, 2, 1)
        x = self.bbox_encoder(x)
        x = x.permute(0, 2, 1)
        return x

AudioEncoder = AudioCNNFeats(512, 512, 25500)
ImageEncoder = ImageCNNEncoder(512)
BBoxEncoder = BboxCNNEncoder(512)

wav, sr = torchaudio.load("dataset/augmented_audios/03234_009.wav")
# print(wav.shape, wav.dtype)

data = torch.rand(4,16567)
print(AudioEncoder(wav).shape)