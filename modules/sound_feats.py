import torch, torchaudio
import torch.nn as nn

class AudioFeatureExtractor(nn.Module):
    def __init__(self, out_channel=512, n_mels=128):
        super().__init__()

        self.audio_feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=n_mels, out_channels=128, kernel_size=7, stride=3, padding=1),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Conv1d(in_channels=256, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channel),
            nn.SiLU(),
        )
        # Accorting to my configuration: 
        # win_length = 500, hop_length = 250, n_fft = 2048, sample_rate = 25500, n_mels = 128 
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(win_length=500, hop_length=250, n_fft=2048, sample_rate=25500,n_mels=n_mels).to(torch.device("cuda"))


    def forward(self, x):
        x = self.mel_spectrogram(x)
        x = torch.log(x + 1e-10)
        x = self.audio_feature_extractor(x)
        return x

