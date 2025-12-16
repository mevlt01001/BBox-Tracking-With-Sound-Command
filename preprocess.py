import torch, torchaudio
from torch import Tensor
import torch.nn as nn

def load_audios(path:str|list[str], target_sr=None, max_seconds:int=None):
    audios = []
    if isinstance(path, str):
        path = [path]

    for path in path:
        padding = 0
        data, sr = torchaudio.load(path)
        if target_sr is not None:
            data = preproecss(data, sr, target_sr)
            sr = target_sr
        if max_seconds is not None:
            if data.shape[1] > max_seconds*sr:
                data = data[:, :max_seconds*sr]
            padding = max_seconds*sr - data.shape[1]
            data = torch.nn.functional.pad(data, (0,padding,0,0), mode='constant', value=0.0)
        audios.append((data))
    else:
        audios = torch.cat(audios, dim=0)
    return audios

def preproecss(audio:Tensor, sr:int, target_sr:int=16000):
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    audio = resampler.forward(audio)
    return audio

def patcher(x):
    pass

class Preprocess(nn.Module):
    def __init__(self, 
                 sr:int=16000,
                 win_length_second:float=0.025,
                 stride_second:float=0.01,
                 n_mels:int=128,
                 embeddim:int=768,
                 patch_size:int=16,
                 patch_stride:int=10,
                 device:torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 ):
        super().__init__()

        self.sr = sr                                # 16KHz for AST (Audio Spectrogram Transformer)
        self.win_length_second = win_length_second  # 25ms for AST (Audio Spectrogram Transformer)
        self.stride_second = stride_second          # 10ms for AST (Audio Spectrogram Transformer)
        self.n_mels = n_mels                        # 128 for AST (Audio Spectrogram Transformer)
        self.embeddim = embeddim                    # 768 for AST (Audio Spectrogram Transformer)
        self.patch_size = patch_size                # 16 for AST (Audio Spectrogram Transformer)
        self.patch_stride = patch_stride            # 10 for AST (Audio Spectrogram Transformer)

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=int(sr * self.win_length_second),
            win_length=int(sr * self.win_length_second),
            hop_length=int(self.sr * self.stride_second),
            n_mels=self.n_mels,
            f_min=0,
            f_max=self.sr // 2,
            power=2.0,
            normalized=True,
            center=False
        ).to(device)

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB("power", top_db=80).to(device)

        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=self.embeddim,
            stride=self.patch_stride,
            kernel_size=self.patch_size,
            bias=False
        ).to(device)

    def forward(self, x:Tensor):
        # x.shape = (B, 1, self.max_seconds*self.sr)
        x = self.mel_spectrogram.forward(x).unsqueeze(1)    # [B, n_mels, 1000]
        x = self.amplitude_to_db.forward(x)                 # [B, n_mels, 1000]
        x = self.proj(x)                                    # [B, embeddim, y_patchs, x_patchs]
        x = torch.flatten(x, start_dim=2, end_dim=-1)       # [B, embeddim, Seq]
        x = x.transpose(1,2)                                # [B, Seq, embeddim]

        return x

file_list = [
    "/home/neuron/workspace/sound_control/dataset_CLR_GEO/augmented_audios/11599.wav",
    "/home/neuron/workspace/sound_control/dataset_CLR_GEO/augmented_audios/11354.wav",
    "/home/neuron/workspace/sound_control/dataset_CLR_GEO/augmented_audios/10653.wav",
    "/home/neuron/workspace/sound_control/dataset_CLR_GEO/augmented_audios/00653.wav",
    "/home/neuron/workspace/sound_control/dataset_CLR_GEO/augmented_audios/02053.wav",
    "/home/neuron/workspace/sound_control/dataset_CLR_GEO/augmented_audios/02833.wav",
]
data = load_audios(file_list, target_sr=16000, max_seconds=10)


import matplotlib.pyplot as plt

mdl = Preprocess(sr=16000, win_length_second=0.025, stride_second=0.010, embeddim=512, device="cpu")
data = mdl.forward(data)
print(data.shape)