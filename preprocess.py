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

class Preprocess(nn.Module):
    def __init__(self, 
                 sr:int=16000,
                 win_length_second:float=0.025,
                 stride_second:float=0.01,
                 n_mels:int=128,
                 embeddim:int=768,
                 patch_size:int=16,
                 patch_stride:int=10,
                 max_second:int=10,
                 device:torch.device = torch.device('cpu'),
                 ):
        super().__init__()
        self.device = device
        self.max_seconds = max_second

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
        self.set_seq() # self.out.shape = [B,1,N_MELS,SEQ]

    @torch.no_grad()
    def forward(self, x:Tensor):
        # x.shape = (B, self.max_seconds*self.sr)
        x = self.mel_spectrogram.forward(x).unsqueeze(1)    # [B, 1, n_mels, 1000]
        x = self.amplitude_to_db.forward(x)                 # [B, 1, n_mels, 1000]
        return x
    
    @torch.no_grad()
    def set_seq(self):
        dummy = torch.rand(1,self.max_seconds*self.sr, device=self.device)
        out = self.forward(dummy)
        self.num_mel_seq = out.shape[-1]
        del dummy, out