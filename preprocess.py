import random, os
import torch, torchaudio
from torch import Tensor
import torch.nn as nn

rir_list_path = "RIRS_NOISES/simulated_rirs/smallroom/rir_list"
noise_list_path = "RIRS_NOISES/pointsource_noises/noise_list"
rir_root = ""

with open(rir_list_path, "r") as f:
    rir_files = f.read().splitlines()

for idx, rir_file in enumerate(rir_files):
    rir_files[idx] = rir_file.split(" ")[-1]

with open(noise_list_path, "r") as f:
    noise_files = f.read().splitlines()

noises = []
for idx, noise_file in enumerate(noise_files):
    noise = noise_file.split(" ")[-1]
    _type = noise_file.split(" ")[-2]
    if _type == "background":
        noises.append(noise)

# DATA AUGMENTATION FUNCTIONS

def add_space(audio, sr, begin_space=0.5, end_space=0.5):
    """
    Adds space to the beginning and end of the audio.
    Args:
        audio (np.ndarray): Audio array.
        sr (int): Sample rate.
        begin_space (float): Space to add to the beginning of the audio in seconds.
        end_space (float): Space to add to the end of the audio in seconds.
    Returns:
        np.ndarray: Audio array with space added.
    """
    begin_space = int(begin_space * sr)
    end_space = int(end_space * sr)
    audio = torch.cat((torch.zeros(1, begin_space), audio, torch.zeros(1, end_space)), dim=1)
    return audio

def add_noise(clean_audio:torch.Tensor, noise:torch.Tensor, snr_db:float):
    """
    Adds noise to the audio.
    Args:
        clean_waveform (torch.Tensor): Clean audio waveform.
        noise_waveform (torch.Tensor): Noise audio waveform.
        snr_db (float): SNR (Signal-to-Noise Ratio) in decibels.
    Returns:
        torch.Tensor: Noisy audio waveform.
    """
    
    noisy_waveform = torchaudio.functional.add_noise(clean_audio, noise, snr=torch.Tensor([snr_db]))

    max_val = torch.abs(noisy_waveform).max()
    if max_val > 1.0:
        noisy_waveform = noisy_waveform / max_val
        
    return noisy_waveform

def add_rir(audio, kernel):
    """
    Adds RIR(Room Impulse Respond) to given audio.
    Args:
        audio (torch.Tensor): (Clean)Audio waveform.
        kernel (torch.Tensor): Room Impulse Respond kernel.
    Returns:
        torch.Tensor: Affected audio waveform by Room Impulse respond Kernel
    """
    n = audio.shape[-1]
    m = kernel.shape[-1]
    target_length = n + m - 1

    kernel = kernel / (torch.norm(kernel, p=2) + 1e-9)
    signal_f = torch.fft.rfft(audio, n=target_length)
    kernel_f = torch.fft.rfft(kernel, n=target_length)
    
    output_f = signal_f * kernel_f
    
    output = torch.fft.irfft(output_f, n=target_length)
    
    return output[..., :n]

def load_audios(path:str|list[str], 
                max_seconds:int, 
                target_sr=None,
                begin_space:float=0.0,
                end_space:float=0.0, 
                rir_ratio:float=0.0, 
                noise_ratio:float=0.0,
                snr:int=10):
    
    audios = []
    lenghts = []
    if isinstance(path, str):
        path = [path]

    for path in path:
        padding = 0
        data, sr = torchaudio.load(path)
        data = add_space(data, sr, begin_space, end_space)
        rir, rir_sr = torchaudio.load(os.path.join(rir_root, random.choice(rir_files)))
        noise, noise_sr = torchaudio.load(os.path.join(rir_root, random.choice(noises)))
        
        
        if target_sr is not None:
            data = preproecss(data, sr, target_sr)
            rir = preproecss(rir, rir_sr, target_sr)
            noise = preproecss(noise, noise_sr, target_sr)
        
        data_lenght, noise_lenght = data.shape[-1], noise.shape[-1]

        if random.uniform(0.0, 1.0) < rir_ratio:
            data = add_rir(data, rir)
        
        if random.uniform(0.0, 1.0) < noise_ratio:
            if data_lenght>noise_lenght:
                repeat = int(data_lenght/noise_lenght)+1
                noise = noise.repeat(1, repeat)
            noise = noise[..., :data_lenght]
            data = add_noise(data, noise, snr)
        

        if max_seconds is not None:
            sr = target_sr
            if data_lenght > max_seconds*sr:
                data = data[:, :max_seconds*sr]
            padding = max_seconds*sr - data.shape[1]
            data = torch.nn.functional.pad(data, (0,padding,0,0), mode='constant', value=0.0)
        audios.append(data)
        lenghts.append(data_lenght)

    audios = torch.cat(audios, dim=0)
    lenghts = torch.tensor(lenghts)

    return audios, lenghts

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

        self.n_fft = int(sr * win_length_second)
        self.hop_length = int(sr * stride_second)

        self.s1_max = (max_second*sr - self.n_fft)//self.hop_length+1 
        """torchaudio.transforms.MelSpectrogram returns [B, NMELS, s1_max]"""
        self.s2_max = ((self.s1_max - self.patch_size)//self.patch_stride) + 1
        """torch.nn.Conv1d returns [B, embeddim, s2_max]"""

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=0,
            f_max=self.sr // 2,
            power=2.0,
            normalized=True,
            center=False
        ).to(device)

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB("power", top_db=80).to(device)

        self.embedding = nn.Conv1d(in_channels=self.n_mels, 
                                   out_channels=self.embeddim, 
                                   kernel_size=self.patch_size, 
                                   stride=self.patch_stride,
                                   bias=False).to(device) # [B, NMELS, s1] to [B, embeddim, ((s1 - patch_size)//patch_stride) + 1]

    def forward(self, x:Tensor, lenghts:Tensor=None):
        # x.shape = (B, self.max_seconds*self.sr)
        B, _ = x.shape
    
        x = self.mel_spectrogram.forward(x)       # [B, n_mels, s1_max]
        x = self.amplitude_to_db.forward(x)       # [B, n_mels, s1_max]
        x = self.embedding(x)                     # [B, embeddim, s2_max]
        x = x.permute(0, 2, 1)                    # [B, s2_max, embeddim]
        
        masks = torch.zeros(B, self.s2_max, device=self.device).bool()
        
        if lenghts is not None:
            with torch.no_grad():
                s1 = (lenghts - self.n_fft)//self.hop_length+1      # [B]
                s1 = torch.clamp(s1, min=0, max=self.s1_max)        # [B]
                s2 = (s1 - self.patch_size)//self.patch_stride+1    # [B]
                s2 = torch.clamp(s2, min=0, max=self.s2_max)        # [B]

                indices = torch.arange(self.s2_max, device=self.device)  # [s2_max]

                masks = indices >= s2.unsqueeze(1)                  # [B, s2_max]

        return x, masks
    