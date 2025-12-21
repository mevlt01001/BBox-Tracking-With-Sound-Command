import os
import math
import json
import time
import torch
import torch.nn as nn
from torch import Tensor
# deleted due to circular import
from .preprocess import load_audios
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self,
                 model,
                 labels_dir:os.PathLike, 
                 audios_dir:os.PathLike,
                 batch_size:int=16, 
                 lr:float=0.01, 
                 min_lr:float=1e-5, 
                 epochs:int=10,
                 test_ratio:float=0.2,
                 valid:bool=True,
                 log_dir:os.PathLike="runs/sound_control",
                 device:torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.labels_dir = labels_dir
        self.audios_dir = audios_dir
        self.log_dir = log_dir
        self.labels = os.listdir(labels_dir)
        self.valid = valid
        self.valid_labels = self.labels[:int(len(self.labels)*test_ratio)]
        self.train_labels = self.labels[int(len(self.labels)*test_ratio):]

        self.train_ds = AudioDataset(labels_dir=self.labels_dir, file_list=self.train_labels, audios_dir=self.audios_dir)
        self.valid_ds = AudioDataset(labels_dir=self.labels_dir, file_list=self.valid_labels, audios_dir=self.audios_dir)
        self.collate_fn = Collator(target_sr=self.model.sr, max_seconds=self.model.max_second)

        self.train_loader = DataLoader(
            self.train_ds, batch_size=batch_size, shuffle=True, 
            num_workers=4, pin_memory=True, collate_fn=self.collate_fn, persistent_workers=True
        )
        
        self.valid_loader = DataLoader(
            self.valid_ds, batch_size=batch_size, shuffle=False, 
            num_workers=4, pin_memory=True, collate_fn=self.collate_fn
        )

        self.device = device
        
    def train(self):
        self.model.train(mode=True)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, len(self.train_loader), T_mult=2, eta_min=self.min_lr)
        self.scaler = GradScaler()
        self.writer = SummaryWriter(self.log_dir)

        self.train_step = 0
        self.valid_step = 0

        for i_epoch in range(self.epochs):
            
            num_batches = len(self.train_loader)
            start = time.time()
            label = ""
            epoch_clr_loss = 0
            epoch_geo_loss = 0
            epoch_loss = 0
            epoch_step = 0

            with autocast(device_type='cuda', dtype=torch.float32):
                for i_batch, batch in enumerate(self.train_loader):
                    if batch is None: continue
                    
                    audios, clrs, geos = [item.to(self.device) for item in batch]
                    self.model.zero_grad()
                    loss, lclr, lgeo = self.forward(audios, clrs, geos)
                    

                    epoch_step += 1
                    self.writer.add_scalar('Loss/Train_Batch', loss, self.train_step)
                    self.writer.add_scalar('CLR_Loss/Train_Batch', lclr, self.train_step)
                    self.writer.add_scalar('GEO_Loss/Train_Batch', lgeo, self.train_step)
                    self.writer.add_scalar('Learning_Rate', self.scheduler.get_last_lr()[0], self.train_step)

                    total = 10
                    progress = math.ceil(i_batch*total/num_batches)
                    remain = total - progress
                    label = (
                        f"Epoch: {i_epoch+1:3d}/{self.epochs} "
                        f"Batch: {i_batch+1:4d}/{num_batches+1}  [{100*(i_batch/num_batches):6.2f}%]{'█'*progress}{'░'*remain} "
                        f"LR: {self.scheduler.get_last_lr()[0]:.07f} "
                        f"CLR_Loss: {lclr:09.06f} GEO_Loss: {lgeo:09.06f} "
                        f"Loss: {loss:09.06f} "
                        f"Elapsed: {time.time()-start:8.02f}s "
                        )
                    print(label, end="\r")

                    epoch_loss += loss
                    epoch_clr_loss += lclr
                    epoch_geo_loss += lgeo

                self.writer.add_scalar('Loss/Epoch', epoch_loss/epoch_step, i_epoch)
                self.writer.add_scalar('CLR_Loss/Epoch', epoch_clr_loss/epoch_step, i_epoch)
                self.writer.add_scalar('GEO_Loss/Epoch', epoch_geo_loss/epoch_step, i_epoch)

            datetime = f"AVG_LOSS: {epoch_loss/epoch_step:09.06f} {time.strftime('%H:%M:%S', time.localtime(time.time()))}"
            print(label, datetime, end="\n")
            self.save(i_epoch)
            if self.valid:
                num_batches = len(self.valid_loader)
                valid_clr_loss = 0
                valid_geo_loss = 0
                valid_loss = 0
                valid_step = 0
                
                self.model.train(mode=False)
                with torch.no_grad():
                    with autocast(device_type='cuda', dtype=torch.float32):
                        for i_batch, batch in enumerate(self.valid_loader):
                            if batch is None: continue
                            audios, clrs, geos = [item.to(self.device) for item in batch]
                            loss, lclr, lgeo = self.forward(audios, clrs, geos, valid=True)
                            
                            self.writer.add_scalar('Loss/Valid_Batch', loss, self.valid_step)
                            self.writer.add_scalar('CLR_Loss/Valid_Batch', lclr, self.valid_step)
                            self.writer.add_scalar('GEO_Loss/Valid_Batch', lgeo, self.valid_step)

                            valid_step += 1
                            total = 10
                            progress = math.ceil(i_batch*total/num_batches)
                            remain = total - progress
                            label = (
                                f"VALİDATION: {i_batch+1:4d}/{num_batches+1}  [{100*(i_batch/num_batches):6.2f}%]{'█'*progress}{'░'*remain} "
                                f"AVG_CLR_Loss: {valid_clr_loss/valid_step:09.06f} AVG_GEO_Loss: {valid_geo_loss/valid_step:09.06f} "
                                f"AVG_Loss: {valid_loss/valid_step:09.06f} "
                                f"Elapsed: {time.time()-start:8.02f}s "
                                )
                            print(label, end="\r")

                            valid_loss += loss
                            valid_clr_loss += lclr
                            valid_geo_loss += lgeo

                        self.writer.add_scalar('Loss/Valid', valid_loss/valid_step, i_epoch)
                        self.writer.add_scalar('CLR_Loss/Valid', valid_clr_loss/valid_step, i_epoch)
                        self.writer.add_scalar('GEO_Loss/Valid', valid_geo_loss/valid_step, i_epoch)
            datetime = f"{time.strftime('%H:%M:%S', time.localtime(time.time()))}"
            print(label, datetime, end="\n")
        
        self.model.train(mode=False)
        return self.model
    
    def forward(self, audios, clrs, geos, valid=False):
        if not valid: self.model.zero_grad()

        pred_clr, pred_geo = self.model.forward(audios)
        clr_loss = self.criterion.forward(pred_clr, clrs)
        geo_loss = self.criterion.forward(pred_geo, geos)
        loss = clr_loss + geo_loss

        if valid:
            self.valid_step += 1

        if not valid: # if training
            self.train_step += 1
            # loss.backward()
            # self.optim.step()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            self.scheduler.step()
        return loss.item(), clr_loss.item(), geo_loss.item()
    
    def save(self, epoch:int):
        os.makedirs("checkpoints", exist_ok=True)
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

        

class Collator(object):
    def __init__(self, 
                 target_sr=16000, 
                 max_seconds=10,
                 ):
        
        self.target_sr = target_sr
        self.max_seconds = max_seconds

    def __call__(self, batch):

        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None

        audio_paths, clrs, geos = zip(*batch)
        try:
            audios = load_audios(audio_paths, target_sr=self.target_sr, max_seconds=self.max_seconds)
        except:
            return None
        clrs = torch.LongTensor(clrs)
        geos = torch.LongTensor(geos)

        return audios, clrs, geos

class AudioDataset(Dataset):
    def __init__(self,
                 file_list,
                 labels_dir, 
                 audios_dir,
                 ):
        
        self.file_list = file_list
        self.labels_dir = labels_dir
        self.audios_dir = audios_dir

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        label_file = self.file_list[index]
        label_path = os.path.join(self.labels_dir, label_file)
        if not os.path.exists(label_path):
            return None
        try:
            with open(label_path, "r") as f:
                data = json.load(f)

            audio_file = os.path.basename(data["Audio_data"])
            audio_path = os.path.join(self.audios_dir, audio_file)
            if not os.path.exists(audio_path):
                return None
            clr, geo = data["CLR"], data["GOE"]

            return audio_path, clr, geo

        except Exception as e:
            print(f"ERROR AT FILE {label_path}", e)
            return None

