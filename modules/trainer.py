import os, math
import json
import torch, cv2
import torchaudio
import random
import numpy as np
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self,
                 model,
                 images_path:os.PathLike,
                 audios_path:os.PathLike,
                 labels_path:os.PathLike,
                 device:torch.device,
                 valid_ratio:float=0.2,
                 valid=True,
                 log_dir:os.PathLike="runs/sound_control"):
        
        self.model = model
        self.imgsz = model.imgsz
        self.images_path = images_path
        self.audios_path = audios_path
        self.labels_path = labels_path
        self.device = device
        self.valid = valid

        self.writer = SummaryWriter(log_dir)

        self.labels = os.listdir(labels_path)
        self.valid_labels = self.labels[:int(len(self.labels)*valid_ratio)]
        self.train_labels = self.labels[int(len(self.labels)*valid_ratio):]

    def train(self, epochs:int, batch_size:int=16, lr:float=0.01):

        train_ds = AudioVisualDataset(
            self.train_labels, 
            self.labels_path, self.images_path, self.audios_path, self.model
        )
        
        valid_ds = AudioVisualDataset(
            self.valid_labels, 
            self.labels_path, self.images_path, self.audios_path, self.model
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, 
            num_workers=8, pin_memory=True, collate_fn=collate_fn, persistent_workers=True
        )
        
        valid_loader = DataLoader(
            valid_ds, batch_size=batch_size, shuffle=False, 
            num_workers=8, pin_memory=True, collate_fn=collate_fn
        )

        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, len(train_loader), T_mult=2, eta_min=1e-5)
        scaler = GradScaler()
        self.model.train(True)

        self.writer.add_hparams(
            {'lr': lr, 'batch_size': batch_size, 'epochs': epochs}, 
            {'hparam/placeholder': 0}
        )

        global_step = 0

        for epoch in range(epochs):
            num_batches = len(train_loader)

            for batch_idx, batch_data in enumerate(train_loader):
                if batch_data is None: continue
                optim.zero_grad()
                global_step += 1

                frames, audios, bboxes, target, bbox_masks = [x.to(self.device, non_blocking=True) for x in batch_data]
                with autocast(device_type='cuda', dtype=torch.float32):
                    pred = self.model.forward(audios, frames, bboxes, bbox_masks) # [B, N]
                    loss = criterion.forward(pred, target.float()) # [B, N]
                    valid_mask = ~bbox_masks
                    if valid_mask.sum() > 0:
                        final_loss = loss[valid_mask].mean()
                    else:
                        final_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                scaler.scale(final_loss).backward()
                scaler.step(optim)
                scheduler.step()
                scaler.update()

                self.writer.add_scalar('Loss/Train_Batch', final_loss.item(), global_step)
                self.writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)

                total = 33
                progress = math.ceil(batch_idx*total/num_batches)
                remain = total - progress
                label = (
                        f"Epoch: {epoch+1:3d}/{epochs} "
                        f"Batch: {batch_idx+1:4d}/{num_batches+1}  [{100*(batch_idx/num_batches):6.2f}%]{'█'*progress}{'░'*remain} "
                        f"LR: {scheduler.get_last_lr()[0]:.07f} "
                        f"Loss: {final_loss.item():.06f}"
                        )
                print(label, end="\r")
            print("\n")
            self.save(epoch)
            if self.valid:
                # Validation
                with torch.no_grad():
                    gloss = 0
                    iter = 0
                    num_batches = len(valid_loader)
                    for batch_idx, batch_data in enumerate(valid_loader):
                        if batch_data is None: continue

                        frames, audios, bboxes, target, bbox_masks = [x.to(self.device, non_blocking=True) for x in batch_data]
                        pred = self.model.forward(audios, frames, bboxes, bbox_masks) # [B, N]

                        loss = criterion.forward(pred, target.float()) # [B, N]
                        valid_mask = ~bbox_masks
                        if valid_mask.sum() > 0:
                            final_loss = loss[valid_mask].mean()
                        else:
                            final_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                        gloss += final_loss.item()
                        iter += 1

                        loss = gloss / iter
                        
                        total = 33
                        progress = math.ceil(batch_idx*total/num_batches)
                        remain = total - progress
                        label = (
                                f"Epoch {epoch+1:3d}/{epochs} Valitadion: "
                                f"[{100*(batch_idx/num_batches):6.2f}%]{'█'*progress}{'░'*remain} "
                                f"Loss: {loss:.06f}"
                                )
                        print(label, end="\r")
                    print("\n")
                avg_valid_loss = gloss / iter
                self.writer.add_scalar('Loss/Validation_Epoch', avg_valid_loss, epoch)

        self.model.train(False)
        return self.model
    
    def save(self, epoch:int):
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'imgsz': self.model.imgsz,
            'embeddim': self.model.embeddim,
            'num_layers': self.model.num_layers,
            'num_heads': self.model.num_heads,
            'dropout': self.model.dropout,
            'mlp_ratio': self.model.mlp_ratio,
            'n_mels': self.model.n_mels,
            'max_seconds': self.model.max_seconds,
            'sr': self.model.sr,
            'bbox_size': self.model.bbox_size
        }, f"checkpoints/model_epoch_{epoch}.pt")

    
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    frames, audios, bboxes, targets, masks = zip(*batch)
    
    return (
        torch.cat(frames, dim=0),
        torch.cat(audios, dim=0),
        torch.cat(bboxes, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(masks, dim=0),
    )

class AudioVisualDataset(Dataset):
    def __init__(self, file_list, labels_dir, images_dir, audios_dir, model_config):
        self.file_list = file_list
        self.labels_dir = labels_dir
        self.images_dir = images_dir
        self.audios_dir = audios_dir
        
        self.imgsz = model_config.imgsz
        self.sr = model_config.sr
        self.max_seconds = model_config.max_seconds
        self.bbox_size = model_config.bbox_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        label_file = self.file_list[idx]
        label_path = os.path.join(self.labels_dir, label_file)
        
        try:
            with open(label_path, "r") as f:
                data = json.load(f)

            frame_name = data["frame_name"]
            audio_file = data["audio_file"]
            bboxes = data["bboxes"]
            if len(bboxes) == 0:
                return None
            selected_box_idx = data["selected_box_idx"]

            frame_path = os.path.join(self.images_dir, frame_name)
            audio_path = os.path.join(self.audios_dir, audio_file)

            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.imgsz, self.imgsz))
            frame = torch.from_numpy(frame).float()
            frame = frame.permute(2, 0, 1).unsqueeze(0)

            wave_lenght = self.sr*self.max_seconds
            audio, sr = torchaudio.load(audio_path)

            if sr != self.sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)
                audio = resampler(audio)
                sr = self.sr

            padding = wave_lenght - audio.shape[1]
            if padding > 0:
                audio = torch.nn.functional.pad(audio, (0, padding))
            else:
                audio = audio[:, :wave_lenght]


            selecting_bbox = torch.zeros((self.bbox_size)).bool()
            for idx in selected_box_idx:
                selecting_bbox[idx] = True

            bbox_mask = torch.ones((self.bbox_size)).bool()
            for i in range(len(bboxes)):
                bbox_mask[i] = False

            bboxes = torch.Tensor(bboxes).float()[:, :4]
            num_bbox = self.bbox_size
            padding = num_bbox - bboxes.shape[0]
            if padding > 0:
                bboxes = torch.nn.functional.pad(bboxes, (0, 0, 0, padding))
            else:
                bboxes = bboxes[:num_bbox]

            return frame, audio, bboxes.unsqueeze(0), selecting_bbox.unsqueeze(0), bbox_mask.unsqueeze(0)
        except Exception as e:
            return None