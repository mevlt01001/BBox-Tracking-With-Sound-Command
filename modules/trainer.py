import os
import json
import torch, cv2
import torchaudio
import random
import numpy as np
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

class Trainer:
    def __init__(self,
                 model,
                 images_path:os.PathLike,
                 audios_path:os.PathLike,
                 labels_path:os.PathLike,
                 device:torch.device,):
        
        self.model = model
        self.imgsz = model.imgsz
        self.images_path = images_path
        self.audios_path = audios_path
        self.labels_path = labels_path
        self.device = device

        self.labels = os.listdir(labels_path)
        self.valid_labels = self.labels[:int(len(self.labels)//10)]
        self.train_labels = self.labels[int(len(self.labels)//10):]

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
            num_workers=4, pin_memory=True, collate_fn=collate_fn, persistent_workers=True
        )
        
        valid_loader = DataLoader(
            valid_ds, batch_size=batch_size, shuffle=False, 
            num_workers=2, pin_memory=True, collate_fn=collate_fn
        )

        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        scaler = GradScaler()
        self.model.train(True)

        for epoch in range(epochs):
            num_batches = len(train_loader)

            for batch_idx, batch_data in enumerate(train_loader):
                if batch_data is None: continue
                optim.zero_grad()

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
                scaler.update()


                label = f"Epoch: {epoch+1:03}/{epochs}, [%{batch_idx/num_batches:.2f}]Batch: \
                    {batch_idx+1:03}/{num_batches+1}, Loss: {final_loss.item():.06f}"
                print(label, end="\r")
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
                    
                    label = (
                        f"Epoch: {epoch+1:03}/{epochs}"
                        f"[%{batch_idx/num_batches:.2f}]Batch: {batch_idx+1:03}/{num_batches+1}"
                        f"Loss: {gloss/iter:.06f}"
                        )
                    print(label, end="\r")
            torch.save(self.model.state_dict(), "model.pt")

        self.model.train(False)
        return self.model
    
    def create_batch(self, start, end, train=True):
        frames = []
        audios = []
        bboxes = []
        target = []
        bbox_masks = []

        for i in range(start, end):
            label_path = os.path.join(self.labels_path, self.train_labels[i]) if train else os.path.join(self.labels_path, self.valid_labels[i])
            frame, audio, bbox, selected_box, bbox_mask = self.load_data(label_path)
            if frame is None:
                continue
            frames.append((frame/255.0).to(self.device))
            audios.append(audio.to(self.device))
            bboxes.append((bbox/self.imgsz).to(self.device))
            target.append(selected_box.to(self.device))
            bbox_masks.append(bbox_mask.to(self.device))

        frames = torch.cat(frames, dim=0)
        audios = torch.cat(audios, dim=0)
        bboxes = torch.cat(bboxes, dim=0)
        target = torch.cat(target, dim=0)
        bbox_masks = torch.cat(bbox_masks, dim=0)

        return frames, audios, bboxes, target, bbox_masks
        
    
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
        # Senin 'load_data' fonksiyonunun içi buraya gelecek
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
            print(f"\nError while running for {audio_path}: {e}",end="\n")
            return None