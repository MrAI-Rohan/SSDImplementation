class SSD(nn.Module):
    def __init__(self, device=None, scheduler=None, mixed_precision=False):
        super().__init__()
        self.device = device if device is not None else torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')
        self.mixed_precision = mixed_precision
        self.scheduler = scheduler

        n_classes = 21
        in_channels_per_map = [512, 1024, 512, 256, 256, 256]
        n_dbox_per_map = [4, 6, 6, 6, 4, 4]

        self.feature_extractor  = SSDFeatureExtractor().to(self.device)
        self.detection_head = SSDDetectionHead(in_channels_per_map, n_dbox_per_map, n_classes).to(self.device)
        self.loss_function = SSDLoss()

        extractor_parameters = list(self.feature_extractor.parameters())
        det_head_parameters = list(self.detection_head.parameters())
        parameters = extractor_parameters + det_head_parameters

        self.optimizer = optim.SGD(parameters, lr=1e-3, momentum=0.9, weight_decay=5e-4)
        self.scaler = None
        
    def forward(self, x):
        maps = self.feature_extractor(x)
        preds = self.detection_head(maps.values())
        return preds

    def train_one_epoch(self, train_dl, epoch):
        self.train()

        total_loss = 0
        pbar = tqdm(train_dl, desc=f"Training epoch {epoch}", leave=True)
        for i, (imgs, gt) in enumerate(pbar):
            imgs = imgs.to(self.device)
            gt_bboxes, gt_labels = gt[0].to(self.device), gt[1].to(self.device)
            self.optimizer.zero_grad()

            if self.mixed_precision and torch.cuda.is_available() and self.scaler is not None:
                with autocast('cuda'):
                    preds = self(imgs)
                    loss = self.loss_function(preds, (gt_bboxes, gt_labels))

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self(imgs)
                loss = self.loss_function(preds, (gt_bboxes, gt_labels))

                loss.backward()
                self.optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss
            avg_loss = total_loss / (i + 1)

            pbar.set_postfix({
            "batch": f"{i+1}/{len(train_dl)}",
            "loss": f"{current_loss:.4f}",
            "avg_loss": f"{avg_loss:.4f}"
        })

        return total_loss / len(train_dl)

    def validate(self, valid_dl):
        self.eval()

        total_loss = 0
        with torch.no_grad():
            for imgs, gt in valid_dl:
                imgs = imgs.to(self.device)
                gt_bboxes, gt_labels = gt[0].to(self.device), gt[1].to(self.device)

                preds = self(imgs)
                loss = self.loss_function(preds, (gt_bboxes, gt_labels))

                total_loss += loss.item()

        return total_loss / len(valid_dl)

    def fit(self, n_epochs, train_dl, valid_dl=None, verbose=True, scheduler=None,
            mixed_precision=None, save_after_training=False, save_every_epoch=False,
            save_path="checkpoints"):
        
        if mixed_precision is not None:
            self.mixed_precision = mixed_precision

        if self.mixed_precision and torch.cuda.is_available() and self.scaler is None:
            self.scaler = GradScaler('cuda')

        if scheduler is not None:
            self.scheduler = scheduler
        
        os.makedirs(save_path, exist_ok=True)

        print(f"Starting training for {n_epochs} epochs...")

        for epoch in range(n_epochs):
            train_loss = self.train_one_epoch(train_dl, epoch+1)

            valid_loss = None
            if valid_dl is not None:
                valid_loss = self.validate(valid_dl)

            if verbose:
                if valid_loss is not None:
                    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}")

            if self.scheduler:
                self.scheduler.step()
            
            if save_every_epoch:
                epoch_path = os.path.join(save_path, f"ssd_checkpoint_epoch_{epoch+1}.pth")
                self.save_checkpoint(epoch, path=epoch_path)

        if save_after_training:
            final_path = os.path.join(save_path, "ssd_final_checkpoint.pth")
            self.save_checkpoint(n_epochs, path=final_path)
            print(f"Final checkpoint saved at {final_path}")
        
    def save_checkpoint(self, epoch, path="ssd_checkpoint.pth"):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at epoch {epoch} to {path}")

    def load_checkpoint(self, path="ssd_checkpoint.pth"):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            print(f"Checkpoint loaded from {path}, Resuming from Epoch {checkpoint['epoch']}")
            return checkpoint['epoch']
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return 0

    def to(self, device):
        self.device = device
        return super().to(device)
