import os
import csv
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.train_file = open(os.path.join(log_dir, f"train_log_{timestamp}.csv"), "w", newline="")
        self.val_file = open(os.path.join(log_dir, f"val_log_{timestamp}.csv"), "w", newline="")

        self.train_writer = csv.writer(self.train_file)
        self.val_writer = csv.writer(self.val_file)

        self.train_writer.writerow(["Epoch", "Batch", "Loss", "Recon", "VQ", "Perceptual"])
        self.val_writer.writerow(["Epoch", "Batch", "Loss", "Recon", "VQ", "Perceptual"])

    def log_train(self, epoch, batch, loss, recon, vq, perceptual):
        self.train_writer.writerow([epoch, batch, loss, recon, vq, perceptual])
        self.train_file.flush()

    def log_val(self, epoch, batch, loss, recon, vq, perceptual):
        self.val_writer.writerow([epoch, batch, loss, recon, vq, perceptual])
        self.val_file.flush()

    def close(self):
        self.train_file.close()
        self.val_file.close()
