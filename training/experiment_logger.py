import csv
import time

class ExperimentLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.start_time = time.time()
        with open(self.log_path, "w", newline = "") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_loss", "val_loss", "elapsed_time"])

    def log(self, step, train_loss, val_loss):
        elapsed_time = time.time() - self.start_time
        with open(self.log_path, "a", newline = "") as f:
            writer = csv.writer(f)
            writer.writerow([step, train_loss, val_loss, elapsed_time])




