"""Exact running training-view means, one row per camera pass (not validation)."""
import csv
from pathlib import Path


class EpochMetrics:
    fields = ['pass_index', 'start_iteration', 'end_iteration', 'train_camera_count',
              'sample_count', 'complete_epoch', 'epoch_progress', 'mean_loss', 'mean_psnr_db']

    def __init__(self, path, camera_count, start_iteration=0):
        if camera_count <= 0:
            raise ValueError('Training camera count must be positive')
        self.path = Path(path)
        self.camera_count = camera_count
        self.end = start_iteration
        self.pass_index = 0
        self.count = 0
        self.loss_sum = self.psnr_sum = 0.0
        with self.path.open('x', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def add(self, iteration, loss, psnr):
        if iteration != self.end + 1:
            raise ValueError('Epoch metrics require every consecutive iteration')
        if self.count == 0:
            self.start = iteration
        self.end = iteration
        self.count += 1
        self.loss_sum += float(loss)
        self.psnr_sum += float(psnr)
        if self.count == self.camera_count:
            return self.flush()

    def flush(self):
        if not self.count:
            return None
        self.pass_index += 1
        row = dict(pass_index=self.pass_index, start_iteration=self.start,
                   end_iteration=self.end, train_camera_count=self.camera_count,
                   sample_count=self.count, complete_epoch=self.count == self.camera_count,
                   epoch_progress=self.end/self.camera_count,
                   mean_loss=self.loss_sum/self.count, mean_psnr_db=self.psnr_sum/self.count)
        with self.path.open('a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(row)
        self.count = 0
        self.loss_sum = self.psnr_sum = 0.0
        return row
