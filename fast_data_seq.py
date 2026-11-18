"""
fast_data_seq.py: GPU-resident batches for fixed-shape window tensors.

Same idea as fast_data.py (graph version), simplified for sequence windows:
the preprocessed splits are already dense tensors X [N, T, C] and y [N], so
no packing, padding or cache file is needed. Each split is moved to the GPU
once when it fits (otherwise pinned CPU memory), and a batch is a single
index operation. No DataLoader, no worker processes, no per-batch collate.
"""

import torch


class TensorSplit:
    def __init__(self, X, y, device, placement="auto", gpu_fraction=0.6, name="", log=print):
        X, y = X.contiguous(), y.long().contiguous()
        nbytes = X.numel() * X.element_size() + y.numel() * y.element_size()
        on_gpu = False
        if device.type == "cuda" and placement in ("auto", "gpu"):
            free, _ = torch.cuda.mem_get_info(device)
            on_gpu = placement == "gpu" or nbytes < gpu_fraction * free
        if on_gpu:
            X, y = X.to(device), y.to(device)
        elif device.type == "cuda":
            X, y = X.pin_memory(), y.pin_memory()
        self.X, self.y, self.on_gpu, self.device = X, y, on_gpu, device
        log(f"  [FastData] {name}: {len(y)} windows {tuple(X.shape[1:])}, "
            f"{nbytes / 2**20:.0f} MB on {'GPU' if on_gpu else 'pinned CPU'}")

    def __len__(self):
        return self.y.numel()


class GPUBatches:
    """Drop-in replacement for DataLoader(TensorDataset(X, y), ...). Yields (X, y) on device."""

    def __init__(self, split, batch_size, shuffle=False, drop_last=False):
        self.s, self.batch_size = split, batch_size
        self.shuffle, self.drop_last = shuffle, drop_last

    def __len__(self):
        n = len(self.s)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        s, n = self.s, len(self.s)
        idx_device = s.device if s.on_gpu else torch.device("cpu")
        rows = (torch.randperm(n, device=idx_device) if self.shuffle
                else torch.arange(n, device=idx_device))
        for k in range(len(self)):
            r = rows[k * self.batch_size:(k + 1) * self.batch_size]
            X, y = s.X[r], s.y[r]
            if not s.on_gpu:
                X = X.pin_memory().to(s.device, non_blocking=True)
                y = y.pin_memory().to(s.device, non_blocking=True)
            yield X, y
