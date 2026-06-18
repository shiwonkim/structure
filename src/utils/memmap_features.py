"""ConcatFeatureStore: virtual concatenation of an in-memory tensor
with a disk-backed numpy memmap array.

Supports the same indexing patterns the alignment trainer uses:
  - store[int_index]
  - store[start:end]  (slice)
  - store[tensor_of_indices]  (fancy indexing for shuffle)
  - len(store), store.shape, store.dim()

Only the accessed rows are loaded into RAM and converted to torch
tensors. The memmap portion stays on disk; the OS pages in data
on demand (~4096 rows per batch = a few MB).
"""

import numpy as np
import torch
import torch.nn.functional as F


class ConcatFeatureStore:
    """Lazily concatenates an in-memory torch.Tensor (part A) with a
    numpy memmap array (part B), optionally subsampled via ``sub_idx``.

    Parameters
    ----------
    tensor_a : torch.Tensor
        In-memory features (e.g., COCO). Shape ``(N_a, ...)``.
    mmap_b : np.memmap
        Disk-backed features (e.g., LAION). Shape ``(N_b_full, ...)``.
    sub_idx : np.ndarray or None
        If provided, only these rows of ``mmap_b`` are used.
        Must be sorted for efficient memmap access.
    T_max : int or None
        If provided, pad the sequence dimension (dim=1) of memmap
        rows to this length. Used when COCO and LAION have different
        text sequence lengths.
    """

    def __init__(self, tensor_a, mmap_b, sub_idx=None, T_max=None):
        self._a = tensor_a
        self._b = mmap_b
        self._sub_idx = sub_idx
        self._T_max = T_max
        self._n_a = tensor_a.shape[0]
        self._n_b = len(sub_idx) if sub_idx is not None else mmap_b.shape[0]
        self._ndim = len(tensor_a.shape)

    def __len__(self):
        return self._n_a + self._n_b

    @property
    def shape(self):
        total = self._n_a + self._n_b
        rest = self._a.shape[1:]
        return torch.Size([total, *rest])

    def dim(self):
        return self._ndim

    def _read_b(self, b_indices):
        """Read rows from memmap, map through sub_idx, return torch tensor."""
        if self._sub_idx is not None:
            actual_idx = self._sub_idx[b_indices]
        else:
            actual_idx = b_indices

        data = np.array(self._b[actual_idx])
        t = torch.from_numpy(data)

        # Pad sequence dim to match tensor_a's length
        if self._T_max is not None and t.dim() >= 2:
            T_cur = t.shape[1]
            if T_cur < self._T_max:
                if t.dim() == 3:  # (B, T, D)
                    t = F.pad(t, (0, 0, 0, self._T_max - T_cur))
                elif t.dim() == 2:  # (B, T) for masks
                    t = F.pad(t, (0, self._T_max - T_cur))

        return t

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            if idx < self._n_a:
                return self._a[idx]
            else:
                return self._read_b(idx - self._n_a)

        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))

            # Fast path: entire slice falls in one region
            if start >= self._n_a:
                # Entirely in memmap B — use contiguous slice read
                b_start = start - self._n_a
                b_stop = stop - self._n_a
                if self._sub_idx is not None:
                    actual_start = self._sub_idx[b_start]
                    actual_stop = self._sub_idx[b_stop - 1] + 1
                    # Only use slice if sub_idx is contiguous in this range
                    if actual_stop - actual_start == b_stop - b_start:
                        data = np.array(self._b[actual_start:actual_stop])
                    else:
                        data = np.array(self._b[self._sub_idx[b_start:b_stop]])
                else:
                    data = np.array(self._b[b_start:b_stop])
                t = torch.from_numpy(data)
                # Pad if needed
                if self._T_max is not None and t.dim() >= 2:
                    T_cur = t.shape[1]
                    if T_cur < self._T_max:
                        if t.dim() == 3:
                            t = F.pad(t, (0, 0, 0, self._T_max - T_cur))
                        elif t.dim() == 2:
                            t = F.pad(t, (0, self._T_max - T_cur))
                return t
            elif stop <= self._n_a:
                # Entirely in tensor A
                return self._a[start:stop]
            else:
                # Spans both A and B
                parts = []
                parts.append(self._a[start:self._n_a])
                b_count = stop - self._n_a
                if self._sub_idx is not None:
                    data = np.array(self._b[self._sub_idx[:b_count]])
                else:
                    data = np.array(self._b[:b_count])
                t = torch.from_numpy(data)
                if self._T_max is not None and t.dim() >= 2:
                    T_cur = t.shape[1]
                    if T_cur < self._T_max:
                        if t.dim() == 3:
                            t = F.pad(t, (0, 0, 0, self._T_max - T_cur))
                        elif t.dim() == 2:
                            t = F.pad(t, (0, self._T_max - T_cur))
                parts.append(t)
                return torch.cat([p.to(parts[0].dtype) for p in parts], dim=0)

        elif isinstance(idx, torch.Tensor):
            idx_np = idx.numpy() if idx.device.type == "cpu" else idx.cpu().numpy()
            a_mask = idx_np < self._n_a
            b_mask = ~a_mask

            # Allocate output with same shape as indexing into tensor_a
            sample_shape = self._a.shape[1:]
            out = torch.empty(len(idx_np), *sample_shape, dtype=self._a.dtype)

            if a_mask.any():
                a_indices = idx_np[a_mask]
                out[a_mask] = self._a[a_indices]

            if b_mask.any():
                b_indices = idx_np[b_mask] - self._n_a
                b_data = self._read_b(b_indices)
                out[b_mask] = b_data.to(out.dtype)

            return out

        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    def to(self, *args, **kwargs):
        """No-op for compatibility — data is loaded per-batch."""
        return self

    def float(self):
        """No-op — conversion happens per-batch in the training loop."""
        return self
