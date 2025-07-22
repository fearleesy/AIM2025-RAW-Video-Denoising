import torch
import numpy as np
from torch.utils.data import Dataset

def _even_crop_coords(h, w, size):
    i = torch.randint(0, h - size + 1, (1,)).item()
    j = torch.randint(0, w - size + 1, (1,)).item()
    i -= i % 2
    j -= j % 2
    return i, j

def _bayer_to_4ch(raw):
    r  = raw[0::2, 0::2]
    g1 = raw[0::2, 1::2]
    g2 = raw[1::2, 0::2]
    b  = raw[1::2, 1::2]
    return torch.stack([r, g1, g2, b], 0)

def _ch4_to_bayer(img):
    img = img.squeeze(0)
    _, H, W = img.shape
    mono = torch.zeros((1, H * 2, W * 2), dtype=img.dtype, device=img.device)

    mono[0, 0::2, 0::2] = img[0]
    mono[0, 0::2, 1::2] = img[1]
    mono[0, 1::2, 0::2] = img[2]
    mono[0, 1::2, 1::2] = img[3]
    return mono

def _load_patch(arr, i, j, sz=1024):
    patch = arr[i:i+sz, j:j+sz]
    return _bayer_to_4ch(torch.from_numpy(patch))


class RawVideoTrainDataset(Dataset):
    def __init__(self, root_dir, crop_size=256):
        self.crop_size = crop_size
        self.paths = sorted(root_dir.glob("**/*.npz"))
        self.keys = (
            [f"frame_{i:02d}" for i in range(10)] +
            [f"extra_noisy_09_{i:02d}" for i in range(20)] +
            ["gt"]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arrays = np.load(self.paths[idx])
        H, W = arrays["frame_00"].shape

        i, j = _even_crop_coords(H, W, self.crop_size)
        do_h = torch.rand(()) > 0.5
        do_v = torch.rand(()) > 0.5

        out = {}
        for key in self.keys:
            tensor = _load_patch(arrays[key], i, j, self.crop_size)
            if do_h:
                tensor = torch.flip(tensor, dims=[2])
            if do_v:
                tensor = torch.flip(tensor, dims=[1])
            out[key] = tensor

        return out
