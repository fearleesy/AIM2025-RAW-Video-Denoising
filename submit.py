import torch
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from dataset import _load_patch, _ch4_to_bayer
from model import UNetDenoise


def build_model(args):
    model = UNetDenoise().to(args.device)
    ckpt = torch.load(args.ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    return model

def make_prediction(sequence_array, c_h, c_w, model):
    src = torch.cat([
        _load_patch(sequence_array[f"frame_{k:02d}"], c_h, c_w)
        for k in range(10)
    ], dim=0)
    
    src = src.to(args.device).unsqueeze(0)
    result = model(src)
    return _ch4_to_bayer(result).cpu().detach().numpy()

def center_crop_prediction_coords(prediction: np.ndarray, crop_size: int = 1024) -> np.ndarray:
    assert len(prediction.shape) == 2, "Prediction must be a 2D Bayer image"
    assert min(prediction.shape) >= crop_size, "Crop size must be smaller than the image"
    H, W = prediction.shape

    c_h = (H - crop_size) // 2
    c_w = (W - crop_size) // 2

    c_h = (c_h + 1) // 2 * 2
    c_w = (c_w + 1) // 2 * 2

    return c_h, c_w

def create_sample_submission(args):
    model = build_model(args)
    denoise_key = "frame_09"
    
    for sequence in args.val_dir.glob("**/*.npz"):
        sequence_array = np.load(sequence)
        c_h, c_w = center_crop_prediction_coords(sequence_array[denoise_key])
        result = make_prediction(sequence_array, c_h, c_w, model)

        save_path = args.save_to / sequence.parent.relative_to(args.val_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / sequence.stem, result)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--val_dir", type=Path, default="val_2")
    parser.add_argument("--save_to", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--device", type=str, default='cuda', choices=['cpu','cuda'])

    args = parser.parse_args()

    create_sample_submission(args)
