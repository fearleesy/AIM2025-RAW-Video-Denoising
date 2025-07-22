import os
import torch
import argparse
from pathlib import Path
from tqdm.auto import tqdm

from dataset import RawVideoTrainDataset
from model import UNetDenoise

def build_model(args):
    model = UNetDenoise(
        in_channels=args.in_channels, 
        out_channels=args.out_channels
    )
    model.to(args.device)
    return model

def build_dataloaders(args):
    ds_train = RawVideoTrainDataset(
        root_dir=args.train_dir,
        crop_size=args.crop_size
    )

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    return dl_train

def main(args):
    model = build_model(args)
    dl_train = build_dataloaders(args)
    loss_fn = torch.nn.L1Loss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"=> Resuming from checkpoint '{args.resume_ckpt}' (epoch {start_epoch})")


    use_keys = [f"frame_{i:02d}" for i in range(10)]  # frame_00 ... frame_09

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        train_losses = []
        pbar_tr = tqdm(dl_train, desc=f"[Train] Epoch {epoch + 1}", leave=False)
        for batch in pbar_tr:
            x_batch = torch.cat([batch[k] for k in use_keys], dim=1)
            y_batch = batch["gt"]

            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar_tr.set_postfix(train_loss=f"{train_losses[-1]:.6f}")
        pbar_tr.close()

        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"Epoch {epoch + 1} - train loss: {avg_train_loss:.7f}")

        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.num_epochs:
            os.makedirs(f"checkpoints/{args.save_dir}", exist_ok=True)
            ckpt_path = f"checkpoints/{args.save_dir}/epoch_{epoch+1}.pt"
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch+1},
                       ckpt_path)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=Path, default="train")
    parser.add_argument('--in_channels', type=int, default=40)
    parser.add_argument('--out_channels', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--lr', type=int, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu','cuda'])
    parser.add_argument('--save_dir', type=str, default='base')
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--resume_ckpt', type=str, default=None)
    
    main(parser.parse_args())

    
