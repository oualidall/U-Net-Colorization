import argparse, yaml
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path
from src.models.unet import UNet
from src.data.datamodule import build_loaders

def evaluate(model, loader, device, crit, amp=True):
    model.eval(); l1s=[]
    with torch.no_grad():
        pbar = tqdm(loader, desc="Val  ", leave=False)
        for batch in pbar:
            if batch is None: continue
            x,y = batch
            x,y = x.to(device), y.to(device)
            with autocast(device.type, enabled=amp):
                yhat = model(x)
                loss = crit(yhat, y)
            l1s.append(loss.item())
            pbar.set_postfix(l1=f"{sum(l1s)/len(l1s):.4f}")
    return sum(l1s)/max(len(l1s),1)

def train_one_epoch(model, loader, device, crit, opt, scaler, amp=True):
    model.train(); l1s=[]
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        if batch is None: continue
        x,y = batch
        x,y = x.to(device), y.to(device)

        opt.zero_grad(set_to_none=True)
        with autocast(device.type, enabled=amp):
            yhat = model(x)
            loss = crit(yhat, y)

        # ✅ ordre correct avec GradScaler
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        l1s.append(loss.item())
        pbar.set_postfix(l1=f"{sum(l1s)/len(l1s):.4f}")
    return sum(l1s)/max(len(l1s),1)

def save_ckpt(model,opt,epoch,path:Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"epoch":epoch}, str(path))

def main(a):
    with open(a.cfg,'r') as f: cfg=yaml.safe_load(f)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size=int(cfg["data"]["img_size"])
    bs=int(cfg["data"]["batch_size"])
    nw=int(cfg["data"]["num_workers"])
    save_dir=Path(cfg["train"]["save_dir"])

    tr,va = build_loaders(a.train_list,a.val_list,img_size=img_size,batch_size=bs,num_workers=nw)
    model=UNet(in_ch=1,out_ch=3).to(device)

    lr=float(cfg["train"]["lr"])
    wd=float(cfg["train"]["weight_decay"])
    opt=AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit=nn.L1Loss()
    scaler=GradScaler(enabled=(device.type=="cuda"))

    best=float("inf")
    for ep in range(1, a.epochs+1):
        print(f"\nEpoch {ep}/{a.epochs}")
        tl1 = train_one_epoch(model,tr,device,crit,opt,scaler,amp=(device.type=="cuda"))
        vl1 = evaluate(model,va,device,crit,amp=(device.type=="cuda"))
        print(f"  Train L1: {tl1:.4f} | Val L1: {vl1:.4f}")
        save_ckpt(model,opt,ep, save_dir/f"unet_epoch{ep:03d}.pt")
        if vl1<best:
            best=vl1
            save_ckpt(model,opt,ep, save_dir/"unet_best.pt")
            print("  ✅ New best model saved.")

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--train_list", required=True)
    ap.add_argument("--val_list", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    a=ap.parse_args(); main(a)
