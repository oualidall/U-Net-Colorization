import argparse, yaml, os, time, torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from tqdm import tqdm

from src.models.unet import UNet
from src.data.datamodule_tv import build_loaders_tv

def fmt(t): t=int(t); h,m,s=t//3600,(t%3600)//60,t%60; return f"{h:02d}:{m:02d}:{s:02d}"

# RGB -> YUV (BT.709): Y=0.2126R+0.7152G+0.0722B
def rgb_to_yuv(x):  # x in [0,1], Bx3xHxW
    r,g,b = x[:,0:1], x[:,1:2], x[:,2:3]
    y = 0.2126*r + 0.7152*g + 0.0722*b
    u = 0.5*(b - y)/(1-0.0722+1e-6)
    v = 0.5*(r - y)/(1-0.2126+1e-6)
    return y, u, v

def chroma_var(u,v):
    # variance moyenne des chromas (encourage couleurs non fades)
    um = torch.var(u, dim=[2,3], unbiased=False)
    vm = torch.var(v, dim=[2,3], unbiased=False)
    return (um+vm).mean()

def train_one_epoch(model, loader, device, opt, scaler, ssim, amp=True, desc="Train"):
    model.train(); tot_l1=tot_y=tot_ssim=tot_col=0.0; n=0
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)   # x: Bx1xH xW ; y: Bx3xH xW
        opt.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            out = model(x)                    # pas d'activation !
            out = torch.clamp(out, 0, 1)     # clamp dur pour la régression [0,1]

            # pertes
            l1_rgb = F.l1_loss(out, y)
            y_pred, u_pred, v_pred = rgb_to_yuv(out)
            y_true, u_true, v_true = rgb_to_yuv(y)

            l1_y   = F.l1_loss(y_pred, y_true)
            ssim_y = 1.0 - ssim(y_pred, y_true)  # (1-SSIM) à minimiser

            # Encourage la chroma: on veut var_pred proche var_true
            var_t  = chroma_var(u_true, v_true).detach()
            var_p  = chroma_var(u_pred, v_pred)
            col_boost = F.relu(var_t - var_p)

            loss = l1_rgb + 0.7*l1_y + 0.3*ssim_y + 0.05*col_boost

        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()

        bs = x.size(0); n += bs
        tot_l1   += l1_rgb.item()*bs
        tot_y    += l1_y.item()*bs
        tot_ssim += (1.0-ssim_y.item())*bs
        tot_col  += col_boost.item()*bs
        pbar.set_postfix(L1=f"{l1_rgb.item():.4f}", LY=f"{l1_y.item():.4f}", SSIMY=f"{1.0-ssim_y.item():.4f}")
    return tot_l1/n, tot_y/n, tot_ssim/n, tot_col/n

@torch.no_grad()
def validate(model, loader, device, ssim, amp=True):
    model.eval(); t1=t2=t3=t4=0.0; n=0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(enabled=amp):
            out = torch.clamp(model(x), 0, 1)
            l1_rgb = F.l1_loss(out, y)
            y_pred, u_pred, v_pred = rgb_to_yuv(out)
            y_true, u_true, v_true = rgb_to_yuv(y)
            l1_y   = F.l1_loss(y_pred, y_true)
            ssim_y = ssim(y_pred, y_true)
            var_t  = chroma_var(u_true, v_true)
            var_p  = chroma_var(u_pred, v_pred)
            col_boost = F.relu(var_t - var_p)
        bs=x.size(0); n+=bs
        t1+=l1_rgb.item()*bs; t2+=l1_y.item()*bs; t3+=ssim_y.item()*bs; t4+=col_boost.item()*bs
    return t1/n, t2/n, t3/n, t4/n

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(args.cfg))
    tr, va = build_loaders_tv(root=args.root,
                              img_size=cfg["data"]["img_size"],
                              batch_size=cfg["train"]["batch_size"],
                              num_workers=cfg["data"].get("num_workers",4))

    model = UNet(in_ch=1, base=64).to(device)
    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"[i] Loaded pretrained: {args.pretrained}")

    opt   = AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    ssim  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    scaler= GradScaler(enabled=(device.type=="cuda"))
    amp   = True

    os.makedirs(args.save_dir, exist_ok=True)
    best = 1e9
    for ep in range(1, args.epochs+1):
        t0=time.time()
        tl1, tly, tssimy, tcol = train_one_epoch(model, tr, device, opt, scaler, ssim, amp, f"Train {ep}/{args.epochs}")
        vl1, vly, vssimy, vcol = validate(model, va, device, ssim, amp)
        dt = fmt(time.time()-t0)
        print(f"Epoch {ep}/{args.epochs} | L1: {vl1:.4f} | L1_Y: {vly:.4f} | SSIM_Y: {vssimy:.4f} | ColBoost: {vcol:.4f} | time {dt}")
        if vl1 < best:
            best = vl1
            torch.save({"model":model.state_dict()}, os.path.join(args.save_dir, "unet_best.pt"))
            print(f"  ✅ New best saved: {args.save_dir}/unet_best.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--root", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--pretrained", default="")
    ap.add_argument("--save_dir", default="checkpoints_antivoile")
    args = ap.parse_args(); main(args)
