import torch
import os, glob, argparse, torch
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
from src.models.unet import UNet
from src.data.oldphoto_transforms import OldPhotoMaker
import numpy as np

def build_argparser():
    p = argparse.ArgumentParser("Visualisation classe: Original | Old N&B | Pred (colorisée)")
    p.add_argument("--ckpt", required=True, help="chemin du checkpoint .pt (state_dict ou {model:...})")
    p.add_argument("--root", required=True, help="racine dataset (ex: ./data/imagenet64_img_full)")
    p.add_argument("--split", default="test", choices=["train","val","test"], help="split à lire")
    p.add_argument("--class_idx", type=int, required=True, help="index de classe (0..K-1)")
    p.add_argument("--n", type=int, default=24, help="nb d'images à afficher")
    p.add_argument("--img_size", type=int, default=64, help="taille (doit matcher l'entrainement)")
    p.add_argument("--out", default="viz_class.png", help="fichier de sortie")
    p.add_argument("--fp16", action="store_true", help="inférence en demi-précision")
    return p

def main():
    args = build_argparser().parse_args()

    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"[!] Checkpoint introuvable: {args.ckpt}")

    split_dir = os.path.join(args.root, args.split)
    classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    if not classes:
        raise SystemExit(f"[!] Aucune classe dans {split_dir}")
    if args.class_idx < 0 or args.class_idx >= len(classes):
        raise SystemExit(f"[!] CLASS_IDX={args.class_idx} hors bornes (0..{len(classes)-1})")

    cls_name = classes[args.class_idx]
    cls_dir  = os.path.join(split_dir, cls_name)
    print(f"[i] Classe: index={args.class_idx} -> '{cls_name}' ({cls_dir})")

    # Lister quelques images
    paths=[]
    for ext in ("*.png","*.jpg","*.jpeg","*.JPEG","*.webp","*.bmp","*.tif","*.tiff"):
        paths += glob.glob(os.path.join(cls_dir, ext))
    if not paths:
        raise SystemExit(f"[!] Aucune image trouvée dans {cls_dir}")
    paths = sorted(paths)[:max(args.n,1)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger modèle
    model = UNet(in_ch=1, base=64).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    state = state.get("model", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[i] load_state: missing={len(missing)} unexpected={len(unexpected)}")

    if args.fp16 and device.type == "cuda":
        model = model.half()
    model.eval()

    # Transfos
    to_tensor = T.ToTensor()
    resize    = T.Resize((args.img_size, args.img_size), interpolation=T.InterpolationMode.BILINEAR)
    old_maker = OldPhotoMaker()

    # Construire batch
    orig_list, old_list = [], []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = resize(img)
        rgb = to_tensor(img).float()                 # [3,H,W]
        rgb_np = (rgb.permute(1,2,0).numpy()).copy() # [H,W,3] float32 in [0,1]
        gray01 = old_maker(rgb_np).astype('float32') # [H,W] float32 in [0,1]
        old = torch.from_numpy(gray01).unsqueeze(0).float()  # [1,H,W]
        orig_list.append(rgb)
        old_list.append(old)

    orig = torch.stack(orig_list, 0).to(device)
    old  = torch.stack(old_list, 0).to(device)
    if args.fp16 and device.type == "cuda":
        orig = orig.half()
        old  = old.half()

    # Inférence
    amp_enabled = (args.fp16 and device.type=="cuda")
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
        pred = torch.clamp(model(old),0,1).float().cpu()

    # Grille (Original | Old | Pred)
    rows = []
    for i in range(orig.size(0)):
        old3 = old[i].detach().float().cpu().repeat(3,1,1)
        rows.append(torch.stack([orig[i].detach().float().cpu(), old3, pred[i]], 0))
    grid = make_grid(torch.cat(rows, 0), nrow=3, padding=2)
    save_image(grid, args.out)
    print(f"[✓] Sauvé: {args.out}")

if __name__ == "__main__":
    main()
