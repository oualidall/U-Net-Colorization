import argparse
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np

from src.models.unet import UNet
from src.data.oldphoto_transforms import OldPhotoMaker

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",".JPEG",".JPG",".PNG"}

def list_images(p: Path):
    if p.is_file() and p.suffix in IMG_EXTS:
        return [p]
    if p.is_dir():
        return [q for q in p.rglob("*") if q.suffix in IMG_EXTS]
    return []

def to_old_gray_tensor(pil_img, img_size):
    tx = T.Resize((img_size, img_size))
    to_tensor = T.ToTensor()
    old = OldPhotoMaker()
    img = tx(pil_img.convert("RGB"))
    rgb = to_tensor(img)              # [3,H,W] [0,1]
    rgb_np = np.moveaxis(rgb.numpy(), 0, 2)  # HWC
    gray = old(rgb_np)                # [H,W] [0,1]
    x = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
    return x

@torch.no_grad()
def infer_image(model, img_path: Path, out_root: Path, img_size: int):
    img = Image.open(img_path).convert("RGB")
    x = to_old_gray_tensor(img, img_size).to(next(model.parameters()).device)
    y = model(x).clamp(0,1).squeeze(0)        # [3,H,W]
    y_pil = T.ToPILImage()(y.cpu())
    # sortie: garder la structure d'input
    rel = img_path.name
    out_dir = out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(rel).stem + "_color.png")
    y_pil.save(out_path)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out",   required=True)
    ap.add_argument("--img_size", type=int, default=64)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_ch=1, out_ch=3).to(device).eval()

    # charge state_dict pur ou checkpoint {"model":...}
    state = torch.load(a.ckpt, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    in_path = Path(a.input); out_root = Path(a.out)
    files = list_images(in_path)
    if not files:
        print(f"[!] Rien trouvé dans: {a.input}")
        return
    for f in files:
        print(f"Colorizing {f.name}...")
        out_path = infer_image(model, f, out_root, a.img_size)
        print("  ->", out_path)

if __name__ == "__main__":
    main()
