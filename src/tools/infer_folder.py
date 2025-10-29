import os, torch
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
from src.models.unet import UNet

def load_model(ckpt_path, device):
    model = UNet(in_ch=1, base=64).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    print(f"[i] Model chargé depuis {ckpt_path}")
    return model

@torch.no_grad()
def colorize_folder(input_dir, output_dir, model, device, img_size=64, half=True):
    os.makedirs(output_dir, exist_ok=True)
    tf_gray = T.Compose([
        T.Resize((img_size,img_size)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor()
    ])
    tf_rgb = T.Compose([
        T.Resize((img_size,img_size)),
        T.ToTensor()
    ])

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".jpg",".png",".jpeg",".bmp")):
            continue
        path = os.path.join(input_dir, fname)
        img = Image.open(path).convert("RGB")
        gray = tf_gray(img).unsqueeze(0).to(device)
        if half: gray = gray.half()
        pred = model(gray).clamp(0,1).float().cpu()[0]
        save_image(pred, os.path.join(output_dir, fname.replace(".","_color.")))
        print(f"[✓] Sauvé : {fname}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--half", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    colorize_folder(args.input, args.out, model, device, img_size=args.img_size, half=args.half)
