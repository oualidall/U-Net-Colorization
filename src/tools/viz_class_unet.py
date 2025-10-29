import torch
import os, glob, random, torch
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
from src.models.unet import UNet
from src.data.oldphoto_transforms import OldPhotoMaker

# --- lire env (définies dans le shell) ---
CKPT      = os.environ.get("CKPT", "checkpoints/unet_best.pt")
ROOT      = os.environ.get("ROOT", "./data/imagenet64_img_full")
SPLIT     = os.environ.get("SPLIT", "test")
IMG_SIZE  = int(os.environ.get("IMG_SIZE", "64"))
CLASS_IDX = int(os.environ.get("CLASS_IDX", "0"))
N         = int(os.environ.get("N", "24"))
OUT       = os.environ.get("OUT", f"viz_class_{CLASS_IDX}_{SPLIT}.png")

# --- trouver la classe ---
split_dir = os.path.join(ROOT, SPLIT)
classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
if not classes:
    raise SystemExit(f"[!] Aucune classe dans {split_dir}")
if CLASS_IDX < 0 or CLASS_IDX >= len(classes):
    raise SystemExit(f"[!] CLASS_IDX={CLASS_IDX} hors bornes (0..{len(classes)-1})")
cls = classes[CLASS_IDX]
cls_dir = os.path.join(split_dir, cls)
print(f"[i] Classe choisie: {CLASS_IDX} -> '{cls}'")

# --- lister images ---
paths = []
for ext in ("*.png","*.jpg","*.jpeg","*.JPEG","*.webp","*.bmp","*.tif","*.tiff"):
    paths += glob.glob(os.path.join(cls_dir, ext))
paths = sorted(paths)[:max(1, N)]
if not paths:
    raise SystemExit(f"[!] Aucune image dans {cls_dir}")

# --- modèle (FP32 pour éviter le mismatch Half/Double) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_ch=1, base=64).to(device).float().eval()
ckpt = torch.load(CKPT, map_location="cpu")
state = ckpt.get("model", ckpt)
miss, unexp = model.load_state_dict(state, strict=False)
print(f"[i] Checkpoint: {CKPT} (missing={len(miss)}, unexpected={len(unexp)})")

# --- transforms ---
to_tensor = T.ToTensor()
resize    = T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BILINEAR)
old_maker = OldPhotoMaker()

orig_list, old_list = [], []
for p in paths:
    img = Image.open(p).convert("RGB")
    img = resize(img)
    rgb = to_tensor(img).clamp(0,1)                # [3,H,W] float32
    rgb_np = (rgb.permute(1,2,0).numpy())          # [H,W,3] float32 [0,1]
    gray01 = old_maker(rgb_np).astype("float32")   # [H,W] float32 [0,1]
    old    = torch.from_numpy(gray01).unsqueeze(0) # [1,H,W]

    orig_list.append(rgb)
    old_list.append(old)

orig = torch.stack(orig_list, 0).to(device).float()
old  = torch.stack(old_list, 0).to(device).float()

# --- inference (sans autocast) ---
with torch.no_grad():
    pred = torch.clamp(model(old),0,1).cpu()
import torchvision.transforms.functional as F
pred = torch.stack([F.adjust_saturation(img, 1.35) for img in pred])
pred = torch.stack([F.adjust_gamma(img, gamma=0.9) for img in pred])

# --- grille: par image -> [orig | old (x3) | pred] en largeur ---
rows = []
for i in range(pred.size(0)):
    triplet = torch.cat([orig[i].cpu(),
                         old[i].repeat(3,1,1).cpu(),
                         pred[i].cpu()], dim=2)     # concat largeur
    rows.append(triplet)
grid = make_grid(rows, nrow=1, padding=2)
save_image(grid, OUT)
print(f"[✓] Sauvé: {OUT}")
