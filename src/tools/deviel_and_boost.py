import os, cv2, numpy as np
from PIL import Image

def deviel_rgb(rgb, sat=1.35, gamma=0.9, do_clahe=True, do_bilateral=True):
    img = (np.clip(rgb,0,1)*255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    # boost saturation
    hsv[...,1] = np.clip(hsv[...,1]*sat, 0, 255)
    v = hsv[...,2] / 255.0
    # gamma < 1 => plus contrasté/lumineux
    v = np.clip(v**gamma, 0, 1)
    v = (v*255).astype(np.uint8)
    if do_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        v = clahe.apply(v)
    hsv[...,2] = v.astype(np.float32)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if do_bilateral:
        out = cv2.bilateralFilter(out, d=5, sigmaColor=40, sigmaSpace=40)
    return np.clip(out.astype(np.float32)/255.0, 0, 1)

def process_grid(path_in, path_out, sat=1.35, gamma=0.9):
    img = np.array(Image.open(path_in).convert("RGB"))/255.0
    # grille 3 colonnes: [GT | OLD | PRED] -> on remplace la 3ᵉ colonne
    h, w, _ = img.shape
    col_w = w // 3
    gt   = img[:, 0:col_w]
    old  = img[:, col_w:2*col_w]
    pred = img[:, 2*col_w:3*col_w]
    pred2 = deviel_rgb(pred, sat=sat, gamma=gamma)
    out = np.concatenate([gt, old, pred2], axis=1)
    Image.fromarray((out*255).astype(np.uint8)).save(path_out)
    print(f"[✓] Sauvé: {path_out}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  required=True)
    ap.add_argument("--out", dest="out",  required=True)
    ap.add_argument("--sat", type=float, default=1.35)
    ap.add_argument("--gamma", type=float, default=0.9)
    args = ap.parse_args()
    process_grid(args.inp, args.out, sat=args.sat, gamma=args.gamma)
