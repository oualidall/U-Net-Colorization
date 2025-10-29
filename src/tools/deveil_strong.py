import os, numpy as np, cv2, argparse
from PIL import Image

def white_balance_grayworld(img):
    b,g,r = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    kb = np.mean(g)/(np.mean(b)+1e-6); kr = np.mean(g)/(np.mean(r)+1e-6)
    b = np.clip(b*kb,0,255).astype(np.uint8); r = np.clip(r*kr,0,255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([b,g,r]), cv2.COLOR_BGR2RGB)

def deviel_rgb(pred):
    u8 = pred if pred.dtype==np.uint8 else (np.clip(pred,0,1)*255).astype(np.uint8)
    u8 = white_balance_grayworld(u8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    L,a,b = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(L)
    out = cv2.cvtColor(cv2.merge([L,a,b]), cv2.COLOR_LAB2RGB).astype(np.float32)/255.
    out = np.clip(out**0.92, 0, 1)  # léger gamma
    return (out*255).astype(np.uint8)

def process_grid(path_in, path_out):
    img = np.array(Image.open(path_in).convert("RGB"))
    H, W, _ = img.shape; w = W//3
    gt, old, pred = img[:,0:w], img[:,w:2*w], img[:,2*w:3*w]
    pred_f = deviel_rgb(pred)
    out = np.concatenate([gt, old, pred_f], axis=1)
    Image.fromarray(out).save(path_out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  required=True)
    ap.add_argument("--out", dest="out",  required=True)
    args = ap.parse_args()
    process_grid(args.inp, args.out)
