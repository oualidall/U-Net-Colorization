import os, argparse

p = argparse.ArgumentParser()
p.add_argument("--root", required=True, help="racine du dataset converti (ex: ./data/imagenet64_img_full)")
p.add_argument("--split", default="test", choices=["train","val","test"])
args = p.parse_args()

split_dir = os.path.join(args.root, args.split)
if not os.path.isdir(split_dir):
    raise SystemExit(f"[!] Split introuvable: {split_dir}")

classes = sorted([d for d in os.listdir(split_dir)
                  if os.path.isdir(os.path.join(split_dir, d))])
if not classes:
    raise SystemExit(f"[!] Aucune classe dans {split_dir}")

print(f"[i] {len(classes)} classes trouvées dans {split_dir}")
for i, c in enumerate(classes):
    print(f"{i:04d}  {c}")
