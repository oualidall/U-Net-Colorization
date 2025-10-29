import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 160

# U-Net (base=64, img=64x64): 64→128→256→512, then back
stages = [
    ("Enc1", "64×64×64"),
    ("Enc2", "32×32×128"),
    ("Enc3", "16×16×256"),
    ("Bottleneck", "8×8×512"),
    ("Dec1", "16×16×256"),
    ("Dec2", "32×32×128"),
    ("Dec3", "64×64×64"),
    ("Output", "64×64×3")
]

# x positions of blocks
xs = [0, 2.3, 4.6, 6.9, 9.5, 11.8, 14.1, 16.4]
ys = [0]*len(xs)

# colors
c_enc = "#2563eb"   # blue
c_bot = "#ef4444"   # red
c_dec = "#16a34a"   # green
c_out = "#111827"   # near-black

def add_block(ax, x, y, w, h, label, sub, color):
    r = Rectangle((x,y), w, h, fill=False, lw=2.2, ec=color, joinstyle="round")
    ax.add_patch(r)
    ax.text(x+w/2, y+h+0.35, label, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(x+w/2, y+h/2, sub, ha="center", va="center", fontsize=8, color="#374151")

def arrow(ax, xy1, xy2, color="#6b7280", lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=8,
                                 lw=lw, color=color, shrinkA=4, shrinkB=4))

fig, ax = plt.subplots(figsize=(11,3.2))
w, h = 1.6, 2.2

# encoder
add_block(ax, xs[0], 0, w, h, "Enc 1", stages[0][1], c_enc)
add_block(ax, xs[1], 0, w, h, "Enc 2", stages[1][1], c_enc)
add_block(ax, xs[2], 0, w, h, "Enc 3", stages[2][1], c_enc)
# bottleneck
add_block(ax, xs[3], 0, w, h, "Bottleneck", stages[3][1], c_bot)
# decoder
add_block(ax, xs[4], 0, w, h, "Dec 1", stages[4][1], c_dec)
add_block(ax, xs[5], 0, w, h, "Dec 2", stages[5][1], c_dec)
add_block(ax, xs[6], 0, w, h, "Dec 3", stages[6][1], c_dec)
# output
add_block(ax, xs[7], 0, w, h, "Output conv", stages[7][1], c_out)

# left-to-right arrows
for i in range(len(xs)-1):
    arrow(ax, (xs[i]+w, h/2), (xs[i+1], h/2))

# skip connections (arched)
def skip(i_enc, i_dec, y_off):
    x1 = xs[i_enc]+w
    x2 = xs[i_dec]
    ctrl_y = h + y_off
    # polyline approximation
    ax.plot([x1, (x1+x2)/2, x2], [h, ctrl_y, h], color="#9ca3af", lw=1.6)
    arrow(ax, ((x1+x2)/2, ctrl_y), (x2, h), color="#9ca3af", lw=1.6)

skip(0, 6, 1.6)  # Enc1 -> Dec3
skip(1, 5, 2.1)  # Enc2 -> Dec2
skip(2, 4, 2.6)  # Enc3 -> Dec1

# title + footer
ax.text(sum(xs[:2])/2, -0.9, "Input (grayscale 1×64×64)", ha="center", fontsize=8)
ax.text(xs[7]+w/2, -0.9, "Predicted RGB (3×64×64)", ha="center", fontsize=8)
ax.set_xlim(xs[0]-0.8, xs[-1]+w+0.8)
ax.set_ylim(-1.4, h+3.0)
ax.axis("off")
fig.suptitle("U-Net Architecture (base=64, IMG_SIZE=64)", y=1.02, fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("docs/unet_architecture.png", bbox_inches="tight", dpi=200)
print("[✓] Saved docs/unet_architecture.png")
