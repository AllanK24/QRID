import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for headless environments

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# ---------- config ----------
DIR_A = Path("/home/omni/Programming/QRID/QRID/validation_datasets/coco_val_noisy_low/images/val2017")
DIR_B = Path("/home/omni/Programming/QRID/QRID/validation_datasets/coco_val_noisy_medium/images/val2017")
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
MAX_PAIRS = 10
FIGSIZE = (9, 4)
OUTDIR = Path("visual_comparisons")
# ----------------------------

OUTDIR.mkdir(exist_ok=True)

def relative_images(root: Path, exts):
    return {
        p.relative_to(root)
        for p in root.rglob("*")
        if p.suffix.lower() in exts
    }

def common_relatives(dir_a: Path, dir_b: Path, exts):
    rels_a = relative_images(dir_a, exts)
    rels_b = relative_images(dir_b, exts)
    return sorted(rels_a & rels_b)

def save_comparison_image(rel, dir_a, dir_b, figsize):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    paths = [dir_a / rel, dir_b / rel]
    labels = ["A", "B"]

    for ax, path, label in zip(axes, paths, labels):
        try:
            img = Image.open(path)
            ax.imshow(img)
            ax.set_title(f"{label}: {rel.name}", fontsize=9)
            ax.axis("off")
        except Exception as e:
            print(f"Error opening {path}: {e}")
            ax.text(0.5, 0.5, "Error loading image", ha="center", va="center")
            ax.axis("off")

    plt.tight_layout()
    out_path = OUTDIR / f"comparison_{rel.stem}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] {out_path}")

def save_side_by_side(dir_a, dir_b, relatives, figsize):
    for rel in relatives:
        save_comparison_image(rel, dir_a, dir_b, figsize)

if __name__ == "__main__":
    shared = common_relatives(DIR_A, DIR_B, ALLOWED_EXTS)
    if MAX_PAIRS:
        shared = shared[:MAX_PAIRS]
    if not shared:
        print("No matching images found.")
    else:
        print(f"Saving {len(shared)} matched image(s) to '{OUTDIR}/'")
        save_side_by_side(DIR_A, DIR_B, shared, FIGSIZE)
