import h5py, numpy as np, os, argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--h5", required=True, help="path to camelyonpatch_level_2_split_train_x.h5")
parser.add_argument("--outdir", default="data/pcam/sample_images")
parser.add_argument("--n", type=int, default=500)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
with h5py.File(args.h5, "r") as f:
    X = f["x"][:]

    n = min(args.n, X.shape[0])
    for i in range(n):
        img = X[i]
        if img.dtype != 'uint8':
            img = (255 * (img - img.min()) / (img.max() - img.min() + 1e-8)).astype('uint8')
        Image.fromarray(img).save(os.path.join(args.outdir, f"pcam_{i:06d}.png"))
print("Saved", n, "images to", args.outdir)

