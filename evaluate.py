"""Re-run the paper's experiments on your own dataset folder (e.g. Set14, BSD68, Kodak24).
Usage: python evaluate.py /path/to/images --sigma 25 --scale 2
Prints mean PSNR/SSIM for denoising and super-resolution baselines vs. this system."""
import argparse, glob, os
import cv2, numpy as np
import enhancer as E

p = argparse.ArgumentParser()
p.add_argument("folder"); p.add_argument("--sigma", type=float, default=25)
p.add_argument("--scale", type=int, default=2); p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
rng = np.random.default_rng(a.seed)
files = sorted(sum([glob.glob(os.path.join(a.folder, e)) for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp")], []))
den, sr = {}, {}
for fp in files:
    im = cv2.imread(fp)
    ny = np.clip(im + rng.normal(0, a.sigma, im.shape), 0, 255).astype(np.uint8)
    h = a.sigma * (0.5 if a.sigma <= 15 else 0.8)
    for k, o in {"Noisy": ny, "Gaussian": cv2.GaussianBlur(ny, (5, 5), 0), "Median": cv2.medianBlur(ny, 5),
                 "Bilateral": cv2.bilateralFilter(ny, 9, 75, 75), "NLM (ours)": E.denoise(ny, h)}.items():
        den.setdefault(k, []).append((E.psnr(o, im), E.ssim(o, im)))
    s = a.scale; H, W = im.shape[0] // s * s, im.shape[1] // s * s; gt = im[:H, :W]
    lr = cv2.resize(cv2.GaussianBlur(gt, (0, 0), 0.4 * s), (W // s, H // s), interpolation=cv2.INTER_AREA)
    for k, o in {"Bicubic": cv2.resize(lr, (W, H), interpolation=cv2.INTER_CUBIC),
                 "Lanczos+sharpen (ours)": E.upscale(lr, s, 0.6)}.items():
        sr.setdefault(k, []).append((E.psnr(o, gt), E.ssim(o, gt)))
for title, d in (("DENOISING sigma=%g" % a.sigma, den), ("SUPER-RESOLUTION x%d" % a.scale, sr)):
    print("\n" + title)
    for k, v in d.items():
        print(f"  {k:26s} PSNR {np.mean([x[0] for x in v]):6.2f} dB   SSIM {np.mean([x[1] for x in v]):.4f}")
