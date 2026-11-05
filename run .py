import sys, json, time
sys.path.insert(0, '/home/claude/backend')
import numpy as np, cv2, enhancer as E
from skimage import data
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
names = ['astronaut', 'chelsea', 'coffee', 'rocket']
def load(n):
    im = cv2.cvtColor(getattr(data, n)(), cv2.COLOR_RGB2BGR)
    s = 384 / max(im.shape[:2]); im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = im.shape[:2]; return im[:h//2*2, :w//2*2]
imgs = {n: load(n) for n in names}
def noise(im, s): return np.clip(im.astype(np.float32) + rng.normal(0, s, im.shape), 0, 255).astype(np.uint8)
def blur(im, s): return cv2.GaussianBlur(im, (0, 0), s)
def M(a, b): return E.psnr(a, b), E.ssim(a, b)
def avg(res): return {k: [float(np.mean([r[i] for r in v])) for i in (0, 1)] for k, v in res.items()}
R = {}; times = {}

def timed(name, fn, *a):
    t = time.time(); o = fn(*a); times.setdefault(name, []).append(time.time() - t); return o

# ---- A. Denoising, sigma sweep
sweep = [5, 10, 15, 25, 35]; den_sweep = {}
for s in sweep:
    res = {}
    for n, im in imgs.items():
        ny = noise(im, s)
        outs = {'Noisy input': ny,
                'Gaussian 5x5': timed('Gaussian', cv2.GaussianBlur, ny, (5, 5), 0),
                'Median 5x5': timed('Median', cv2.medianBlur, ny, 5),
                'Bilateral': timed('Bilateral', cv2.bilateralFilter, ny, 9, 75, 75),
                'NLM (ours)': timed('NLM', E.denoise, ny, (s * 0.5 if s <= 15 else s * 0.8))}
        for k, o in outs.items(): res.setdefault(k, []).append(M(o, im))
    den_sweep[s] = avg(res)
R['denoise'] = den_sweep

# ---- B. Deblurring (blur sigma 2 + noise 2)
db = {}
for bs in [1, 1.5, 2, 2.5, 3]:
    res = {}
    for n, im in imgs.items():
        bl = noise(blur(im, bs), 1.5)
        outs = {'Blurred input': bl, 'Unsharp mask': E.unsharp(bl, 1.0, bs),
                'Wiener': timed('Wiener', E.wiener_deblur, bl, bs, 0.01),
                'Richardson-Lucy': timed('RL', E.richardson_lucy, bl, bs, 15)}
        for k, o in outs.items(): res.setdefault(k, []).append(M(o, im))
    db[bs] = avg(res)
R['deblur'] = db

# ---- C. Super-resolution x2 and x3
sr = {}
for sc in [2, 3, 4]:
    res = {}
    for n, im in imgs.items():
        h, w = im.shape[:2]; h2, w2 = h // sc * sc, w // sc * sc; gt = im[:h2, :w2]
        lr = cv2.resize(cv2.GaussianBlur(gt, (0, 0), 0.4 * sc), (w2 // sc, h2 // sc), interpolation=cv2.INTER_AREA)
        sz = (w2, h2)
        outs = {'Nearest': cv2.resize(lr, sz, interpolation=cv2.INTER_NEAREST),
                'Bilinear': cv2.resize(lr, sz, interpolation=cv2.INTER_LINEAR),
                'Bicubic': cv2.resize(lr, sz, interpolation=cv2.INTER_CUBIC),
                'Lanczos': cv2.resize(lr, sz, interpolation=cv2.INTER_LANCZOS4),
                'Lanczos+sharpen (ours)': E.unsharp(cv2.resize(lr, sz, interpolation=cv2.INTER_LANCZOS4), 0.6, 1.0)}
        for k, o in outs.items(): res.setdefault(k, []).append(M(o, gt))
    sr[sc] = avg(res)
R['sr'] = sr

# ---- D. Inpainting (random scratches + blobs)
def make_mask(shape):
    m = np.zeros(shape[:2], np.uint8)
    for _ in range(12):
        p1 = (int(rng.integers(0, shape[1])), int(rng.integers(0, shape[0])))
        p2 = (int(np.clip(p1[0] + rng.integers(-120, 120), 0, shape[1]-1)), int(np.clip(p1[1] + rng.integers(-120, 120), 0, shape[0]-1)))
        cv2.line(m, p1, p2, 255, int(rng.integers(1, 4)))
    for _ in range(4):
        cv2.circle(m, (int(rng.integers(0, shape[1])), int(rng.integers(0, shape[0]))), int(rng.integers(6, 14)), 255, -1)
    return m
res = {}; masks = {}
for n, im in imgs.items():
    m = make_mask(im.shape); masks[n] = m
    dmg = im.copy(); dmg[m > 0] = 255
    outs = {'Damaged input': dmg,
            'Gaussian-blur fill': np.where(m[..., None] > 0, cv2.GaussianBlur(dmg, (0, 0), 6), dmg).astype(np.uint8),
            'Navier-Stokes': timed('NS', E.inpaint, dmg, m, 3, 'ns'),
            'Telea FMM (ours)': timed('Telea', E.inpaint, dmg, m, 3, 'telea')}
    for k, o in outs.items(): res.setdefault(k, []).append(M(o, im))
R['inpaint'] = avg(res)

# ---- E. Low-contrast / illumination
res = {}
for n, im in imgs.items():
    low = np.clip(im.astype(np.float32) * 0.45 + 70, 0, 255).astype(np.uint8)
    he = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    lh = cv2.cvtColor(low, cv2.COLOR_BGR2YCrCb); lh[..., 0] = cv2.equalizeHist(lh[..., 0]); he = cv2.cvtColor(lh, cv2.COLOR_YCrCb2BGR)
    outs = {'Low-contrast input': low, 'Gamma 1.5': E.gamma(low, 1.5), 'Global HE': he,
            'CLAHE (L-channel, ours)': E.clahe_contrast(low, 2.0)}
    for k, o in outs.items(): res.setdefault(k, []).append(M(o, im))
R['contrast'] = avg(res)

# ---- F. Combined degradation, full pipeline vs single-op baselines
res = {}; sample = None
for n, im in imgs.items():
    deg = noise(blur(im, 1.5), 12); deg = np.clip(deg.astype(np.float32) * 0.7 + 25, 0, 255).astype(np.uint8)
    outs = {'Degraded input': deg,
            'NLM only': E.denoise(deg, 8),
            'CLAHE only': E.clahe_contrast(deg, 2.0),
            'NLM + Wiener': E.wiener_deblur(E.denoise(deg, 8), 1.5, 0.01),
            'Full pipeline (ours)': timed('Pipeline', E.enhance_pipeline, deg, 8, 1.5, 2.0, False, 1.0, 0.4, 1)}
    for k, o in outs.items(): res.setdefault(k, []).append(M(o, im))
    if n == 'astronaut': sample = (im, outs)
R['combined'] = avg(res)
R['times'] = {k: float(np.mean(v)) for k, v in times.items()}
json.dump(R, open('/home/claude/exp/results.json', 'w'), indent=1)

# ---------- figures
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .3})
def line(d, xs, fig, xl, title, idx=0, order=None):
    plt.figure(figsize=(6.2, 3.8))
    keys = order or list(d[xs[0]].keys())
    for k in keys:
        plt.plot(xs, [d[x][k][idx] for x in xs], marker='o', label=k, lw=2.2 if 'ours' in k else 1.4)
    plt.xlabel(xl); plt.ylabel('PSNR (dB)' if idx == 0 else 'SSIM'); plt.title(title); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f'/home/claude/exp/{fig}.png', dpi=170); plt.close()
line(den_sweep, sweep, 'fig_denoise_psnr', 'Noise std (sigma)', 'Denoising: PSNR vs. noise level')
line(den_sweep, sweep, 'fig_denoise_ssim', 'Noise std (sigma)', 'Denoising: SSIM vs. noise level', 1)
line(db, [1, 1.5, 2, 2.5, 3], 'fig_deblur_psnr', 'Blur std (sigma)', 'Deblurring: PSNR vs. blur level')
def bars(d, fig, title, rot=15):
    ks = list(d.keys()); fig_, ax = plt.subplots(1, 2, figsize=(8, 3.6))
    for i, (lab, c) in enumerate([('PSNR (dB)', '#3b6fb6'), ('SSIM', '#c4622d')]):
        v = [d[k][i] for k in ks]; cols = [c if 'ours' not in k else '#2a9d5c' for k in ks]
        ax[i].bar(range(len(ks)), v, color=cols); ax[i].set_xticks(range(len(ks))); ax[i].set_xticklabels(ks, rotation=rot, ha='right', fontsize=7.5); ax[i].set_ylabel(lab)
        if i == 1: ax[i].set_ylim(max(0, min(v) - .1), 1)
        else: ax[i].set_ylim(min(v) - 2, max(v) + 1)
    fig_.suptitle(title); fig_.tight_layout(); fig_.savefig(f'/home/claude/exp/{fig}.png', dpi=170); plt.close()
bars(R['sr'][2], 'fig_sr2', 'Super-resolution x2 (mean of test images)', 25)
bars(R['inpaint'], 'fig_inpaint', 'Inpainting of scratches and blobs', 20)
bars(R['contrast'], 'fig_contrast', 'Low-contrast enhancement', 20)
bars(R['combined'], 'fig_combined', 'Combined degradation: full pipeline vs. single operations', 25)
# qualitative figure
gt, outs = sample; ks = ['Degraded input', 'NLM only', 'NLM + Wiener', 'Full pipeline (ours)']
fig, ax = plt.subplots(1, 5, figsize=(12, 2.9))
for a, (t, im) in zip(ax, [('Ground truth', gt)] + [(k, outs[k]) for k in ks]):
    a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)[40:200, 120:280]); a.axis('off')
    a.set_title(t + ('' if t == 'Ground truth' else f"\n{E.psnr(im, gt):.2f} dB"), fontsize=8)
plt.tight_layout(); plt.savefig('/home/claude/exp/fig_qual.png', dpi=170); plt.close()
# inpaint qualitative
n = 'chelsea'; im = imgs[n]; m = masks[n]; dmg = im.copy(); dmg[m > 0] = 255
fig, ax = plt.subplots(1, 3, figsize=(10, 3.3))
for a, (t, x) in zip(ax, [('Damaged', dmg), ('Restored (Telea)', E.inpaint(dmg, m)), ('Original', im)]):
    a.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)); a.axis('off'); a.set_title(t, fontsize=9)
plt.tight_layout(); plt.savefig('/home/claude/exp/fig_qual_inpaint.png', dpi=170); plt.close()
print(json.dumps({k: R[k] for k in ['inpaint', 'contrast', 'combined', 'times']}, indent=1))
print(json.dumps(R['sr'][2], indent=1)); print(json.dumps(R['denoise'][25], indent=1)); print(json.dumps(R['deblur'][2], indent=1))
