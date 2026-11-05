"""Core image enhancement and restoration algorithms (classical, CPU-only)."""
import cv2
import numpy as np
from scipy.signal import fftconvolve


def denoise(img, strength=10):
    """Non-local means denoising (colour-aware). strength ~ 3..25."""
    if strength <= 0:
        return img
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def clahe_contrast(img, clip=2.0, tile=8):
    """CLAHE applied to the L channel of CIELAB to avoid colour shifts."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def gray_world(img):
    """Gray-world white balance."""
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(0)
    f *= means.mean() / (means + 1e-6)
    return np.clip(f, 0, 255).astype(np.uint8)


def gamma(img, g=1.0):
    lut = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(img, lut)


def unsharp(img, amount=0.8, sigma=1.2):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def _psf(sigma, size=None):
    size = size or int(6 * sigma + 1) | 1
    ax = np.arange(size) - size // 2
    k = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def wiener_deblur(img, sigma=1.5, K=0.01):
    """Frequency-domain Wiener deconvolution assuming a Gaussian PSF."""
    f = img.astype(np.float32) / 255.0
    psf = _psf(sigma)
    # embed PSF and roll its centre to the origin so deconvolution does not shift the image
    pad = np.zeros(f.shape[:2], np.float32)
    pad[:psf.shape[0], :psf.shape[1]] = psf
    pad = np.roll(pad, (-(psf.shape[0] // 2), -(psf.shape[1] // 2)), axis=(0, 1))
    H = np.fft.fft2(pad)
    G = np.conj(H) / (np.abs(H) ** 2 + K)
    out = np.empty_like(f)
    for c in range(3):
        out[..., c] = np.real(np.fft.ifft2(np.fft.fft2(f[..., c]) * G))
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def richardson_lucy(img, sigma=1.5, iters=15):
    f = np.clip(img.astype(np.float32) / 255.0, 1e-4, 1)
    psf = _psf(sigma)
    psf_m = psf[::-1, ::-1]
    est = f.copy()
    for _ in range(iters):
        for c in range(3):
            conv = fftconvolve(est[..., c], psf, mode="same") + 1e-6
            est[..., c] *= fftconvolve(f[..., c] / conv, psf_m, mode="same")
    return (np.clip(est, 0, 1) * 255).astype(np.uint8)


def upscale(img, scale=2, sharpen=0.6):
    """Lanczos interpolation followed by mild edge-aware sharpening."""
    if scale == 1:
        return img
    up = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    return unsharp(up, sharpen, 1.0) if sharpen > 0 else up


def inpaint(img, mask, radius=3, method="telea"):
    """Restore masked (damaged) regions. mask: uint8, 255 = damaged."""
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(img, mask, radius, flag)


def auto_scratch_mask(img, thresh=40, dilate=2):
    """Heuristic detector for thin bright/dark scratches (top-hat transform)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    m = cv2.add(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k),
                cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k))
    _, m = cv2.threshold(m, thresh, 255, cv2.THRESH_BINARY)
    return cv2.dilate(m, np.ones((dilate * 2 + 1,) * 2, np.uint8))


def enhance_pipeline(img, denoise_strength=8, deblur_sigma=0.0, contrast_clip=1.5,
                     white_balance=False, gamma_val=1.0, sharpen=0.5, scale=1,
                     mask=None, auto_scratch=False):
    """Full restoration + enhancement chain.
    Order: inpaint -> denoise -> deblur -> white balance -> contrast -> gamma -> upscale/sharpen."""
    out = img.copy()
    if auto_scratch and mask is None:
        mask = auto_scratch_mask(out)
    if mask is not None:
        out = inpaint(out, mask)
    out = denoise(out, denoise_strength)
    if deblur_sigma > 0:
        out = wiener_deblur(out, deblur_sigma, K=0.01)
    if white_balance:
        out = gray_world(out)
    if contrast_clip > 0:
        out = clahe_contrast(out, contrast_clip)
    if gamma_val != 1.0:
        out = gamma(out, gamma_val)
    out = upscale(out, scale, sharpen) if scale > 1 else (unsharp(out, sharpen) if sharpen > 0 else out)
    return out


def psnr(a, b):
    return float(cv2.PSNR(a, b))


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))
