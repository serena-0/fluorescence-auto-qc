"""
Fluorescence Frame Auto-Centering and Quality Control
------------------------------------------------------
Automatically detects circular cells in fluorescence microscopy frames, applies quality metrics,
discards low-quality frames, and outputs cropped 1024×1024 images centered on valid detections.

Author : Serena
AI helper: chatGPT
Date   : 2025-07-21

Dependencies:
- numpy, scipy, opencv-python, tifffile, scikit-learn

Usage:
- Modify `path1` and `path2` to your .ome.tif files
- Run the script with: python main.py
"""

# --------------------------------------------------
# 0. Imports
# --------------------------------------------------
import os, math, cv2, time, warnings
import numpy as np
from scipy import ndimage as ndi
from sklearn.linear_model import LinearRegression, RANSACRegressor
import tifffile

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------
# 1. Paths & Hyperparameters
# --------------------------------------------------
path_root = 'path/to/save/results'
os.makedirs(path_root, exist_ok=True)

# Region of interest in original image (row=a~a+d, col=b~b+d)
a, b, d = 850, 600, 1024

TARGET_SIZE = 1024         # Final saved resolution
FACTOR_DET = 0.3           # Downscale factor for detection

# Quality control thresholds
R_MIN, R_MAX = 25, 40      # Acceptable radius range (on downscaled image)
ROUND_THR    = 0.04        # Mean squared circularity error (relative)
COV_THR      = 0.45        # Minimum edge coverage ratio
CONTR_THR    = 5.0         # Minimum ring contrast

VISUALIZE = False           # If True, show each detection and allow manual rejection

# --------------------------------------------------
# 2. Helper Functions
# --------------------------------------------------
def preprocess(img):
    """Normalize to uint8 and apply median filter."""
    img_f = img.astype(np.float32)
    img_f = (img_f - img_f.min()) / (np.ptp(img_f) + 1e-6) * 255
    img_u8 = img_f.astype(np.uint8)
    return cv2.medianBlur(img_u8, 3)

def coarse_mask(img_u8):
    """Segment main structure via adaptive threshold and morphology."""
    thr = cv2.adaptiveThreshold(img_u8, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 51, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num < 2:
        return mask * 0
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(lbl == biggest, 255, 0).astype(np.uint8)

def max_inscribed_circle(mask):
    """Return (cx, cy, r) of largest circle inside mask."""
    dist = ndi.distance_transform_edt(mask)
    cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
    return cx, cy, dist[cy, cx]

def ransac_circle(edge_pts, res_thresh=5, max_trials=1000):
    """Fit circle via RANSAC: x²+y² + Ax + By + C = 0"""
    xs = edge_pts[:,0]; ys = edge_pts[:,1]
    X = np.column_stack((xs, ys, np.ones_like(xs)))
    y = -(xs**2 + ys**2)
    base = LinearRegression(fit_intercept=False)
    ransac = RANSACRegressor(base, min_samples=3, residual_threshold=res_thresh,
                             max_trials=max_trials, stop_probability=0.99)
    ransac.fit(X, y)
    A, B, C = ransac.estimator_.coef_
    cx = -A / 2; cy = -B / 2
    r = math.sqrt(max(cx**2 + cy**2 - C, 1e-6))
    return cx, cy, r, ransac.inlier_mask_

def edge_coverage(edge_pts, cx, cy, r, tol=3, bins=36):
    """Percentage of 360° edge coverage within ±tol."""
    dx, dy = edge_pts[:,0] - cx, edge_pts[:,1] - cy
    d = np.hypot(dx, dy)
    good = np.abs(d - r) <= tol
    if not np.any(good): return 0.
    ang = (np.arctan2(dy[good], dx[good]) + 2*np.pi) % (2*np.pi)
    hist, _ = np.histogram(ang, bins=bins, range=(0, 2*np.pi))
    return np.count_nonzero(hist) / bins

def ring_contrast(img_u8, cx, cy, r):
    """Mean intensity difference: inner - outer ring."""
    h, w = img_u8.shape
    Y, X = np.ogrid[:h, :w]
    dist = np.hypot(X-cx, Y-cy)
    inner = img_u8[dist <= 0.9*r]
    outer = img_u8[(dist >= 1.1*r) & (dist <= 1.3*r)]
    if outer.size == 0: outer = np.array([1])
    return inner.mean() - outer.mean()

def circle_mse(edge_pts, cx, cy, r):
    """Relative mean squared distance from edge to circle."""
    d = np.hypot(edge_pts[:,0]-cx, edge_pts[:,1]-cy)
    return np.mean((d - r)**2) / (r**2 + 1e-6)

# --------------------------------------------------
# 3. Load Input Data
# --------------------------------------------------
path1 = "data/input1.ome.tif"
path2 = "data/input2.ome.tif"

print('Reading OME-TIFF...')
data1 = tifffile.imread(path1)
data2 = tifffile.imread(path2)
raw = np.concatenate((data1, data2), axis=0)
print("Loaded:", raw.shape, raw.dtype)

# --------------------------------------------------
# 4. Main Loop
# --------------------------------------------------
kept_imgs, kept_locs = [], []

for idx, frame in enumerate(raw):
    print(f"[{idx+1}/{len(raw)}] ", end='')

    big = cv2.resize(frame, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    small = cv2.resize(big, None, fx=FACTOR_DET, fy=FACTOR_DET, interpolation=cv2.INTER_AREA)

    pre = preprocess(small)
    mask = coarse_mask(pre)

    if mask.sum() < 1000:
        print("skip (empty mask)")
        continue

    cx0, cy0, r0 = max_inscribed_circle(mask)
    edges = cv2.Canny(pre, 30, 90)
    ys, xs = np.nonzero(edges)
    if len(xs) < 50:
        print("skip (few edges)")
        continue

    cx1, cy1, r1, inliers = ransac_circle(np.column_stack((xs, ys)))

    ecov = edge_coverage(np.column_stack((xs[inliers], ys[inliers])), cx1, cy1, r1)
    rnd_err = circle_mse(np.column_stack((xs[inliers], ys[inliers])), cx1, cy1, r1)
    contrast = ring_contrast(pre, cx1, cy1, r1)

    qc_msg = f"r={r1:.1f}px  cov={ecov:.2f}  mse={rnd_err:.3f}  ΔI={contrast:.1f}"
    print(qc_msg, end=' ')

    qc = (R_MIN <= r1 <= R_MAX) and (ecov >= COV_THR) and (rnd_err <= ROUND_THR) and (contrast >= CONTR_THR)
    if not qc:
        print("skip (QC fail)")
        continue

    if VISUALIZE:
        disp = cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)
        cv2.circle(disp, (int(cx1), int(cy1)), int(r1), (0,255,0), 2)
        cv2.putText(disp, qc_msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)
        cv2.imshow('QC', cv2.resize(disp, None, fx=2, fy=2))
        key = cv2.waitKey(0)
        if key == 32:
            print("deleted by user")
            continue

    norm_x = cx1 / (TARGET_SIZE*FACTOR_DET) - 0.5
    norm_y = cy1 / (TARGET_SIZE*FACTOR_DET) - 0.5

    cx_full = int(cx1)
    cy_full = int(cy1)
    r_full  = int(35) # Fixed radius for final image cropping (adjust as needed)

    mask = np.zeros_like(big, np.uint8)
    cv2.circle(mask, (cx_full, cy_full), r_full, 255, -1)
    img_circle = np.where(mask, (big >> 8), 0).astype(np.uint8)

    kept_imgs.append(img_circle)
    kept_locs.append([norm_x, norm_y, r1])
    print("keep")

# --------------------------------------------------
# 5. Save Results
# --------------------------------------------------
if kept_imgs:
    new_img = np.stack(kept_imgs)
    new_loc = np.stack(kept_locs).astype(np.float32)
    np.save(os.path.join(path_root, 'new_img1024org.npy'), new_img)
    np.save(os.path.join(path_root, 'new_location1024org.npy'), new_loc)
    print("Saved:", new_img.shape, new_loc.shape)
else:
    print("No frame kept! Nothing saved.")
