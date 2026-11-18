import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import distance_transform_edt

RES = 100
grid_y, grid_x = torch.meshgrid(
    torch.linspace(0, 1, RES), 
    torch.linspace(0, 1, RES), 
    indexing='ij'
)

def load_or_create_non_mmc_literature_image(custom_path="real_paper.png"):
    """
    Loads a real cropped literature PNG if present; otherwise generates a
    NON-MMC rasterized polygon image (Classic Re-Entrant Auxetic Honeycomb Cell)
    using flat pixel lines and sharp joints (zero ellipses used in creation).
    """
    if os.path.exists(custom_path):
        print(f"Loading real literature image from: {custom_path}")
        img = Image.open(custom_path).convert("L").resize((RES, RES))
        arr = np.array(img, dtype=np.float32) / 255.0
        # Invert if background is white (1.0) so solid material = 1.0
        if arr.mean() > 0.5:
            arr = 1.0 - arr
        return torch.tensor(np.flipud(arr).copy(), dtype=torch.float32)

    print("Generating NON-MMC Polygon Raster Target: Re-Entrant Auxetic Unit Cell (K=6)")
    img = Image.new("L", (RES, RES), 0)
    draw = ImageDraw.Draw(img)
    w = 9  # Flat pixel wall thickness (rectangular SIMP-like bars, NOT ellipses)

    # 6 classic re-entrant auxetic polygon struts in pixel coordinates (0..99)
    segments = [
        [(15, 82), (85, 82)],  # Top horizontal beam
        [(15, 18), (85, 18)],  # Bottom horizontal beam
        [(18, 82), (38, 50)],  # Top-left inward re-entrant V-strut
        [(38, 50), (18, 18)],  # Bottom-left inward re-entrant V-strut
        [(82, 82), (62, 50)],  # Top-right inward re-entrant V-strut
        [(62, 50), (82, 18)],  # Bottom-right inward re-entrant V-strut
    ]
    for seg in segments:
        draw.line(seg, fill=255, width=w, joint="curve")

    # Slight anti-aliasing typical of published SIMP density plots
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr, dtype=torch.float32)

def render_mmc(P, gx, gy, eps=0.04, p_norm=2.0):
    """Forward MMC Renderer: Maps [K, 5] ellipse table P -> [100, 100] density image."""
    xc = P[:, 0].view(-1, 1, 1)
    yc = P[:, 1].view(-1, 1, 1)
    a  = torch.clamp(P[:, 2].view(-1, 1, 1), min=0.02, max=0.60)
    b  = torch.clamp(P[:, 3].view(-1, 1, 1), min=0.02, max=0.14)
    th = P[:, 4].view(-1, 1, 1)

    dx = gx.unsqueeze(0) - xc
    dy = gy.unsqueeze(0) - yc
    cos_t, sin_t = torch.cos(th), torch.sin(th)
    x_rot =  dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t

    phi_i = 1.0 - ((x_rot / a) ** p_norm + (y_rot / b) ** p_norm)
    alpha = 30.0
    phi_max = (1.0 / alpha) * torch.logsumexp(alpha * phi_i, dim=0)
    return torch.sigmoid(phi_max / eps)

def greedy_strut_initialize_general(rho_img, gx, gy, K=6):
    """Step 1: Ray-Marching Strut Pursuit on non-MMC pixel mask."""
    mask = (rho_img.numpy() > 0.5).astype(np.float32)
    dist_map = distance_transform_edt(mask) / RES
    residual = mask.copy()

    angles = np.linspace(-np.pi / 2, np.pi / 2, 60, endpoint=False)
    steps = np.linspace(0.0, 0.75, 80)
    init_list = []

    for k in range(K):
        ridge_score = residual * (dist_map + 0.01)
        if ridge_score.max() < 1e-3:
            init_list.append([0.5, 0.5, 0.15, 0.04, 0.0])
            continue

        flat_indices = np.argsort(ridge_score.ravel())[::-1][:140]
        cand_ys, cand_xs = np.unravel_index(flat_indices, mask.shape)
        best_score, best_p = -1.0, None

        for idx in range(0, len(cand_xs), 2):
            x0, y0 = cand_xs[idx] / (RES - 1), cand_ys[idx] / (RES - 1)
            for th in angles:
                cos_t, sin_t = np.cos(th), np.sin(th)

                xf = np.clip(((x0 + steps * cos_t) * (RES - 1)).astype(int), 0, RES - 1)
                yf = np.clip(((y0 + steps * sin_t) * (RES - 1)).astype(int), 0, RES - 1)
                f_out = np.where(mask[yf, xf] < 0.5)[0]
                t_pos = steps[f_out[0] - 1] if len(f_out) > 0 and f_out[0] > 0 else steps[-1]

                xb = np.clip(((x0 - steps * cos_t) * (RES - 1)).astype(int), 0, RES - 1)
                yb = np.clip(((y0 - steps * sin_t) * (RES - 1)).astype(int), 0, RES - 1)
                b_out = np.where(mask[yb, xb] < 0.5)[0]
                t_neg = steps[b_out[0] - 1] if len(b_out) > 0 and b_out[0] > 0 else steps[-1]

                a_est = 0.5 * (t_pos + t_neg)
                if a_est < 0.08:
                    continue

                t_mid = 0.5 * (t_pos - t_neg)
                xc = np.clip(x0 + t_mid * cos_t, 0.05, 0.95)
                yc = np.clip(y0 + t_mid * sin_t, 0.05, 0.95)

                s_pts = np.linspace(-a_est * 0.9, a_est * 0.9, 25)
                xs = np.clip(((xc + s_pts * cos_t) * (RES - 1)).astype(int), 0, RES - 1)
                ys = np.clip(((yc + s_pts * sin_t) * (RES - 1)).astype(int), 0, RES - 1)

                new_cov = residual[ys, xs].mean()
                b_est = np.clip(1.2 * np.median(dist_map[ys, xs]), 0.03, 0.09)
                score = (a_est ** 1.1) * (new_cov ** 1.5)

                if score > best_score:
                    best_score = score
                    best_p = [xc, yc, a_est, b_est, th]

        init_list.append(best_p)
        P_single = torch.tensor([best_p], dtype=torch.float32)
        with torch.no_grad():
            covered = (render_mmc(P_single, gx, gy, eps=0.03).numpy() > 0.5)
        residual[covered] = 0.0

    return torch.tensor(init_list, dtype=torch.float32)

# ================= RUN EXP 05: NON-MMC REAL LITERATURE BENCHMARK =================
print("=== Exp 05: Non-MMC Polygon / Literature Image -> MMC Ellipsoid Extraction ===")
K = 6
rho_target = load_or_create_non_mmc_literature_image("real_paper.png")
targ_bin = (rho_target > 0.5).float()

# Step 1: Ray-March Strut Pursuit
P_init = greedy_strut_initialize_general(rho_target, grid_x, grid_y, K=K)
with torch.no_grad():
    rho_init = render_mmc(P_init, grid_x, grid_y, eps=0.04)
    init_iou = (((rho_init > 0.5).float() * targ_bin).sum() /
                (((rho_init > 0.5).float() + targ_bin).clamp(0, 1).sum() + 1e-8))
print(f"Step 1 (Ray-March Init) Overlap IoU: {init_iou.item()*100:.2f}%")

# Step 2 & 3: Differentiable MMC Refinement
P_opt = P_init.clone().detach().requires_grad_(True)
optimizer = torch.optim.Adam([P_opt], lr=0.015)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250)

for step in range(250):
    optimizer.zero_grad()
    rho_hat = render_mmc(P_opt, grid_x, grid_y, eps=0.04)
    mse_loss = torch.mean((rho_hat - rho_target) ** 2)
    fat_pen  = 0.002 * torch.mean(torch.relu(P_opt[:, 3] - 0.09) ** 2)
    loss = mse_loss + fat_pen
    loss.backward()
    optimizer.step()
    scheduler.step()

with torch.no_grad():
    rho_final = render_mmc(P_opt, grid_x, grid_y, eps=0.04)
    final_iou = (((rho_final > 0.5).float() * targ_bin).sum() /
                 (((rho_final > 0.5).float() + targ_bin).clamp(0, 1).sum() + 1e-8))

print(f"Step 3 (Optimized MMC)  Overlap IoU: {final_iou.item()*100:.2f}% | Final MSE: {mse_loss.item():.5f}")

# Save 4-Panel Figure
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
titles = [
    "1. Non-MMC Pixel Target (Re-Entrant Cell)",
    f"2. Step 1: Ray-March Init (IoU {init_iou.item()*100:.1f}%)",
    f"3. Step 3: Fitted MMC (IoU {final_iou.item()*100:.1f}%)",
    f"4. Extracted K={K} Ellipsoid Scaffold"
]
imgs = [rho_target, rho_init, rho_final, rho_target]
for idx, ax in enumerate(axes):
    ax.imshow(imgs[idx].numpy(), cmap="gray_r", origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=1)
    ax.set_title(titles[idx], fontsize=10.5, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

P_final = P_opt.detach().numpy()
colors = ['#ff3b30', '#007aff', '#34c759', '#ff9500', '#af52de', '#00c7be']
for k in range(K):
    xc, yc, a, b, th = P_final[k]
    ell = Ellipse((xc, yc), width=2*a, height=2*b, angle=np.degrees(th),
                  edgecolor=colors[k % len(colors)], facecolor='none', lw=2.5)
    axes[3].add_patch(ell)
    axes[3].plot(xc, yc, 'o', color=colors[k % len(colors)])

plt.tight_layout()
plt.savefig("mmc_real_lit_result.png", dpi=200)
print("Saved figure to: mmc_real_lit_result.png")
print("\nExtracted [K x 5] Ellipsoid Table (x_c, y_c, a, b, theta_rad):")
print(np.round(P_final, 3))