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

def load_target_image(custom_path="real_paper.png"):
    """Loads a real literature image if present; otherwise generates a non-MMC re-entrant cell."""
    if os.path.exists(custom_path):
        print(f"Loading real literature image: {custom_path}")
        img = Image.open(custom_path).convert("L").resize((RES, RES))
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.mean() > 0.5:  # Invert white background so solid = 1.0
            arr = 1.0 - arr
        # Clean light-gray background noise (< 0.20 -> 0.0)
        arr = np.where(arr < 0.20, 0.0, arr)
        return torch.tensor(np.flipud(arr).copy(), dtype=torch.float32)

    print("Generating NON-MMC Polygon Target: Re-Entrant Auxetic Cell")
    img = Image.new("L", (RES, RES), 0)
    draw = ImageDraw.Draw(img)
    segments = [
        [(15, 82), (85, 82)], [(15, 18), (85, 18)],
        [(18, 82), (38, 50)], [(38, 50), (18, 18)],
        [(82, 82), (62, 50)], [(62, 50), (82, 18)],
    ]
    for seg in segments:
        draw.line(seg, fill=255, width=9, joint="curve")
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return torch.tensor(np.array(img, dtype=np.float32) / 255.0, dtype=torch.float32)

def render_single_ellipses(P_batch, gx, gy, eps=0.04, p_norm=2.0):
    """Renders each row of P_batch [M, 5] as its own [M, RES, RES] density mask."""
    xc = P_batch[:, 0].view(-1, 1, 1)
    yc = P_batch[:, 1].view(-1, 1, 1)
    a  = torch.clamp(P_batch[:, 2].view(-1, 1, 1), min=0.015, max=0.65)
    b  = torch.clamp(P_batch[:, 3].view(-1, 1, 1), min=0.015, max=0.18)
    th = P_batch[:, 4].view(-1, 1, 1)

    dx = gx.unsqueeze(0) - xc
    dy = gy.unsqueeze(0) - yc
    cos_t, sin_t = torch.cos(th), torch.sin(th)
    x_rot =  dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t

    phi_i = 1.0 - ((torch.abs(x_rot) / a) ** p_norm + (torch.abs(y_rot) / b) ** p_norm)
    return torch.sigmoid(phi_i / eps)

def render_mmc(P, gx, gy, eps=0.04, p_norm=2.0):
    """Forward MMC Union Renderer: Maps [K, 5] table P -> [RES, RES] density image."""
    xc = P[:, 0].view(-1, 1, 1)
    yc = P[:, 1].view(-1, 1, 1)
    a  = torch.clamp(P[:, 2].view(-1, 1, 1), min=0.015, max=0.65)
    b  = torch.clamp(P[:, 3].view(-1, 1, 1), min=0.015, max=0.18)
    th = P[:, 4].view(-1, 1, 1)

    dx = gx.unsqueeze(0) - xc
    dy = gy.unsqueeze(0) - yc
    cos_t, sin_t = torch.cos(th), torch.sin(th)
    x_rot =  dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t

    phi_i = 1.0 - ((torch.abs(x_rot) / a) ** p_norm + (torch.abs(y_rot) / b) ** p_norm)
    alpha = 30.0
    phi_max = (1.0 / alpha) * torch.logsumexp(alpha * phi_i, dim=0)
    return torch.sigmoid(phi_max / eps)

def clamp_params_(P):
    """In-place parameter bounds so a and b NEVER become negative or freeze."""
    with torch.no_grad():
        P[:, 0].clamp_(0.02, 0.98)
        P[:, 1].clamp_(0.02, 0.98)
        P[:, 2].clamp_(0.02, 0.65)
        P[:, 3].clamp_(0.015, 0.16)

def propose_next_strut_2d(rho_target, rho_curr, gx, gy, p_norm=2.0):
    """
    Finds the single best new ellipse to cover the remaining 2D residual:
    1. Distance transform on UNCOVERED residual (ignores 1-px edge slivers).
    2. Ray-march across full solid mask to propose candidate struts.
    3. Exact 2D pixel gain = new solid covered - 1.5 * void spilled.
    """
    full_mask = (rho_target > 0.5).numpy().astype(np.float32)
    void_mask = 1.0 - full_mask
    res_mask  = ((rho_target > 0.5) & (rho_curr < 0.5)).numpy().astype(np.float32)

    dist_full = distance_transform_edt(full_mask) / RES
    dist_res  = distance_transform_edt(res_mask) / RES

    # Stop if only thin 1-px edge slivers remain
    if dist_res.max() < 0.015 or res_mask.sum() < 20:
        return None, 0.0

    # Seed from thickest uncovered residual regions
    flat_idx = np.argsort(dist_res.ravel())[::-1][:80]
    cand_ys, cand_xs = np.unravel_index(flat_idx, full_mask.shape)

    angles = np.linspace(-np.pi / 2, np.pi / 2, 36, endpoint=False)
    steps  = np.linspace(0.0, 0.75, 75)
    proposals = []

    for idx in range(0, len(cand_xs), 4):
        x0, y0 = cand_xs[idx] / (RES - 1), cand_ys[idx] / (RES - 1)
        if dist_res[cand_ys[idx], cand_xs[idx]] < 0.012:
            continue

        for th in angles:
            cos_t, sin_t = np.cos(th), np.sin(th)

            xf = np.clip(((x0 + steps * cos_t) * (RES - 1)).astype(int), 0, RES - 1)
            yf = np.clip(((y0 + steps * sin_t) * (RES - 1)).astype(int), 0, RES - 1)
            f_out = np.where(full_mask[yf, xf] < 0.5)[0]
            t_pos = steps[f_out[0] - 1] if len(f_out) > 0 and f_out[0] > 0 else steps[-1]

            xb = np.clip(((x0 - steps * cos_t) * (RES - 1)).astype(int), 0, RES - 1)
            yb = np.clip(((y0 - steps * sin_t) * (RES - 1)).astype(int), 0, RES - 1)
            b_out = np.where(full_mask[yb, xb] < 0.5)[0]
            t_neg = steps[b_out[0] - 1] if len(b_out) > 0 and b_out[0] > 0 else steps[-1]

            a_est = 0.5 * (t_pos + t_neg)
            if a_est < 0.05:
                continue

            t_mid = 0.5 * (t_pos - t_neg)
            xc = np.clip(x0 + t_mid * cos_t, 0.03, 0.97)
            yc = np.clip(y0 + t_mid * sin_t, 0.03, 0.97)

            s_pts = np.linspace(-a_est * 0.85, a_est * 0.85, 20)
            xs = np.clip(((xc + s_pts * cos_t) * (RES - 1)).astype(int), 0, RES - 1)
            ys = np.clip(((yc + s_pts * sin_t) * (RES - 1)).astype(int), 0, RES - 1)

            quick_score = a_est * res_mask[ys, xs].mean() * (dist_res[ys, xs].mean() + 0.005)
            b_est = np.clip(1.15 * np.median(dist_full[ys, xs]), 0.025, 0.12)
            proposals.append((quick_score, [xc, yc, a_est, b_est, th]))

    if not proposals:
        return None, 0.0

    # Vectorized exact 2D gain check on top 20 proposals
    proposals.sort(key=lambda x: x[0], reverse=True)
    top_P = torch.tensor([p[1] for p in proposals[:20]], dtype=torch.float32)

    with torch.no_grad():
        ell_masks = (render_single_ellipses(top_P, gx, gy, eps=0.03, p_norm=p_norm) > 0.5).float()
        res_t  = torch.tensor(res_mask)
        void_t = torch.tensor(void_mask)
        gains  = (ell_masks * res_t).sum(dim=(1, 2)) - 1.5 * (ell_masks * void_t).sum(dim=(1, 2))
        best_i = torch.argmax(gains).item()
        return top_P[best_i].tolist(), gains[best_i].item()

def fit_mmc_progressive(rho_target, K_max=15, p_norm=2.0, min_gain_frac=0.015):
    """Progressive Residual MMC Pursuit: Adds struts one by one + joint Adam refinement."""
    targ_bin = (rho_target > 0.5).float()
    solid_area = targ_bin.sum().item()
    P_list = []
    rho_curr = torch.zeros_like(rho_target)

    for k in range(K_max):
        cand, gain = propose_next_strut_2d(rho_target, rho_curr, grid_x, grid_y, p_norm=p_norm)
        if cand is None or gain < min_gain_frac * solid_area:
            print(f"Stopping automatically at K={len(P_list)} (remaining 2D gain < {min_gain_frac*100:.1f}% solid area).")
            break

        P_list.append(cand)
        P_opt = torch.tensor(P_list, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([P_opt], lr=0.015)

        # Short joint refinement so placed struts snap tightly before next residual
        for _ in range(45):
            opt.zero_grad()
            rho_hat = render_mmc(P_opt, grid_x, grid_y, eps=0.04, p_norm=p_norm)
            loss = torch.mean((rho_hat - rho_target) ** 2)
            loss.backward()
            opt.step()
            clamp_params_(P_opt)

        P_list = P_opt.detach().tolist()
        with torch.no_grad():
            rho_curr = render_mmc(P_opt.detach(), grid_x, grid_y, eps=0.04, p_norm=p_norm)
            iou = (((rho_curr > 0.5).float() * targ_bin).sum() /
                   (((rho_curr > 0.5).float() + targ_bin).clamp(0, 1).sum() + 1e-8)).item()
        print(f"  + Added Strut #{len(P_list):02d} | 2D Gain: {gain:5.0f} px | Current IoU: {iou*100:.2f}%")
        if iou >= 0.965:
            print(f"Reached >96.5% IoU at K={len(P_list)} — stopping early.")
            break

    # Final 150-step polish with cosine annealing
    P_final_t = torch.tensor(P_list, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([P_final_t], lr=0.012)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)

    for _ in range(150):
        opt.zero_grad()
        rho_hat = render_mmc(P_final_t, grid_x, grid_y, eps=0.035, p_norm=p_norm)
        loss = torch.mean((rho_hat - rho_target) ** 2)
        loss.backward()
        opt.step()
        clamp_params_(P_final_t)
        sched.step()

    with torch.no_grad():
        rho_final = render_mmc(P_final_t.detach(), grid_x, grid_y, eps=0.035, p_norm=p_norm)
        final_iou = (((rho_final > 0.5).float() * targ_bin).sum() /
                     (((rho_final > 0.5).float() + targ_bin).clamp(0, 1).sum() + 1e-8)).item()
        final_mse = torch.mean((rho_final - rho_target) ** 2).item()

    return P_final_t.detach().numpy(), rho_final, final_iou, final_mse

# ================= RUN EXP 06: PROGRESSIVE RESIDUAL MMC =================
print("=== Exp 06: Progressive 2D-Residual MMC Pursuit ===")
P_NORM = 2.0  # Use 2.0 for pure ellipses, or 6.0 for flat-sided MMB bars
rho_target = load_target_image("real_paper.png")

P_final, rho_final, final_iou, final_mse = fit_mmc_progressive(
    rho_target, K_max=15, p_norm=P_NORM, min_gain_frac=0.015
)

print(f"\nFinal Result | K={len(P_final)} Ellipsoids | Overlap IoU: {final_iou*100:.2f}% | MSE: {final_mse:.5f}")
print("\nExtracted [K x 5] Ellipsoid Table (x_c, y_c, a, b, theta_rad):")
print(np.round(P_final, 3))

# Plot 3-Panel Visual Proof
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
titles = [
    "1. Target Pixel Image",
    f"2. Progressive MMC Fit (K={len(P_final)}, IoU {final_iou*100:.1f}%)",
    f"3. Extracted K={len(P_final)} Ellipsoid Vectors"
]
for idx, img in enumerate([rho_target, rho_final, rho_target]):
    axes[idx].imshow(img.numpy(), cmap="gray_r", origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=1)
    axes[idx].set_title(titles[idx], fontsize=10.5, fontweight="bold")
    axes[idx].set_xlim(0, 1); axes[idx].set_ylim(0, 1)

cmap = plt.get_cmap("tab20")
for k, (xc, yc, a, b, th) in enumerate(P_final):
    col = cmap(k % 20)
    ell = Ellipse((xc, yc), width=2*a, height=2*b, angle=np.degrees(th),
                  edgecolor=col, facecolor='none', lw=2.3)
    axes[2].add_patch(ell)
    axes[2].plot(xc, yc, 'o', color=col, markersize=4)

plt.tight_layout()
plt.savefig("mmc_progressive_result.png", dpi=200)
print("Saved figure to: mmc_progressive_result.png")