import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.ndimage import distance_transform_edt

RES = 100
grid_y, grid_x = torch.meshgrid(
    torch.linspace(0, 1, RES), 
    torch.linspace(0, 1, RES), 
    indexing='ij'
)

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

def greedy_strut_initialize_general(rho_img, gx, gy, K):
    """
    General Non-Overfitted Strut Pursuit:
    Uses continuous ray-marching from ridge pixels to measure arbitrary strut lengths
    and midpoints directly from the binary mask (zero hardcoded length arrays).
    """
    mask = (rho_img.numpy() > 0.5).astype(np.float32)
    dist_map = distance_transform_edt(mask) / RES
    residual = mask.copy()

    angles = np.linspace(-np.pi / 2, np.pi / 2, 60, endpoint=False)
    steps = np.linspace(0.0, 0.70, 75)  # Ray-marching steps in unit coordinates
    init_list = []

    for k in range(K):
        ridge_score = residual * (dist_map + 0.01)
        if ridge_score.max() < 1e-3:
            init_list.append([0.5, 0.5, 0.2, 0.05, 0.0])
            continue

        flat_indices = np.argsort(ridge_score.ravel())[::-1][:120]
        cand_ys, cand_xs = np.unravel_index(flat_indices, mask.shape)

        best_score = -1.0
        best_p = None

        for idx in range(0, len(cand_xs), 3):
            x0 = cand_xs[idx] / (RES - 1)
            y0 = cand_ys[idx] / (RES - 1)

            for th in angles:
                cos_t, sin_t = np.cos(th), np.sin(th)

                # March forward (+t) until hitting white void
                xf = np.clip(((x0 + steps * cos_t) * (RES - 1)).astype(int), 0, RES - 1)
                yf = np.clip(((y0 + steps * sin_t) * (RES - 1)).astype(int), 0, RES - 1)
                f_out = np.where(mask[yf, xf] < 0.5)[0]
                t_pos = steps[f_out[0] - 1] if len(f_out) > 0 and f_out[0] > 0 else steps[-1]

                # March backward (-t) until hitting white void
                xb = np.clip(((x0 - steps * cos_t) * (RES - 1)).astype(int), 0, RES - 1)
                yb = np.clip(((y0 - steps * sin_t) * (RES - 1)).astype(int), 0, RES - 1)
                b_out = np.where(mask[yb, xb] < 0.5)[0]
                t_neg = steps[b_out[0] - 1] if len(b_out) > 0 and b_out[0] > 0 else steps[-1]

                a_est = 0.5 * (t_pos + t_neg)
                if a_est < 0.10:
                    continue

                # True midpoint of the continuous chord
                t_mid = 0.5 * (t_pos - t_neg)
                xc = np.clip(x0 + t_mid * cos_t, 0.05, 0.95)
                yc = np.clip(y0 + t_mid * sin_t, 0.05, 0.95)

                # Sample points along this chord to score residual coverage & thickness
                s_pts = np.linspace(-a_est * 0.9, a_est * 0.9, 25)
                xs = np.clip(((xc + s_pts * cos_t) * (RES - 1)).astype(int), 0, RES - 1)
                ys = np.clip(((yc + s_pts * sin_t) * (RES - 1)).astype(int), 0, RES - 1)

                new_cov = residual[ys, xs].mean()
                b_est = np.clip(1.25 * np.median(dist_map[ys, xs]), 0.035, 0.095)
                score = (a_est ** 1.2) * (new_cov ** 1.5)

                if score > best_score:
                    best_score = score
                    best_p = [xc, yc, a_est, b_est, th]

        init_list.append(best_p)
        P_single = torch.tensor([best_p], dtype=torch.float32)
        with torch.no_grad():
            covered = (render_mmc(P_single, gx, gy, eps=0.03).numpy() > 0.5)
        residual[covered] = 0.0

    return torch.tensor(init_list, dtype=torch.float32)

# ================= DEFINE 3 METAMATERIAL TEST BENCHMARKS =================
benchmarks = [
    (
        "Test A: Hourglass Auxetic (K=4)",
        torch.tensor([
            [0.50, 0.50, 0.42, 0.08,  0.785],
            [0.50, 0.50, 0.42, 0.08, -0.785],
            [0.50, 0.22, 0.28, 0.07,  0.000],
            [0.50, 0.78, 0.28, 0.07,  0.000],
        ], dtype=torch.float32)
    ),
    (
        "Test B: Asymmetric Diamond Frame (K=4)",
        torch.tensor([
            [0.34, 0.66, 0.31, 0.075,  0.72],
            [0.66, 0.34, 0.31, 0.075,  0.72],
            [0.66, 0.66, 0.31, 0.055, -0.85],
            [0.34, 0.34, 0.31, 0.055, -0.85],
        ], dtype=torch.float32)
    ),
    (
        "Test C: 6-Strut Kagome Star Cell (K=6)",
        torch.tensor([
            [0.50, 0.50, 0.44, 0.055,  1.047],  # +60 deg diagonal
            [0.50, 0.50, 0.44, 0.055, -1.047],  # -60 deg diagonal
            [0.50, 0.30, 0.38, 0.055,  0.000],  # Lower horizontal bar
            [0.50, 0.70, 0.38, 0.055,  0.000],  # Upper horizontal bar
            [0.24, 0.50, 0.22, 0.050,  1.571],  # Left vertical pillar
            [0.76, 0.50, 0.22, 0.050,  1.571],  # Right vertical pillar
        ], dtype=torch.float32)
    ),
]

fig, axes = plt.subplots(len(benchmarks), 4, figsize=(16, 11.5))
colors = ['#ff3b30', '#007aff', '#34c759', '#ff9500', '#af52de', '#00c7be']

print("=== Exp 04: Generalization Test Across 3 Metamaterial Topologies ===")

for row_idx, (name, P_true) in enumerate(benchmarks):
    K = P_true.shape[0]
    with torch.no_grad():
        rho_target = render_mmc(P_true, grid_x, grid_y, eps=0.04)
        targ_bin = (rho_target > 0.5).float()

    # Step 1: General Ray-Marching Strut Pursuit
    P_init = greedy_strut_initialize_general(rho_target, grid_x, grid_y, K=K)
    with torch.no_grad():
        rho_init = render_mmc(P_init, grid_x, grid_y, eps=0.04)
        init_bin = (rho_init > 0.5).float()
        init_iou = (init_bin * targ_bin).sum() / ((init_bin + targ_bin).clamp(0, 1).sum() + 1e-8)

    # Step 2 & 3: Adam Refinement
    P_opt = P_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([P_opt], lr=0.015)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250)

    for step in range(250):
        optimizer.zero_grad()
        rho_hat = render_mmc(P_opt, grid_x, grid_y, eps=0.04)
        mse_loss = torch.mean((rho_hat - rho_target) ** 2)
        fat_pen  = 0.002 * torch.mean(torch.relu(P_opt[:, 3] - 0.095) ** 2)
        loss = mse_loss + fat_pen
        loss.backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        rho_final = render_mmc(P_opt, grid_x, grid_y, eps=0.04)
        pred_bin = (rho_final > 0.5).float()
        final_iou = (pred_bin * targ_bin).sum() / ((pred_bin + targ_bin).clamp(0, 1).sum() + 1e-8)

    print(f"{name:38s} | Init IoU: {init_iou.item()*100:5.1f}% -> Final IoU: {final_iou.item()*100:6.2f}% (MSE: {mse_loss.item():.6f})")

    # Plot row
    imgs = [rho_target, rho_init, rho_final, rho_target]
    col_titles = [
        f"{name}\n1. Target Pixel Image",
        f"2. Ray-March Init (IoU {init_iou.item()*100:.1f}%)",
        f"3. Optimized MMC (IoU {final_iou.item()*100:.1f}%)",
        f"4. Extracted K={K} Ellipsoids"
    ]
    for col_idx in range(4):
        ax = axes[row_idx, col_idx]
        ax.imshow(imgs[col_idx].numpy(), cmap="gray_r", origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=1)
        ax.set_title(col_titles[col_idx], fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    P_final = P_opt.detach().numpy()
    for k in range(K):
        xc, yc, a, b, th = P_final[k]
        ell = Ellipse((xc, yc), width=2*a, height=2*b, angle=np.degrees(th),
                      edgecolor=colors[k % len(colors)], facecolor='none', lw=2.2)
        axes[row_idx, 3].add_patch(ell)
        axes[row_idx, 3].plot(xc, yc, 'o', color=colors[k % len(colors)], markersize=4)

plt.tight_layout()
plt.savefig("mmc_suite_result.png", dpi=200)
print("\nSaved 3-topology validation figure to: mmc_suite_result.png")