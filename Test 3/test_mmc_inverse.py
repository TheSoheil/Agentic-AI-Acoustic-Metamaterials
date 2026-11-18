import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.ndimage import distance_transform_edt

# 1. Setup 100x100 2D Coordinate Grid in [0, 1]^2
RES = 100
grid_y, grid_x = torch.meshgrid(
    torch.linspace(0, 1, RES), 
    torch.linspace(0, 1, RES), 
    indexing='ij'
)

def render_mmc(P, gx, gy, eps=0.04, p_norm=2.0):
    """
    Forward MMC Renderer: Maps [K, 5] ellipse table P -> [100, 100] density image in [0, 1].
    Columns of P: [x_c, y_c, a, b, theta (radians)]
    """
    xc = P[:, 0].view(-1, 1, 1)
    yc = P[:, 1].view(-1, 1, 1)
    a  = torch.clamp(P[:, 2].view(-1, 1, 1), min=0.02, max=0.60)
    b  = torch.clamp(P[:, 3].view(-1, 1, 1), min=0.02, max=0.12)
    th = P[:, 4].view(-1, 1, 1)

    dx = gx.unsqueeze(0) - xc
    dy = gy.unsqueeze(0) - yc
    cos_t, sin_t = torch.cos(th), torch.sin(th)
    x_rot =  dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t

    phi_i = 1.0 - ((x_rot / a) ** p_norm + (y_rot / b) ** p_norm)

    # Smooth max across K ellipses + regularized Heaviside
    alpha = 30.0
    phi_max = (1.0 / alpha) * torch.logsumexp(alpha * phi_i, dim=0)
    rho_hat = torch.sigmoid(phi_max / eps)
    return rho_hat

def greedy_strut_initialize(rho_img, gx, gy, K=4):
    """
    Step 1: Greedy Strut Pursuit Initialization (Replaces Quadrant-Biased GMM).
    Uses Distance Transform + directional line scanning to find the K longest 
    solid struts one by one, subtracting covered pixels after each strut is placed.
    """
    mask = (rho_img.numpy() > 0.5).astype(np.float32)
    dist_map = distance_transform_edt(mask) / RES  # Local half-thickness at every pixel
    residual = mask.copy()

    # Candidate angle bank (36 angles from -90 deg to +85 deg)
    angles = np.linspace(-np.pi / 2, np.pi / 2, 36, endpoint=False)
    init_list = []

    for k in range(K):
        # Find remaining uncovered medial ridge pixels
        ridge_score = residual * (dist_map + 0.02)
        if ridge_score.max() < 1e-3:
            break

        # Sample top candidate center points along the uncovered ridges
        flat_indices = np.argsort(ridge_score.ravel())[::-1][:150]
        cand_ys, cand_xs = np.unravel_index(flat_indices, mask.shape)
        
        best_score = -1.0
        best_p = None

        # Test candidate struts through ridge peaks to find the longest continuous bar
        for idx in range(0, len(cand_xs), 3):
            xc = cand_xs[idx] / (RES - 1)
            yc = cand_ys[idx] / (RES - 1)
            b_est = np.clip(1.3 * dist_map[cand_ys[idx], cand_xs[idx]], 0.04, 0.09)

            for th in angles:
                cos_t, sin_t = np.cos(th), np.sin(th)
                # March along the line in both directions while inside the solid mask
                for a_test in [0.42, 0.35, 0.28, 0.22]:
                    t_vals = np.linspace(-a_test, a_test, 25)
                    xs = np.clip(((xc + t_vals * cos_t) * (RES - 1)).astype(int), 0, RES - 1)
                    ys = np.clip(((yc + t_vals * sin_t) * (RES - 1)).astype(int), 0, RES - 1)
                    
                    in_solid = mask[ys, xs].mean()
                    new_coverage = residual[ys, xs].mean()
                    # Must stay >= 90% inside solid material and cover new residual pixels
                    if in_solid >= 0.90:
                        score = a_test * (new_coverage + 0.3 * in_solid)
                        if score > best_score:
                            best_score = score
                            best_p = [xc, yc, a_test, b_est, th]
                        break

        init_list.append(best_p)
        # Carve out pixels covered by this strut so the next strut finds a different bar
        P_single = torch.tensor([best_p], dtype=torch.float32)
        with torch.no_grad():
            covered = (render_mmc(P_single, gx, gy, eps=0.03).numpy() > 0.5)
        residual[covered] = 0.0

    return torch.tensor(init_list, dtype=torch.float32)

# ================= MAIN TEST =================
print("=== Forward & Inverse MMC Ellipsoid Test (Exp 03: Greedy Strut Pursuit) ===")

# A. Ground-Truth Target Image (4 intersecting ellipses: Hourglass / X-truss)
P_true = torch.tensor([
    [0.50, 0.50, 0.42, 0.08,  0.785],  # +45 deg diagonal strut
    [0.50, 0.50, 0.42, 0.08, -0.785],  # -45 deg diagonal strut
    [0.50, 0.22, 0.28, 0.07,  0.000],  # Bottom horizontal bar
    [0.50, 0.78, 0.28, 0.07,  0.000],  # Top horizontal bar
], dtype=torch.float32)

with torch.no_grad():
    rho_target = render_mmc(P_true, grid_x, grid_y, eps=0.04)

# B. Step 1: Greedy Strut Pursuit Initialization
K = 4
P_init = greedy_strut_initialize(rho_target, grid_x, grid_y, K=K)
with torch.no_grad():
    rho_init = render_mmc(P_init, grid_x, grid_y, eps=0.04)
    init_bin = (rho_init > 0.5).float()
    targ_bin = (rho_target > 0.5).float()
    init_iou = (init_bin * targ_bin).sum() / ((init_bin + targ_bin).clamp(0, 1).sum() + 1e-8)
print(f"Step 1 (Strut Pursuit Init) Overlap IoU: {init_iou.item()*100:.2f}%")

# C. Step 2 & 3: Differentiable MMC Refinement (250 steps of Adam)
P_opt = P_init.clone().detach().requires_grad_(True)
optimizer = torch.optim.Adam([P_opt], lr=0.015)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250)

print("\n--- Running 250-Step Inverse MMC Optimization ---")
for step in range(250):
    optimizer.zero_grad()
    rho_hat = render_mmc(P_opt, grid_x, grid_y, eps=0.04)
    
    # Pixel MSE + slight penalty on minor-axis fatness so struts never inflate into blobs
    mse_loss = torch.mean((rho_hat - rho_target) ** 2)
    fatness_penalty = 0.002 * torch.mean(torch.relu(P_opt[:, 3] - 0.09) ** 2)
    loss = mse_loss + fatness_penalty
    
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 50 == 0 or step == 249:
        pred_bin = (rho_hat.detach() > 0.5).float()
        iou = (pred_bin * targ_bin).sum() / ((pred_bin + targ_bin).clamp(0, 1).sum() + 1e-8)
        print(f"Step {step:03d} | Pixel MSE Loss: {mse_loss.item():.6f} | Overlap IoU: {iou.item()*100:.2f}%")

# Evaluate Final Model
rho_final = render_mmc(P_opt.detach(), grid_x, grid_y, eps=0.04)
final_iou = ((rho_final > 0.5).float() * targ_bin).sum() / (((rho_final > 0.5).float() + targ_bin).clamp(0, 1).sum() + 1e-8)

# D. Plot and Save 4-Panel Comparison Figure
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
titles = [
    "1. Target Pixel Image (rho_lit)",
    f"2. Step 1: Strut Pursuit (IoU {init_iou.item()*100:.1f}%)",
    f"3. Step 3: Optimized MMC (IoU {final_iou.item()*100:.1f}%)",
    "4. Extracted Ellipsoid Vectors (P)"
]
images = [rho_target, rho_init, rho_final, rho_target]

for idx, ax in enumerate(axes):
    ax.imshow(images[idx].numpy(), cmap="gray_r", origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=1)
    ax.set_title(titles[idx], fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

P_final = P_opt.detach().numpy()
colors = ['#ff3b30', '#007aff', '#34c759', '#ff9500']
for k in range(K):
    xc, yc, a, b, th = P_final[k]
    ell = Ellipse((xc, yc), width=2*a, height=2*b, angle=np.degrees(th),
                  edgecolor=colors[k % len(colors)], facecolor='none', lw=2.5)
    axes[3].add_patch(ell)
    axes[3].plot(xc, yc, 'o', color=colors[k % len(colors)])

plt.tight_layout()
plt.savefig("mmc_inverse_result.png", dpi=200)
print("\nSaved updated visual proof to: mmc_inverse_result.png")
print("\nGround-Truth [K x 5] Table:")
print(np.round(P_true.numpy(), 3))
print("\nExtracted [K x 5] Ellipsoid Table:")
print(np.round(P_final, 3))