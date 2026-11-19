import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

# 1. Setup 100x100 2D Coordinate Grid in [0, 1]^2
RES = 100
grid_y, grid_x = torch.meshgrid(
    torch.linspace(0, 1, RES), 
    torch.linspace(0, 1, RES), 
    indexing='ij'
)

def render_mmc(P, gx, gy, eps=0.05, p_norm=2.0):
    """
    Forward MMC Renderer: Maps [K, 5] ellipse table P -> [100, 100] density image in [0, 1].
    Columns of P: [x_c, y_c, a, b, theta (radians)]
    """
    xc = P[:, 0].view(-1, 1, 1)
    yc = P[:, 1].view(-1, 1, 1)
    a  = torch.clamp(P[:, 2].view(-1, 1, 1), min=0.02, max=0.60)
    b  = torch.clamp(P[:, 3].view(-1, 1, 1), min=0.02, max=0.25)
    th = P[:, 4].view(-1, 1, 1)

    dx = gx.unsqueeze(0) - xc
    dy = gy.unsqueeze(0) - yc
    cos_t, sin_t = torch.cos(th), torch.sin(th)
    x_rot =  dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t

    phi_i = 1.0 - ((x_rot / a) ** p_norm + (y_rot / b) ** p_norm)

    # Smooth max across K ellipses + regularized Heaviside
    alpha = 25.0
    phi_max = (1.0 / alpha) * torch.logsumexp(alpha * phi_i, dim=0)
    rho_hat = torch.sigmoid(phi_max / eps)
    return rho_hat

def gmm_strut_initialize(rho_img, K=4):
    """
    Step 1: Boundary-aware GMM + Thin-Strut Initialization.
    Extracts skeleton/core pixels (rho > 0.75) and initializes elongated struts
    so fat ellipses don't swallow interior white holes.
    """
    pts_y, pts_x = torch.where(rho_img > 0.75)
    coords = torch.stack([pts_x.float() / (RES - 1), pts_y.float() / (RES - 1)], dim=1).numpy()

    # Fit GMM with multi-start to find principal strut branches
    gmm = GaussianMixture(n_components=K, covariance_type='full', n_init=10, random_state=0)
    gmm.fit(coords)

    init_list = []
    for k in range(K):
        xc, yc = gmm.means_[k]
        cov = gmm.covariances_[k]
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        
        # Stretch major axis along branch & clamp minor axis thin to preserve holes
        a = np.clip(2.4 * np.sqrt(max(eigvals[0], 1e-4)), 0.15, 0.45)
        b = 0.065  # Start as thin strut so white holes stay open!
        theta = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        init_list.append([xc, yc, a, b, theta])

    return torch.tensor(init_list, dtype=torch.float32)

# ================= MAIN TEST =================
print("=== Forward & Inverse MMC Ellipsoid Test (Strut-Preserving) ===")

# A. Ground-Truth Target Image (4 intersecting ellipses: Hourglass / X-truss)
P_true = torch.tensor([
    [0.50, 0.50, 0.42, 0.08,  0.785],  # +45 deg diagonal strut
    [0.50, 0.50, 0.42, 0.08, -0.785],  # -45 deg diagonal strut
    [0.50, 0.22, 0.28, 0.07,  0.000],  # Bottom horizontal bar
    [0.50, 0.78, 0.28, 0.07,  0.000],  # Top horizontal bar
], dtype=torch.float32)

with torch.no_grad():
    rho_target = render_mmc(P_true, grid_x, grid_y, eps=0.04)

# B. Step 1: Thin-Strut GMM Initialization
K = 4
P_init = gmm_strut_initialize(rho_target, K=K)
with torch.no_grad():
    rho_init = render_mmc(P_init, grid_x, grid_y, eps=0.05)

# C. Step 2 & 3: Multi-Start Inverse Gradient Refinement (Picks best of 3 angle seeds)
best_loss = float('inf')
best_P = None

for trial in range(3):
    torch.manual_seed(trial)
    P_cand = P_init.clone().detach()
    if trial > 0:
        # Perturb initial centers & angles slightly to escape quadrant local minima
        P_cand[:, 0:2] += 0.05 * torch.randn_like(P_cand[:, 0:2])
        P_cand[:, 4]   += 0.60 * torch.randn_like(P_cand[:, 4])
    P_cand.requires_grad_(True)

    optimizer = torch.optim.Adam([P_cand], lr=0.025)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=350)

    for step in range(350):
        optimizer.zero_grad()
        rho_hat = render_mmc(P_cand, grid_x, grid_y, eps=0.045)
        
        # Penalize covering white holes slightly more during first half of optimization
        void_weight = 1.8 if step < 180 else 1.0
        pixel_weights = torch.where(rho_target < 0.5, void_weight, 1.0)
        loss = torch.mean(pixel_weights * (rho_hat - rho_target) ** 2)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

    final_mse = torch.mean((render_mmc(P_cand.detach(), grid_x, grid_y, eps=0.04) - rho_target) ** 2).item()
    if final_mse < best_loss:
        best_loss = final_mse
        best_P = P_cand.detach().clone()

# Evaluate Final Best Model
rho_final = render_mmc(best_P, grid_x, grid_y, eps=0.04)
pred_bin = (rho_final > 0.5).float()
targ_bin = (rho_target > 0.5).float()
final_iou = (pred_bin * targ_bin).sum() / ((pred_bin + targ_bin).clamp(0, 1).sum() + 1e-8)

print(f"\nFinal Best MSE Loss: {best_loss:.5f} | Final Overlap IoU: {final_iou.item()*100:.2f}%")

# D. Plot and Save 4-Panel Comparison Figure
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
titles = [
    "1. Target Pixel Image (rho_lit)",
    "2. Step 1: Thin-Strut GMM Init",
    f"3. Step 3: Optimized MMC (IoU {final_iou.item()*100:.1f}%)",
    "4. Extracted Ellipsoid Vectors (P)"
]
images = [rho_target, rho_init, rho_final, rho_target]

for idx, ax in enumerate(axes):
    ax.imshow(images[idx].numpy(), cmap="gray_r", origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=1)
    ax.set_title(titles[idx], fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

P_final = best_P.numpy()
colors = ['#ff3b30', '#007aff', '#34c759', '#ff9500']
for k in range(K):
    xc, yc, a, b, th = P_final[k]
    ell = Ellipse((xc, yc), width=2*a, height=2*b, angle=np.degrees(th),
                  edgecolor=colors[k % len(colors)], facecolor='none', lw=2.5)
    axes[3].add_patch(ell)
    axes[3].plot(xc, yc, 'o', color=colors[k % len(colors)])

plt.tight_layout()
plt.savefig("mmc_inverse_result.png", dpi=200)
print("Saved updated visual proof to: mmc_inverse_result.png")
print("\nGround-Truth [K x 5] Table:")
print(np.round(P_true.numpy(), 3))
print("\nExtracted [K x 5] Ellipsoid Table:")
print(np.round(P_final, 3))
