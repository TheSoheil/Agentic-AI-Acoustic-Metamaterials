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

def render_mmc(P, gx, gy, eps=0.08, p_norm=2.0):
    """
    Forward MMC Renderer: Maps [K, 5] ellipse table P -> [100, 100] density image in [0, 1].
    Columns of P: [x_c, y_c, a, b, theta (radians)]
    """
    xc = P[:, 0].view(-1, 1, 1)
    yc = P[:, 1].view(-1, 1, 1)
    a  = torch.clamp(P[:, 2].view(-1, 1, 1), min=0.02)
    b  = torch.clamp(P[:, 3].view(-1, 1, 1), min=0.02)
    th = P[:, 4].view(-1, 1, 1)

    # Shift and rotate pixel coordinates into each ellipse's local frame
    dx = gx.unsqueeze(0) - xc
    dy = gy.unsqueeze(0) - yc
    cos_t, sin_t = torch.cos(th), torch.sin(th)
    x_rot =  dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t

    # MMC Topology Description Function: phi_i(x, y)
    phi_i = 1.0 - ((x_rot / a) ** p_norm + (y_rot / b) ** p_norm)

    # Smooth maximum across all K ellipses + regularized Heaviside (sigmoid)
    alpha = 20.0
    phi_max = (1.0 / alpha) * torch.logsumexp(alpha * phi_i, dim=0)
    rho_hat = torch.sigmoid(phi_max / eps)
    return rho_hat

def gmm_initialize(rho_img, K=4):
    """Step 1: Convert solid pixels (rho > 0.5) into K initial ellipses via GMM."""
    pts_y, pts_x = torch.where(rho_img > 0.5)
    coords = torch.stack([pts_x.float() / (RES - 1), pts_y.float() / (RES - 1)], dim=1).numpy()

    gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=42)
    gmm.fit(coords)

    init_list = []
    for k in range(K):
        xc, yc = gmm.means_[k]
        cov = gmm.covariances_[k]
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort largest eigenvalue first
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        
        a = 2.0 * np.sqrt(max(eigvals[0], 1e-4))
        b = 2.0 * np.sqrt(max(eigvals[1], 1e-4))
        theta = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        init_list.append([xc, yc, a, b, theta])

    return torch.tensor(init_list, dtype=torch.float32)

# ================= MAIN TEST =================
print("=== Forward & Inverse MMC Ellipsoid Test ===")

# A. Create Ground-Truth Target Image (4 intersecting ellipses: X-truss + center bars)
P_true = torch.tensor([
    [0.50, 0.50, 0.42, 0.08,  0.785],  # +45 deg diagonal strut
    [0.50, 0.50, 0.42, 0.08, -0.785],  # -45 deg diagonal strut
    [0.50, 0.22, 0.28, 0.07,  0.000],  # Bottom horizontal bar
    [0.50, 0.78, 0.28, 0.07,  0.000],  # Top horizontal bar
], dtype=torch.float32)

with torch.no_grad():
    rho_target = render_mmc(P_true, grid_x, grid_y, eps=0.04)

# B. Step 1: GMM Smart Initialization from pixels ONLY
K = 4
P_init = gmm_initialize(rho_target, K=K)
with torch.no_grad():
    rho_init = render_mmc(P_init, grid_x, grid_y)

# C. Step 2 & 3: Inverse Gradient Refinement (200 steps of Adam)
P_opt = P_init.clone().detach().requires_grad_(True)
optimizer = torch.optim.Adam([P_opt], lr=0.02)

print("\n--- Running 200-Step Inverse MMC Optimization ---")
for step in range(200):
    optimizer.zero_grad()
    rho_hat = render_mmc(P_opt, grid_x, grid_y, eps=0.06)
    loss = torch.mean((rho_hat - rho_target) ** 2)
    loss.backward()
    optimizer.step()

    if step % 40 == 0 or step == 199:
        # Compute Intersection over Union (IoU) accuracy
        pred_bin = (rho_hat.detach() > 0.5).float()
        targ_bin = (rho_target > 0.5).float()
        iou = (pred_bin * targ_bin).sum() / ((pred_bin + targ_bin).clamp(0, 1).sum() + 1e-8)
        print(f"Step {step:03d} | Pixel MSE Loss: {loss.item():.5f} | Overlap IoU: {iou.item()*100:.1f}%")

# D. Plot and Save 4-Panel Comparison Figure
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
titles = [
    "1. Target Pixel Image (rho_lit)",
    "2. Step 1: GMM Initialization",
    "3. Step 3: Optimized MMC (rho_hat)",
    "4. Extracted Ellipsoid Vectors (P)"
]
images = [rho_target, rho_init, rho_hat.detach(), rho_target]

for idx, ax in enumerate(axes):
    ax.imshow(images[idx].numpy(), cmap="gray_r", origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=1)
    ax.set_title(titles[idx], fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

# Overlay actual geometric ellipse boundaries on Panel 4
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
print("\nSaved visual proof to: mmc_inverse_result.png")
print("\nExtracted [K x 5] Ellipsoid Table (x_c, y_c, a, b, theta_rad):")
print(np.round(P_final, 3))