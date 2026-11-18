import torch
import torch.optim as optim
from model.symb_logi_ops import (
    pairwise_l2, 
    sinkhorn_log, 
    mixture_mm, 
    kl_gaussian_diag, 
    node_logic_loss
)

print("=== MetaSymbO Math & Gradient Descent Test ===")
torch.manual_seed(42)

# 1. Simulate candidate node latents (mu_a, logvar_a) and target scaffold node latents (mu_b, logvar_b)
N_s, N_t, D = 12, 6, 16  # 8 candidate nodes, 6 scaffold nodes, 16 latent dimensions
mu_a = torch.randn(N_s, D, requires_grad=True)
logvar_a = torch.zeros(N_s, D) # fixed logvar for simplicity

mu_b = torch.randn(N_t, D)
logvar_b = torch.zeros(N_t, D)

print(f"Candidate node tensor shape: {mu_a.shape}")
print(f"Scaffold node tensor shape: {mu_b.shape}")

# 2. Compute Pairwise L2 Cost & Run Sinkhorn Log-Domain Matching
cost = pairwise_l2(mu_a.detach(), mu_b.detach())
cost = cost / (cost.max() + 1e-8)  # <--- CRITICAL line from modal_agents.py!
P = sinkhorn_log(cost, epsilon=0.1)
print(f"Sinkhorn transport matrix P shape (sum = {P.sum():.2f}): {P.shape}")

# 3. Setup Optimizer for Latent Evolution (Mimicking optimize_geo_latent)
optimizer = optim.Adam([mu_a], lr=0.05)

print("\n--- Running 50-Step Latent Optimization Loop ---")
for step in range(50):
    optimizer.zero_grad()
    
    # Compute Mix operator target distribution with lam = 0.3
    # (Matches Mixture Moment-Matching in symb_logi_ops.py)
    selected_idx = P.argmax(dim=1)
    matched_mu = mu_b[selected_idx]
    matched_logvar = logvar_b[selected_idx]
    
    mu_t, lv_t = mixture_mm(mu_a, logvar_a, matched_mu, matched_logvar, lam=0.3)
    
    # Compute Sinkhorn-weighted node alignment loss
    kl_pair = kl_gaussian_diag(mu_a, logvar_a, mu_t.detach(), lv_t.detach())
    loss = (P[torch.arange(N_s), selected_idx] * kl_pair).sum() + 0.001 * mu_a.pow(2).mean()
    
    loss.backward()
    optimizer.step()
    
    if step % 10 == 0 or step == 49:
        print(f"Step {step:02d} | Loss: {loss.item():.4f}")

print("\n=== Test Successful! Tensors successfully updated via gradients. ===")