import torch
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Discretized 2D domain
# =========================
n = 60
x1 = torch.linspace(0, 1, n)
x2 = torch.linspace(0, 1, n)
X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
X = torch.stack([X1.flatten(), X2.flatten()], dim=1)  # (n^2, 2)

# =========================
# 2. Ground truth (only for generating samples)
# =========================
def f(x):
    return torch.sin(2*np.pi*x[:,0]) * torch.cos(2*np.pi*x[:,1])

y_true = f(X)

# =========================
# 3. Construct envelope ℓ, u
# =========================
# spatially varying width
width = 0.15 + 0.35 * torch.exp(
    -6*((X[:,0]-0.5)**2 + (X[:,1]-0.5)**2)
)

# NOTE: this "center" is only to build an example
center = y_true

l = center - width
u = center + width

# =========================
# 4. Generate samples
# =========================
num_samples = 30
idx = torch.randperm(X.shape[0])[:num_samples]
X_sample = X[idx]
y_sample = f(X_sample) + 0.05*torch.randn(num_samples)

# =========================
# 5. Snap samples to grid (important)
# =========================
dist = torch.norm(X.unsqueeze(1) - X_sample.unsqueeze(0), dim=2)
nearest_idx = torch.argmin(dist, dim=0)

# enforce interpolation: ℓ = u = y at samples
l[nearest_idx] = y_sample
u[nearest_idx] = y_sample

# =========================
# 6. Reshape for plotting
# =========================
L = l.reshape(n, n)
U = u.reshape(n, n)
W = (U - L)

# =========================
# 7. 3D Plot
# =========================
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

# --- upper surface (colored by width) ---
surf = ax.plot_surface(
    X1.numpy(), X2.numpy(), U.numpy(),
    facecolors=plt.cm.viridis((W / W.max()).numpy()),
    rstride=1, cstride=1,
    linewidth=0,
    antialiased=True,
    alpha=0.8
)

# --- lower surface ---
ax.plot_surface(
    X1.numpy(), X2.numpy(), L.numpy(),
    color='blue',
    alpha=0.25
)

# --- sample points ---
ax.scatter(
    X_sample[:,0].numpy(),
    X_sample[:,1].numpy(),
    y_sample.numpy(),
    color='red',
    s=60,
    label="samples"
)

# --- vertical lines (tube visualization) ---
for i in nearest_idx:
    xi = X[i]
    ax.plot(
        [xi[0], xi[0]],
        [xi[1], xi[1]],
        [l[i], u[i]],
        color='black',
        linewidth=1
    )

# =========================
# 8. Styling
# =========================
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("value")

ax.view_init(elev=30, azim=135)

plt.tight_layout()
plt.show()



fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# normalize width
Wn = (W / W.max()).numpy()

# --- upper surface (colored by uncertainty) ---
ax.plot_surface(
    X1.numpy(), X2.numpy(), U.numpy(),
    facecolors=plt.cm.viridis(Wn),
    linewidth=0,
    antialiased=True,
    alpha=0.9
)

# --- lower surface (very light) ---
ax.plot_surface(
    X1.numpy(), X2.numpy(), L.numpy(),
    color='gray',
    alpha=0.15
)

# --- samples ---
ax.scatter(
    X_sample[:,0].numpy(),
    X_sample[:,1].numpy(),
    y_sample.numpy(),
    color='red',
    s=60
)

# colorbar
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(W.numpy())
fig.colorbar(mappable, ax=ax, shrink=0.6, label='u(x)-l(x)')

ax.view_init(elev=30, azim=135)
plt.tight_layout()
plt.show()



import torch
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Discretized 2D domain
# =========================
n = 60
x1 = torch.linspace(0, 1, n)
x2 = torch.linspace(0, 1, n)
X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
X = torch.stack([X1.flatten(), X2.flatten()], dim=1)  # (n^2, 2)

# =========================
# 2. Ground truth (only to create samples)
# =========================
def f(x):
    return torch.sin(2*np.pi*x[:,0]) * torch.cos(2*np.pi*x[:,1])

# =========================
# 3. Construct ℓ and u
# =========================
width = 0.15 + 0.35 * torch.exp(
    -6*((X[:,0]-0.5)**2 + (X[:,1]-0.5)**2)
)

center = f(X)  # only for synthetic example
l = center - width
u = center + width

# =========================
# 4. Generate samples
# =========================
num_samples = 30
idx = torch.randperm(X.shape[0])[:num_samples]
X_sample = X[idx]
y_sample = f(X_sample) + 0.05 * torch.randn(num_samples)

# =========================
# 5. Snap samples to grid
# =========================
dist = torch.norm(X.unsqueeze(1) - X_sample.unsqueeze(0), dim=2)
nearest_idx = torch.argmin(dist, dim=0)

# enforce interpolation: ℓ = u at samples
l[nearest_idx] = y_sample
u[nearest_idx] = y_sample

# =========================
# 6. Compute uncertainty
# =========================
W = u - l  # this is what we plot

# reshape for surface
W_grid = W.reshape(n, n)

# =========================
# 7. 3D plot (uncertainty only)
# =========================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

Wn = (W_grid / W_grid.max()).numpy()

# surface
ax.plot_surface(
    X1.numpy(), X2.numpy(), W_grid.numpy(),
    facecolors=plt.cm.viridis(Wn),
    linewidth=0,
    antialiased=True,
    alpha=0.95
)

# samples (should lie at zero uncertainty)
ax.scatter(
    X[nearest_idx,0].numpy(),
    X[nearest_idx,1].numpy(),
    W[nearest_idx].numpy(),
    color='red',
    s=70
)

# colorbar
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(W_grid.numpy())
fig.colorbar(mappable, ax=ax, shrink=0.6, label='u(x)-l(x)')

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("uncertainty width")

ax.view_init(elev=30, azim=135)

plt.tight_layout()
plt.show()