import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
import random

from matplotlib.patches import Arc

# === Config ===
num_points = 50
batch_size = 512
val_batch_size = 256
num_steps = 100000
val_interval = 1000
cone_angle_deg = 10
save_vis_interval = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Utils ===
def deg2rad(deg):
    return deg * math.pi / 180.0


def angle_between(v1, v2):
    v1 = F.normalize(v1, dim=-1)
    v2 = F.normalize(v2, dim=-1)
    return torch.acos((v1 * v2).sum(-1).clamp(-1.0, 1.0))


# === Data Generation ===
def generate_batch(batch_size, num_points):
    radar_xy = torch.rand(batch_size, 2, device=device)
    angles = torch.rand(batch_size, device=device) * 2 * math.pi
    radar_dir = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    pts = torch.rand(batch_size, num_points, 2, device=device) * 2.5 - 0.5

    vecs = pts - radar_xy[:, None, :]
    dists = vecs.norm(dim=-1)
    angles_to_pts = angle_between(vecs, radar_dir[:, None, :])

    in_cone = angles_to_pts < deg2rad(cone_angle_deg)
    masked_dists = dists.clone()
    masked_dists[~in_cone] = 1

    closest_dist = masked_dists.min(dim=1).values
    return radar_xy, radar_dir, pts, closest_dist


# === Fully Connected Model ===
class FCModel(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 + 2 + num_points * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, radar_xy, radar_dir, pts):
        x = torch.cat([radar_xy, radar_dir, pts.view(pts.shape[0], -1)], dim=1)
        return self.fc(x).squeeze(-1)


class MultiHeadMLPAttentionModel(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Shared point encoder: maps [radar + point] → feature
        self.point_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # One attention scoring MLP per head (each outputs (B, N, 1))
        self.attn_score_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(6, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_heads)
        ])

        # Final MLP after concatenating head outputs
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.last_attn_weights = None  # shape: (B, num_heads, N)

    def forward(self, radar_xy, radar_dir, pts):
        B, N, _ = pts.shape

        # Expand radar to per-point shape: (B, N, 4)
        radar = torch.cat([radar_xy, radar_dir], dim=-1).unsqueeze(1).expand(-1, N, -1)
        pairwise = torch.cat([radar, pts], dim=-1)  # (B, N, 6)

        # Shared feature embedding for points
        point_features = self.point_encoder(pairwise)  # (B, N, hidden_dim)

        all_contexts = []
        all_weights = []

        for i in range(self.num_heads):
            logits = self.attn_score_nets[i](pairwise)  # (B, N, 1)
            weights = torch.softmax(logits, dim=1)      # (B, N, 1)
            context = (point_features * weights).sum(dim=1)  # (B, hidden_dim)

            all_contexts.append(context)
            all_weights.append(weights.squeeze(-1))  # (B, N)

        # (B, num_heads, N)
        self.last_attn_weights = torch.stack(all_weights, dim=1)

        # (B, hidden_dim * num_heads)
        full_context = torch.cat(all_contexts, dim=1)

        return self.output_mlp(full_context).squeeze(-1)  # (B,)



# === Training ===
def train(model, optimizer, name, val_data):
    model.to(device)
    model.train()
    val_losses = []
    val_xy, val_dir, val_pts, val_target = val_data

    for step in range(1, num_steps + 1):
        radar_xy, radar_dir, pts, target = generate_batch(batch_size, num_points)

        pred = model(radar_xy, radar_dir, pts)
        loss = F.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log validation loss occasionally (every val_interval)
        if step % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_xy, val_dir, val_pts)
                val_loss = F.mse_loss(val_pred, val_target).item()
                val_losses.append((step, val_loss))
                print(f"[{name}] Step {step}, Val Loss: {val_loss:.4f}")
            model.train()

    # Final validation loss plot
    plot_loss(val_losses, name)

    # Visualize 20 random samples after training
    print(f"[{name}] Generating final visualizations...")
    model.eval()
    visualize_sample(model, name=name, step="final", val_data=val_data)
    print(f"[{name}] Visualization complete.")



# === Visualization ===
def visualize_sample(model, name, step, val_data):
    model.eval()
    with torch.no_grad():
        radar_xy, radar_dir, pts, target = val_data
        pred = model(radar_xy, radar_dir, pts)

        B = min(20, radar_xy.shape[0])

        for i in range(B):
            radar_xy_np = radar_xy[i].cpu().numpy()
            radar_dir_np = radar_dir[i].cpu().numpy()
            pts_np = pts[i].cpu().numpy()
            target_val = target[i].item()
            pred_val = pred[i].item()

            attn_weights = None
            num_heads = 1

            if hasattr(model, "last_attn_weights") and model.last_attn_weights is not None:
                attn = model.last_attn_weights[i].cpu().numpy()
                if attn.ndim == 2:  # (num_heads, N)
                    attn_weights = attn
                    num_heads = attn.shape[0]
                elif attn.ndim == 1:  # (N,)
                    attn_weights = attn[None, :]  # (1, N)

            fig, axes = plt.subplots(1, num_heads, figsize=(5 * num_heads, 5), squeeze=False)

            for h in range(num_heads):
                ax = axes[0, h]
                ax.scatter(pts_np[:, 0], pts_np[:, 1],
                           c=attn_weights[h] if attn_weights is not None else "gray",
                           cmap='viridis', s=60, label="Points")
                ax.plot(radar_xy_np[0], radar_xy_np[1], 'ro', label="Radar")

                # Draw cone lines
                for angle_offset in [-cone_angle_deg, cone_angle_deg]:
                    theta = math.atan2(radar_dir_np[1], radar_dir_np[0]) + deg2rad(angle_offset)
                    end = radar_xy_np + np.array([math.cos(theta), math.sin(theta)])
                    ax.plot([radar_xy_np[0], end[0]], [radar_xy_np[1], end[1]], 'r--')

                # Draw arcs for target and predicted distances
                angle_center = math.atan2(radar_dir_np[1], radar_dir_np[0]) * 180 / math.pi
                arc_extent = 2 * cone_angle_deg

                for radius, color, label in [
                    (target_val, 'blue', "Target"),
                    (pred_val, 'orange', "Prediction")
                ]:
                    arc = Arc((radar_xy_np[0], radar_xy_np[1]), width=2 * radius, height=2 * radius,
                              angle=0, theta1=angle_center - cone_angle_deg, theta2=angle_center + cone_angle_deg,
                              color=color, lw=2, label=label)
                    ax.add_patch(arc)

                ax.set_title(f"Head {h} | Target: {target_val:.2f}, Pred: {pred_val:.2f}")
                ax.legend()
                ax.axis('equal')
                ax.grid(True)

                if attn_weights is not None:
                    sm = plt.cm.ScalarMappable(cmap='viridis')
                    sm.set_array(attn_weights[h])
                    fig.colorbar(sm, ax=ax, label='Attention Weight')

            plt.tight_layout()
            plt.savefig(f"{name}_step{step}_{i}.png")
            plt.close()



# === Plotting Loss ===
def plot_loss(losses, name):
    steps, vals = zip(*losses)
    plt.figure()
    plt.plot(steps, vals)
    plt.xlabel("Step")
    plt.ylabel("Validation Loss")
    plt.title(f"Validation Loss - {name}")
    plt.grid(True)
    plt.savefig(f"{name}_val_loss.png")
    plt.close()

val_data =  generate_batch(val_batch_size, num_points)

# === Run Experiments ===
fc_model = FCModel(num_points)
fc_optim = torch.optim.Adam(fc_model.parameters(), lr=1e-3)
train(fc_model, fc_optim, "fc", val_data)

attn_model = MultiHeadMLPAttentionModel()
attn_optim = torch.optim.Adam(attn_model.parameters(), lr=1e-3)
train(attn_model, attn_optim, "attention", val_data)
