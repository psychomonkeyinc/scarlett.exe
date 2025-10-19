# file: multimind_gan.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
import os
from typing import Tuple, Dict
import io
from PIL import Image

# Ensure full-precision default
torch.set_default_dtype(torch.float32)

# ---------------------------
# Config / Hyperparameters
# ---------------------------
IMG_SIZE = 128            # start small: 64 or 128 for iterating quickly
CHANNELS = 3
BATCH_SIZE = 32
Z_DIM = 128
LR_G = 2e-4
LR_D = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "./checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Scarlett router env var (used by adapter if available)
SCARLETT_ROUTER_URL = os.environ.get("SCARLETT_ROUTER_URL", "http://localhost:9000/router")

# ---------------------------
# Utility blocks
# ---------------------------
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# ---------------------------
# Generators
# ---------------------------
class GeneratorCoarse(nn.Module):
    """G1: generate a coarse layout from latent z"""
    def __init__(self, z_dim=Z_DIM, out_channels=CHANNELS, base=64):
        super().__init__()
        self.net = nn.Sequential(
            # project and reshape
            nn.Linear(z_dim, base * 8 * 4 * 4),
            nn.ReLU(True),
            View((-1, base * 8, 4, 4)),
            # upsample blocks
            ConvT_Block(base*8, base*4), # 8x -> 4x
            ConvT_Block(base*4, base*2), # 4x -> 2x
            ConvT_Block(base*2, base),   # 2x -> 1x
            nn.ConvTranspose2d(base, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class GeneratorRefiner(nn.Module):
    """G2: takes coarse output and refines (residual-style). Input is image-like"""
    def __init__(self, in_channels=CHANNELS, base=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, 1, 1),
            nn.ReLU(True),
            Conv_Block(base, base*2, downsample=True),
            Conv_Block(base*2, base*4, downsample=True)
        )
        self.middle = nn.Sequential(
            ResBlock(base*4),
            ResBlock(base*4)
        )
        self.decoder = nn.Sequential(
            ConvT_Block(base*4, base*2),
            ConvT_Block(base*2, base),
            nn.Conv2d(base, in_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, img):
        x = self.encoder(img)
        x = self.middle(x)
        refined = self.decoder(x)
        # Residual add (refines coarse)
        return torch.clamp(img + 0.2 * refined, -1.0, 1.0)

# small helper modules
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)


def ConvT_Block(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(True)
    )

def Conv_Block(in_ch, out_ch, downsample=False):
    if downsample:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch)
        )
    def forward(self, x):
        return F.relu(x + self.net(x))

# ---------------------------
# Discriminators (Adversaries)
# ---------------------------
class DiscriminatorSimple(nn.Module):
    """Simple patch-style discriminator; could be specialized per A1/A2 later"""
    def __init__(self, in_channels=CHANNELS, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base*2, 4, 2, 1),
            nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*2, base*4, 4, 2, 1),
            nn.BatchNorm2d(base*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*4, 1, 4, 1, 0),  # output single patch score
            View((-1, 1))
        )

    def forward(self, x):
        return self.net(x).view(-1)

# ---------------------------
# MetaMind Controller
# ---------------------------
class MetaMind(nn.Module):
    """
    Lightweight controller:
    - receives recent adversary losses and stability stats
    - outputs weights for each adversary and optional LR multipliers
    - exposes reporting hooks for Scarlett integration
    """
    def __init__(self, num_adversaries=2, hidden=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_adversaries * 2, hidden),  # e.g., [loss, variance] per adv
            nn.ReLU(True),
            nn.Linear(hidden, num_adversaries),
            nn.Softmax(dim=-1)  # produce normalized weights
        )
        # small LR control head
        self.lr_head = nn.Sequential(
            nn.Linear(num_adversaries * 2, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, 1),
            nn.Sigmoid()  # factor 0..1 -> multiply base LR
        )

    def forward(self, adv_stats: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        adv_stats: tensor shape (batch_or_step, num_adversaries*2)
            arranged as [loss1, var1, loss2, var2, ...] or simply summary stats per step
        returns: dict: { 'weights': tensor(num_adversaries), 'lr_factor': scalar }
        """
        weights = self.fc(adv_stats)
        lr_factor = self.lr_head(adv_stats).mean()  # simple scalar from stats
        return {"weights": weights, "lr_factor": lr_factor}

    # Hook: integrate with Scarlett's Global Conscience Router
    def report_to_scarlett(self, report: dict):
        """
        Attempt to send report via scarlett adapter if available. This avoids hard dependency on requests
        in training scripts while enabling the inference service to post to Scarlett.
        """
        try:
            # Importing here keeps multimind_gan usable without adding network deps in some environments
            from scarlett_adapter import send_report
            send_report(report)
        except Exception:
            # If adapter or network isn't available, silently no-op to avoid breaking training loops
            return

# ---------------------------
# Training utilities
# ---------------------------
def adversary_loss_fn(real_scores, fake_scores):
    # simple hinge loss / relativistic style could be used; here's BCELoss alternative
    # For stability, we use LSGAN (MSE) style
    real_label = torch.ones_like(real_scores)
    fake_label = torch.zeros_like(fake_scores)
    return F.mse_loss(real_scores, real_label) + F.mse_loss(fake_scores, fake_label)

def generator_loss_fn(fake_scores):
    real_label = torch.ones_like(fake_scores)
    return F.mse_loss(fake_scores, real_label)

# ---------------------------
# Dataset loader (example: ImageFolder or CelebA)
# ---------------------------
def get_dataloader(root_dir, img_size=IMG_SIZE, batch=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    ds = datasets.ImageFolder(root_dir, transform=transform)
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)

# ---------------------------
# Instantiate models
# ---------------------------
G1 = GeneratorCoarse().to(DEVICE)
G2 = GeneratorRefiner().to(DEVICE)
A1 = DiscriminatorSimple().to(DEVICE)  # realism judge
A2 = DiscriminatorSimple().to(DEVICE)  # coherence judge (could use different arch or pretrained perceptual)
M  = MetaMind(num_adversaries=2).to(DEVICE)

for m in [G1, G2, A1, A2, M]:
    m.apply(weights_init)

opt_G = torch.optim.Adam(list(G1.parameters()) + list(G2.parameters()), lr=LR_G, betas=(0.5, 0.999))
opt_A1 = torch.optim.Adam(A1.parameters(), lr=LR_D, betas=(0.5, 0.999))
opt_A2 = torch.optim.Adam(A2.parameters(), lr=LR_D, betas=(0.5, 0.999))

# ---------------------------
# Training loop skeleton
# ---------------------------
def train(data_loader, epochs=10):
    step = 0
    for epoch in range(epochs):
        for real_imgs, _ in data_loader:
            real_imgs = real_imgs.to(DEVICE)

            # === 1. Update Adversaries ===
            z = torch.randn(real_imgs.size(0), Z_DIM, device=DEVICE)
            coarse = G1(z)                       # G1 output (img-like)
            refined = G2(coarse)                 # G2 refines G1

            # A1/A2 on real and fake
            real_scores_A1 = A1(real_imgs)
            fake_scores_A1 = A1(refined.detach())

            real_scores_A2 = A2(real_imgs)
            fake_scores_A2 = A2(refined.detach())

            loss_A1 = adversary_loss_fn(real_scores_A1, fake_scores_A1)
            loss_A2 = adversary_loss_fn(real_scores_A2, fake_scores_A2)

            opt_A1.zero_grad(); loss_A1.backward(); opt_A1.step()
            opt_A2.zero_grad(); loss_A2.backward(); opt_A2.step()

            # === 2. MetaMind computes weights / LR factor ===
            # For simplicity we use loss scalars + simple variance proxies (here zeros)
            adv_stats = torch.tensor([loss_A1.item(), 0.0, loss_A2.item(), 0.0], device=DEVICE)
            adv_stats = adv_stats.unsqueeze(0)  # shape (1, 4)
            meta_out = M(adv_stats)
            weights = meta_out['weights'].squeeze(0)   # two weights
            lr_factor = meta_out['lr_factor'].item()

            # Optionally adapt generator LR (simple example)
            for gparam_group in opt_G.param_groups:
                gparam_group['lr'] = LR_G * (0.5 + lr_factor)  # keep it bounded

            # === 3. Update Generators ===
            # Compute generator losses from both adversaries and combine by metamind weights
            fake_scores_A1_forG = A1(refined)
            fake_scores_A2_forG = A2(refined)

            loss_G_A1 = generator_loss_fn(fake_scores_A1_forG)
            loss_G_A2 = generator_loss_fn(fake_scores_A2_forG)

            combined_gen_loss = weights[0] * loss_G_A1 + weights[1] * loss_G_A2

            opt_G.zero_grad()
            combined_gen_loss.backward()
            opt_G.step()

            # === 4. Optional: Metamind reporting and snapshot ===
            report = {
                "step": step,
                "losses": {"A1": loss_A1.item(), "A2": loss_A2.item(),
                           "G_A1": loss_G_A1.item(), "G_A2": loss_G_A2.item(),
                           "combined_G": combined_gen_loss.item()},
                "weights": weights.detach().cpu().numpy().tolist()
            }
            M.report_to_scarlett(report)  # hook to Scarlett router / logger

            if step % 200 == 0:
                print(f"Epoch {epoch} Step {step} | losses: {report['losses']} | weights: {report['weights']}")
                # save example outputs
                save_img = (refined[:16].detach().cpu() + 1) * 0.5
                utils.save_image(save_img, f"{CHECKPOINT_DIR}/sample_{epoch}_{step}.png", nrow=4)

            step += 1

        # checkpoint
        torch.save({
            "G1": G1.state_dict(),
            "G2": G2.state_dict(),
            "A1": A1.state_dict(),
            "A2": A2.state_dict(),
            "M": M.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_A1": opt_A1.state_dict(),
            "opt_A2": opt_A2.state_dict()
        }, f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch}.pt")

# ---------------------------
# Inference helpers for Scarlett integration
# ---------------------------

def _tensor_to_png_bytes(t: torch.Tensor) -> bytes:
    """Convert a single image tensor (C,H,W) in [-1,1] to PNG bytes."""
    t = t.detach().cpu()
    t = (t + 1) * 0.5  # 0..1
    t = t.clamp(0, 1)
    arr = (t.numpy() * 255).astype('uint8')
    # C,H,W -> H,W,C
    arr = arr.transpose(1, 2, 0)
    if arr.shape[2] == 1:
        mode = 'L'
        img = Image.fromarray(arr[:, :, 0], mode)
    else:
        mode = 'RGB'
        img = Image.fromarray(arr, mode)
    bio = io.BytesIO()
    img.save(bio, format='PNG')
    return bio.getvalue()


def generate(num_samples: int = 1, z: torch.Tensor = None, device: torch.device = DEVICE):
    """Generate num_samples images from G1->G2 and return list of PNG byte blobs.
    Models are expected to be instantiated in this module (G1,G2)."""
    G1.eval(); G2.eval()
    with torch.no_grad():
        if z is None:
            z = torch.randn(num_samples, Z_DIM, device=device)
        else:
            z = z.to(device)
        coarse = G1(z)
        refined = G2(coarse)
        imgs = []
        for i in range(refined.size(0)):
            imgs.append(_tensor_to_png_bytes(refined[i]))
        return imgs


def load_checkpoint(path: str):
    """Load model weights from a checkpoint saved by train. Non-fatal if not found."""
    try:
        data = torch.load(path, map_location=DEVICE)
        if 'G1' in data: G1.load_state_dict(data['G1'])
        if 'G2' in data: G2.load_state_dict(data['G2'])
        if 'A1' in data: A1.load_state_dict(data['A1'])
        if 'A2' in data: A2.load_state_dict(data['A2'])
        if 'M' in data: M.load_state_dict(data['M'])
        return True
    except Exception:
        return False

# ---------------------------
# If executed as script
# ---------------------------
if __name__ == "__main__":
    # example: point to a folder with subfolders of images (ImageFolder structure)
    data_path = "/path/to/your/images"  # change this to your dataset path
    loader = get_dataloader(data_path, IMG_SIZE, BATCH_SIZE)
    train(loader, epochs=5)