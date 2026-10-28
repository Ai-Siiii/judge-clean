"""JudgeClean: one tiny shared encoder, two heads.
Input: a small crop (64x64) that may contain text.
Head 1 (decision): does this crop contain text? 0/1.
Head 2 (clean): reconstruct the background with the text removed.
Training data is fully synthetic: random backgrounds (solid, gradient,
screentone dots, stripes) + PIL-drawn fake text. No manga pages, no
copyrighted art. The point is the weight trick, not SOTA cleaning:
one encoder serves both a decision and a pixel output, so the model
decides AND cleans in a single forward pass.
"""
import math, os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

random.seed(0); np.random.seed(0); torch.manual_seed(0)
torch.set_num_threads(2)
DEV = "cpu"; S = 64

def make_bg(rng):
    kind = rng.choice(["solid", "gradient", "dots", "stripes"])
    c = np.array([rng.randint(180, 256) for _ in range(3)], dtype=np.float32)
    img = np.ones((S, S, 3), dtype=np.float32) * c
    if kind == "gradient":
        g = np.linspace(0, 60, S)[:, None, None] * rng.choice([-1, 1])
        img = np.clip(img + g, 0, 255)
    elif kind == "dots":
        for y in range(0, S, 6):
            for x in range(0, S, 6):
                if rng.random() < 0.4:
                    img[y:y+2, x:x+2] = np.clip(c - 40, 0, 255)
    elif kind == "stripes":
        for x in range(0, S, 5):
            if rng.random() < 0.5:
                img[:, x:x+2] = np.clip(c - 30, 0, 255)
    return img.astype(np.float32)

WORDS = ["HELLO", "TEXT", "MANGA", "CLEAN", "JUDGE", "ABC", "XYZ", "NOISE"]
def make_sample(rng):
    bg = make_bg(rng)
    has_text = rng.random() < 0.5
    dirty = bg.copy()
    if has_text:
        im = Image.fromarray(dirty.astype(np.uint8))
        d = ImageDraw.Draw(im)
        for _ in range(rng.randint(1, 3)):
            w = rng.choice(WORDS)
            x, y = rng.randint(0, 30), rng.randint(0, 40)
            d.text((x, y), w, fill=(0, 0, 0))
        dirty = np.array(im).astype(np.float32)
    x = torch.tensor(dirty.transpose(2, 0, 1) / 255.0)
    y = torch.tensor(bg.transpose(2, 0, 1) / 255.0)
    return x, y, float(has_text)

class JudgeClean(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 16
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 8
        )
        self.decide = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1))
        self.clean = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.ReLU(), # 16
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(), # 32
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(), # 64
        )
    def forward(self, x):
        f = self.enc(x)
        return self.decide(f).squeeze(1), self.clean(f)

def batch(rng, n):
    xs, ys, ds = [], [], []
    for _ in range(n):
        x, y, d = make_sample(rng)
        xs.append(x); ys.append(y); ds.append(d)
    return torch.stack(xs), torch.stack(ys), torch.tensor(ds)

def main():
    rng = random.Random(1234)
    model = JudgeClean()
    nparam = sum(p.numel() for p in model.parameters())
    print(f"params: {nparam}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(12):
        model.train()
        tot_l = tot_a = tot_n = 0
        for _ in range(40):
            x, y, d = batch(rng, 32)
            pred, clean = model(x)
            loss = F.binary_cross_entropy_with_logits(pred, d) + F.l1_loss(clean, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_l += loss.item(); tot_a += ((pred > 0).float() == d).float().mean().item(); tot_n += 1
        print(f"epoch {epoch}: loss {tot_l/tot_n:.4f} decision_acc {tot_a/tot_n:.3f}", flush=True)
    # held-out test
    model.eval()
    rng2 = random.Random(999)
    with torch.no_grad():
        x, y, d = batch(rng2, 200)
        pred, clean = model(x)
        acc = ((pred > 0).float() == d).float().mean().item()
        psnr = -10 * torch.log10(F.mse_loss(clean, y) + 1e-9).item()
    print(f"TEST decision_acc {acc:.3f} clean_psnr {psnr:.2f} dB")
    os.makedirs("out", exist_ok=True)
    torch.save(model.state_dict(), "out/judgeclean-64.pt")
    # save a demo pair as png for eyeballing
    with torch.no_grad():
        x, y, d = batch(random.Random(7), 4)
        pred, clean = model(x)
        demo = torch.cat([x, clean], dim=3)  # dirty | cleaned, 4 rows
        grid = demo.permute(0, 2, 3, 1).reshape(4 * S, 2 * S, 3).numpy()
        Image.fromarray((np.clip(grid, 0, 1) * 255).astype(np.uint8)).save("out/demo.png")
        print("demo decisions:", [round(float(v), 2) for v in pred], "true:", [int(v) for v in d])
    print("saved out/judgeclean-64.pt")

if __name__ == "__main__":
    main()
