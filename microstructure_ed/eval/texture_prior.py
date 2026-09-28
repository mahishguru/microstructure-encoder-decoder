#!/usr/bin/env python
"""z -> ODF-histogram prior for GT-free ODF calibration ("self-calibB3").

Phase B's calibB3 needs the ORIGINAL image's grain palette at inference,
which (a) leaks information around the z bottleneck and (b) is unavailable
for inverse design. This module makes the calibration target a function of
z alone:

  1. build   : global K-center codebook over stereographic S-space pixels
               (torch k-means), then per-image z (frozen encoder) + pixel
               histogram over the codebook, for dataset_train + dataset_test.
  2. train   : MLP head z -> histogram (soft cross-entropy), early-stopped
               on the test split; reports the per-class mean-histogram
               baseline (the head must beat class collapse to be useful).
  3. predict : ID=path pairs -> predicted histograms npz consumed by
               microstructure_ed/eval/odf_calibrate.py --pred_hist.

Everything runs on one GPU and is independent of the diffusion decoder, so
the head can be retrained in minutes for any encoder checkpoint.
"""
from __future__ import annotations
import argparse, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(os.environ.get("MSED_ROOT", os.getcwd())).resolve()

from microstructure_ed.eval.run_reconstruction import load_variant_modules  # noqa: E402
from microstructure_ed.encoder_arch_pretrained import Compressor            # noqa: E402
from torchvision import transforms                                # noqa: E402

TF = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
VARIANT = os.environ.get("TEXPRIOR_VARIANT", "vitfmdit_1280")
PRIOR_DIR = Path(os.environ.get("TEXPRIOR_DIR",
                                str(REPO_ROOT / "eval_outputs" / "texture_prior")))


Z_DIM = load_variant_modules(VARIANT)[2]


def build_encoder(device, ckpt_path):
    _, _, target_dim, _, _, _, spatial_tokens = load_variant_modules(VARIANT)
    enc = Compressor(use_gradient_checkpointing=False, trainable_blocks=0,
                     target_dim=target_dim, spatial_tokens=spatial_tokens).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()
    return enc


def load_S(path):
    """Native-resolution pixels as stereographic S in [-1,1], (N,3) float32."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return (arr.reshape(-1, 3) / 255.0) * 2.0 - 1.0


def class_of(stem: str) -> str:
    return stem.split("_orientation_")[0]


# -- k-means codebook ---------------------------------------------------------

def kmeans_gpu(pts, K, iters, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    centers = pts[torch.randperm(pts.shape[0], generator=g)[:K]].to(device)
    pts = pts.to(device)
    for it in range(iters):
        assign = torch.cdist(pts, centers).argmin(1)
        new = torch.zeros_like(centers)
        cnt = torch.zeros(K, device=device)
        new.index_add_(0, assign, pts)
        cnt.index_add_(0, assign, torch.ones_like(assign, dtype=torch.float))
        dead = cnt == 0
        new[~dead] /= cnt[~dead].unsqueeze(1)
        if dead.any():  # re-seed dead centers from random points
            ridx = torch.randint(pts.shape[0], (int(dead.sum()),), device=device)
            new[dead] = pts[ridx]
        shift = (new - centers).norm(dim=1).max().item()
        centers = new
        if shift < 1e-4:
            break
    return centers.cpu()


def cmd_build(args):
    device = torch.device(args.device)
    PRIOR_DIR.mkdir(parents=True, exist_ok=True)
    train_dir = REPO_ROOT / "dataset_train"
    test_dir = REPO_ROOT / "dataset_test"
    train_files = sorted(train_dir.glob("*.png"))
    test_files = sorted(test_dir.glob("*.png"))
    print(f"[build] train={len(train_files)} test={len(test_files)}", flush=True)

    # 1) codebook: balanced pixel subsample per class from the train split
    rng = np.random.default_rng(0)
    by_class = {}
    for f in train_files:
        by_class.setdefault(class_of(f.stem), []).append(f)
    samples = []
    for c, files in sorted(by_class.items()):
        pick = rng.choice(len(files), size=min(args.imgs_per_class, len(files)),
                          replace=False)
        for i in pick:
            S = load_S(files[i])
            idx = rng.choice(S.shape[0], size=args.px_per_img, replace=False)
            samples.append(S[idx])
    pts = torch.from_numpy(np.concatenate(samples)).float()
    print(f"[build] kmeans on {pts.shape[0]} pts -> K={args.K}", flush=True)
    t0 = time.time()
    centers = kmeans_gpu(pts, args.K, args.kmeans_iters, device)
    np.save(PRIOR_DIR / "codebook.npy", centers.numpy())
    print(f"[build] codebook saved ({time.time()-t0:.0f}s)", flush=True)

    # 2) encoder z + histogram per image
    enc = build_encoder(device, args.ckpt)
    centers_d = centers.to(device)

    def one_split(files, tag):
        N = len(files)
        zs = np.zeros((N, Z_DIM), dtype=np.float32)
        hs = np.zeros((N, args.K), dtype=np.float16)
        names = np.array([f.stem for f in files])
        pool = ThreadPoolExecutor(8)

        def load_pair(f):
            img = Image.open(f).convert("RGB")
            x = TF(img)
            S = (np.asarray(img, dtype=np.float32).reshape(-1, 3) / 255.0) * 2 - 1
            return x, S

        B = args.batch
        t0 = time.time()
        for s in range(0, N, B):
            batch = list(pool.map(load_pair, files[s:s + B]))
            xs = torch.stack([b[0] for b in batch]).to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z = enc(xs)
            zs[s:s + B] = z.float().cpu().numpy()
            for j, (_, S) in enumerate(batch):
                Sd = torch.from_numpy(S).to(device)
                a = torch.cdist(Sd, centers_d).argmin(1)
                h = torch.bincount(a, minlength=args.K).float()
                hs[s + j] = (h / h.sum()).cpu().numpy().astype(np.float16)
            if s % (B * 40) == 0:
                el = time.time() - t0
                print(f"[build] {tag} {s}/{N} ({el:.0f}s)", flush=True)
        np.save(PRIOR_DIR / f"z_{tag}.npy", zs)
        np.save(PRIOR_DIR / f"hist_{tag}.npy", hs)
        np.save(PRIOR_DIR / f"names_{tag}.npy", names)

    one_split(train_files, "train")
    one_split(test_files, "test")
    print("[build] done", flush=True)


# -- head training ------------------------------------------------------------

class HistHead(torch.nn.Module):
    def __init__(self, in_dim=1280, K=1024, hidden=1024, p=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, hidden), torch.nn.GELU(), torch.nn.Dropout(p),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, K),
        )

    def forward(self, z):
        return self.net(z)


def soft_ce(logits, target):
    return -(target * torch.log_softmax(logits, dim=-1)).sum(-1).mean()


def cmd_train(args):
    device = torch.device(args.device)
    z_tr = torch.from_numpy(np.load(PRIOR_DIR / "z_train.npy"))
    h_tr = torch.from_numpy(np.load(PRIOR_DIR / "hist_train.npy")).float()
    z_te = torch.from_numpy(np.load(PRIOR_DIR / "z_test.npy")).to(device)
    h_te = torch.from_numpy(np.load(PRIOR_DIR / "hist_test.npy")).float().to(device)
    names_tr = np.load(PRIOR_DIR / "names_train.npy")
    names_te = np.load(PRIOR_DIR / "names_test.npy")
    K = h_tr.shape[1]

    # class-mean baseline (the bar the head must clear)
    cls_tr = np.array([class_of(n) for n in names_tr])
    cls_te = np.array([class_of(n) for n in names_te])
    means = {c: h_tr[cls_tr == c].mean(0) for c in np.unique(cls_tr)}
    base = torch.stack([means[c] for c in cls_te]).to(device)
    eps = 1e-9
    ce_base = -(h_te * (base + eps).log()).sum(-1).mean().item()
    l1_base = (h_te - base).abs().sum(-1).mean().item()
    ent = -(h_te * (h_te + eps).log()).sum(-1).mean().item()
    print(f"[train] test entropy={ent:.4f}  baseline: CE={ce_base:.4f} L1={l1_base:.4f}",
          flush=True)

    head = HistHead(z_tr.shape[1], K).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    N = z_tr.shape[0]
    best = (1e9, -1)
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(N)
        tot = 0.0
        for s in range(0, N, args.batch):
            idx = perm[s:s + args.batch]
            z = z_tr[idx].to(device)
            h = h_tr[idx].to(device)
            loss = soft_ce(head(z), h)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        head.eval()
        with torch.no_grad():
            ce_te = soft_ce(head(z_te), h_te).item()
            l1_te = (torch.softmax(head(z_te), -1) - h_te).abs().sum(-1).mean().item()
        if ce_te < best[0]:
            best = (ce_te, ep)
            torch.save({"state": head.state_dict(), "K": K,
                        "in_dim": z_tr.shape[1]}, PRIOR_DIR / "head.pt")
        print(f"[train] ep{ep:02d} train_CE={tot/N:.4f} test_CE={ce_te:.4f} "
              f"test_L1={l1_te:.4f} (baseline CE {ce_base:.4f} L1 {l1_base:.4f})",
              flush=True)
    print(f"[train] best test_CE={best[0]:.4f} @ep{best[1]} -> head.pt", flush=True)


# -- prediction for calibration -----------------------------------------------

def cmd_predict(args):
    device = torch.device(args.device)
    enc = build_encoder(device, args.ckpt)
    blob = torch.load(PRIOR_DIR / "head.pt", map_location=device)
    head = HistHead(blob["in_dim"], blob["K"]).to(device)
    head.load_state_dict(blob["state"]); head.eval()
    out = {}
    for e in args.examples:
        eid, path = e.split("=", 1)
        x = TF(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            z = enc(x)
            hist = torch.softmax(head(z.float()), -1)[0]
        out[eid] = hist.float().cpu().numpy()
        print(f"[predict] {eid}: max_p={out[eid].max():.4f} "
              f"nnz(>1e-4)={(out[eid] > 1e-4).sum()}", flush=True)
    np.savez(args.out, **out)
    print(f"[predict] wrote {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--ckpt", required=True)
    b.add_argument("--device", default="cuda:0")
    b.add_argument("--K", type=int, default=1024)
    b.add_argument("--imgs_per_class", type=int, default=150)
    b.add_argument("--px_per_img", type=int, default=2048)
    b.add_argument("--kmeans_iters", type=int, default=30)
    b.add_argument("--batch", type=int, default=48)

    t = sub.add_parser("train")
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=512)

    p = sub.add_parser("predict")
    p.add_argument("examples", nargs="+", help="ID=ABS_PATH pairs")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")

    args = ap.parse_args()
    {"build": cmd_build, "train": cmd_train, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
