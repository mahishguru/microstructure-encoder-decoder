# microstructure-encoder-decoder

A ViT-H/14 encoder and a flow-matching diffusion-transformer (FM-DiT) decoder that learn a compact, searchable latent space `z ∈ R^(16 × d_z)` of grain morphology and HCP texture in extruded Mg alloys. It comes in four bottleneck widths (512, 768, 1024 and 1280), with ViT-DiT, ViT-SDXL and ViT-VQGAN baselines.

🤗 **Weights:** [huggingface.co/mahishguru/microstructure-encoder-decoder](https://huggingface.co/mahishguru/microstructure-encoder-decoder)

📦 **Training data:** 101,000 synthetic RVEs (codec-encoded PNG orientation maps, 15 alloy classes), [Zenodo record 23036836](https://zenodo.org/records/23036836) (DOI [10.5281/zenodo.23036836](https://doi.org/10.5281/zenodo.23036836), CC BY 4.0; files available on request through Zenodo)

**Companion repositories:**
- [orientation-codec](https://github.com/mahishguru/orientation-codec) converts between RGB images and HCP orientation fields.
- [meridian](https://github.com/mahishguru/meridian) optimises in this latent space against a DAMASK oracle.

**Documentation:**
- [Model](docs/model.md): architecture, training objective and baselines.
- [Results](docs/results.md): reconstruction metrics, gallery and pole figures.
- [Usage](docs/usage.md): data, training, evaluation, environment variables and licences.

## Installation

```bash
git clone https://github.com/mahishguru/microstructure-encoder-decoder.git
cd microstructure-encoder-decoder
pip install -e ".[eval]"        # [vqgan] for the ViT-VQGAN baseline
export HF_TOKEN=...             # access to the gated stabilityai/stable-diffusion-3.5-medium
```

Requires Python ≥ 3.9, PyTorch ≥ 2.3 and CUDA. The versions used for the paper models are pinned in [`requirements-lock.txt`](requirements-lock.txt).

## Quick start

```python
import os, torch
os.environ.update(FMDIT_LORA_RANK_OVERRIDE="64", FMDIT_LORA_ALPHA_OVERRIDE="64",
                  FMDIT_LORA_FFN="1", FMDIT_TEXHEAD="1")   # architecture of the released models

from microstructure_ed.checkpoints import download, load_encoder_decoder
from microstructure_ed.encoder_arch_pretrained import Compressor
from microstructure_ed.fmdit.decoder_arch_pretrained import FlowMatchingDiTDecoder, SD35VAE

W = 512                                                     # 512 | 768 | 1024 | 1280
encoder = Compressor(use_gradient_checkpointing=False, trainable_blocks=0, target_dim=W, spatial_tokens=16)
decoder = FlowMatchingDiTDecoder(target_dim=W)
load_encoder_decoder(encoder, decoder, download("vitfmdit" if W == 512 else f"vitfmdit_{W}"))
vae = SD35VAE(token=os.environ["HF_TOKEN"])
encoder, decoder, vae = encoder.cuda().eval(), decoder.cuda().eval(), vae.to("cuda")

x = ...  # (B, 3, 512, 512) orientation images in [-1, 1]
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    z = encoder(x)
    img = vae.decode(decoder.sample(z, num_steps=25, noise_temp=0.9, shift=1.0, guidance_scale=3.0))
```

## Training and evaluation

```bash
python scripts/split_dataset.py /path/to/rve --out .          # dataset_train/ + dataset_test/
scripts/train_fmdit.sh 512                                     # 768 | 1024 | 1280; baselines: scripts/train_baseline.sh vitdit
scripts/eval_testset.sh vitfmdit checkpoints/fmdit_512_v4scratch/checkpoint_epoch_040.pth
scripts/compute_testset_metrics.sh
```

See [docs/usage.md](docs/usage.md) for the full recipe.

## License

The code is released under the MIT license ([LICENSE](LICENSE)). The released weights inherit the licences of their base models; ViT-DiT is **non-commercial**. See the [model card](https://huggingface.co/mahishguru/microstructure-encoder-decoder).

## Acknowledgements

Developed at the Institute of Material and Process Design, Helmholtz-Zentrum Hereon. The ViT-VQGAN baseline builds on [PaintMind](https://github.com/Qiyuan-Ge/PaintMind) by Qiyuan Ge.
