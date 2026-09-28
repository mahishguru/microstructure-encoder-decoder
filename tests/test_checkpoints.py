"""Release-checkpoint export/load logic, on a toy model (no downloads, CPU only)."""
import pytest
import torch
import torch.nn as nn

from microstructure_ed.checkpoints import RELEASE_FORMAT, export_release, load_encoder_decoder


class ToyDecoder(nn.Module):
    """Mimics FM-DiT naming: trained adapter + frozen `transformer` with LoRA tensors."""

    def __init__(self):
        super().__init__()
        self.token_generator = nn.Linear(4, 4)
        self.transformer = nn.Module()
        self.transformer.backbone = nn.Linear(4, 4)          # frozen (dropped on export)
        self.transformer.lora_A = nn.Linear(4, 2, bias=False)  # trained (kept)


def _training_ckpt(path, enc, dec):
    torch.save({"epoch": 40, "encoder": enc.state_dict(), "decoder": dec.state_dict(),
                "optimizer": {"state": {}}, "lr_scheduler": {}}, path)


def test_fmdit_release_roundtrip(tmp_path):
    torch.manual_seed(0)
    enc, dec = nn.Linear(3, 4), ToyDecoder()
    _training_ckpt(tmp_path / "train.pth", enc, dec)

    info = export_release(tmp_path / "train.pth", tmp_path / "rel.pth", target_dim=4, strip_frozen_backbone=True)
    rel = torch.load(tmp_path / "rel.pth", weights_only=False)
    assert rel["format"] == RELEASE_FORMAT and rel["stripped_frozen_backbone"]
    assert "optimizer" not in rel
    assert info["dropped_frozen_decoder_tensors"] == 2           # backbone weight + bias
    assert set(rel["decoder"]) == {"token_generator.weight", "token_generator.bias", "transformer.lora_A.weight"}

    enc2, dec2 = nn.Linear(3, 4), ToyDecoder()
    backbone_before = dec2.transformer.backbone.weight.clone()
    load_encoder_decoder(enc2, dec2, tmp_path / "rel.pth")
    assert torch.equal(enc2.weight, enc.weight)
    assert torch.equal(dec2.token_generator.weight, dec.token_generator.weight)
    assert torch.equal(dec2.transformer.lora_A.weight, dec.transformer.lora_A.weight)
    assert torch.equal(dec2.transformer.backbone.weight, backbone_before)  # left to the pretrained init


def test_full_decoder_kept_without_stripping(tmp_path):
    enc, dec = nn.Linear(3, 4), ToyDecoder()
    _training_ckpt(tmp_path / "train.pth", enc, dec)
    export_release(tmp_path / "train.pth", tmp_path / "rel.pth", strip_frozen_backbone=False)
    rel = torch.load(tmp_path / "rel.pth", weights_only=False)
    assert len(rel["decoder"]) == len(dec.state_dict())


def test_missing_trained_key_is_an_error(tmp_path):
    enc, dec = nn.Linear(3, 4), ToyDecoder()
    _training_ckpt(tmp_path / "train.pth", enc, dec)
    export_release(tmp_path / "train.pth", tmp_path / "rel.pth", strip_frozen_backbone=True)
    rel = torch.load(tmp_path / "rel.pth", weights_only=False)
    del rel["decoder"]["token_generator.weight"]
    torch.save(rel, tmp_path / "broken.pth")
    with pytest.raises(RuntimeError, match="missing trained keys"):
        load_encoder_decoder(nn.Linear(3, 4), ToyDecoder(), tmp_path / "broken.pth")
