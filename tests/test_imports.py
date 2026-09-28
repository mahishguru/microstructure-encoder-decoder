"""Every library module imports without side effects (no downloads, no files created)."""
import importlib
import os
import pkgutil

import pytest

import microstructure_ed

ENTRY_POINTS = ("trainer", "inference")          # training / inference scripts (run DDP at import)
MODULES = [m.name for m in pkgutil.walk_packages(microstructure_ed.__path__, "microstructure_ed.")
           if not m.name.endswith(ENTRY_POINTS) and "vitvqgan" not in m.name]


@pytest.mark.parametrize("name", MODULES)
def test_import(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    importlib.import_module(name)
    assert not os.listdir(tmp_path), f"importing {name} created files in the working directory"


def test_fmdit_width_is_a_constructor_argument():
    import inspect
    from microstructure_ed.fmdit.decoder_arch_pretrained import FlowMatchingDiTDecoder, LatentToTokens
    assert "target_dim" in inspect.signature(FlowMatchingDiTDecoder.__init__).parameters
    assert "target_dim" in inspect.signature(LatentToTokens.__init__).parameters
