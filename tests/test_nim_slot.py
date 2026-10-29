import signal
from pathlib import Path

import pytest

pytest.importorskip("openai")
from experiments.llm import nim_slot


def test_shared_slot_released_on_termination(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    occupied = tmp_path / ".cache/nim-slot-1"
    occupied.mkdir(parents=True)
    original = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as error, nim_slot():
        owned = tmp_path / ".cache/nim-slot-2"
        assert (owned / "owner").exists()
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    assert error.value.code == 143
    assert occupied.exists() and not owned.exists()
    assert signal.getsignal(signal.SIGTERM) == original
