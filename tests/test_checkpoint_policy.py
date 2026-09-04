"""Checkpoint disk-policy tests: latest-only by default, archives opt-in."""

from __future__ import annotations

import torch
import pytest

from train.checkpoint import save_checkpoint_atomic, save_training_state


def _payload(iteration: int) -> dict:
    return {
        "model": {"w": torch.ones(1)},
        "optimizer": {"state": {}, "param_groups": []},
        "iteration": iteration,
        "config": {},
        "seeds": {},
        "meta": {},
    }


def _listing(tmp_path, prefix):
    return sorted(p.name for p in tmp_path.glob(f"{prefix}*.pt"))


def test_default_keeps_only_latest(tmp_path):
    for i in (1, 2, 3):
        save_training_state(tmp_path, "run", _payload(i), archive_every=1, iteration=i)
    names = _listing(tmp_path, "run")
    assert names == ["run_latest.pt"]
    # The LATEST file is the most recent payload, not an old archive.
    latest = torch.load(tmp_path / "run_latest.pt", map_location="cpu",
                        weights_only=False)
    assert latest["iteration"] == 3


def test_keep_previous_opt_in(tmp_path):
    save_training_state(tmp_path, "run", _payload(1), archive_every=1,
                        iteration=1, keep_previous=True)
    save_training_state(tmp_path, "run", _payload(2), archive_every=1,
                        iteration=2, keep_previous=True)
    names = _listing(tmp_path, "run")
    assert sorted(names) == ["run_latest.pt", "run_prev.pt"]
    prev = torch.load(tmp_path / "run_prev.pt", map_location="cpu",
                      weights_only=False)
    assert prev["iteration"] == 1


def test_archives_opt_in(tmp_path):
    save_training_state(tmp_path, "run", _payload(1), archive_every=1,
                        iteration=1, keep_previous=False, keep_archives=True)
    save_training_state(tmp_path, "run", _payload(2), archive_every=1,
                        iteration=2, keep_previous=False, keep_archives=True)
    names = _listing(tmp_path, "run")
    assert sorted(names) == ["run_iter000001.pt", "run_iter000002.pt", "run_latest.pt"]


def test_disabling_archives_cleans_stale_files(tmp_path):
    # Simulate a run that previously stored archives, now switched to latest-only.
    save_training_state(tmp_path, "run", _payload(1), archive_every=1,
                        iteration=1, keep_archives=True)
    assert len(list(tmp_path.glob("run_iter*.pt"))) == 1
    save_training_state(tmp_path, "run", _payload(2), archive_every=1,
                        iteration=2, keep_archives=False)
    # Stale archives and prev files are actively deleted.
    assert _listing(tmp_path, "run") == ["run_latest.pt"]


def test_find_latest_prefers_latest_over_prev(tmp_path):
    from train.checkpoint import find_latest_checkpoint
    save_training_state(tmp_path, "run", _payload(1), archive_every=1,
                        iteration=1, keep_previous=True)
    save_training_state(tmp_path, "run", _payload(2), archive_every=1,
                        iteration=2, keep_previous=True)
    path = find_latest_checkpoint(tmp_path, "run")
    assert path.name == "run_latest.pt"
    latest = torch.load(path, map_location="cpu", weights_only=False)
    assert latest["iteration"] == 2


def test_atomic_write_still_works(tmp_path):
    save_checkpoint_atomic(tmp_path / "run_latest.pt", _payload(9))
    latest = torch.load(tmp_path / "run_latest.pt", map_location="cpu",
                        weights_only=False)
    assert latest["iteration"] == 9