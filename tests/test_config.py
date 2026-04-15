"""Tests for dyson_cnn.config: JSON loading, profile resolution, errors."""

from __future__ import annotations

import json

import pytest

from dyson_cnn.config import (
    load_all,
    load_dataset_cfg,
    load_inference_cfg,
    load_paths,
    load_training_cfg,
)


def test_paths_example_is_valid_json_with_required_fields(repo_root):
    example = repo_root / "config" / "paths.example.json"
    assert example.exists(), "paths.example.json must be shipped in git"
    raw = json.loads(example.read_text())
    for field in [
        "drive_root_mac",
        "drive_root_colab",
        "project_subdir",
        "runs_subdir",
    ]:
        assert field in raw, f"paths.example.json missing required field: {field}"


def test_load_paths_resolves_derived_fields(config_dir):
    paths = load_paths(config_dir)
    assert paths["runtime"] in ("mac", "colab")
    assert "project_dir" in paths
    assert "runs_dir" in paths
    # project_dir must be an absolute path containing the subdir name
    assert paths["project_subdir"] in paths["project_dir"]
    # runs_dir must be rooted under project_dir
    assert paths["project_dir"] in paths["runs_dir"]


def test_load_dataset_cfg_has_core_invariants(config_dir):
    ds = load_dataset_cfg(config_dir)
    # Npoints=4096 is a hard invariant — if this ever changes, coordinated
    # edits are required across MATLAB, Python data pipeline, and any saved
    # models (their input layer shape is baked in).
    assert ds["Npoints"] == 4096, "Npoints must stay 4096"
    assert ds["N"] > 0
    assert ds["dB_thr_G"] == 1000
    assert len(ds["B0Range_G"]) == 2
    assert len(ds["dBRange_G"]) == 2
    assert len(ds["pRange"]) == 2
    # Ranges must be ordered low -> high
    assert ds["B0Range_G"][0] < ds["B0Range_G"][1]
    assert ds["dBRange_G"][0] < ds["dBRange_G"][1]


def test_load_training_cfg_default_profile_has_required_fields(config_dir):
    tr = load_training_cfg(config_dir)
    assert "profile_name" in tr
    assert tr["profile_name"] in ("colab_full", "mac_full", "mac_dev")
    for field in [
        "epochs",
        "batch_size",
        "dropout",
        "learning_rate",
        "loss_weights",
        "max_samples",
        "random_seed",
        "early_stopping_patience",
        "reduce_lr_patience",
    ]:
        assert field in tr, f"Missing training field: {field}"
    # Loss weights must cover all three heads
    lw = tr["loss_weights"]
    assert set(lw.keys()) >= {"B0", "dB", "p3"}


def test_load_training_cfg_mac_dev_profile(config_dir):
    tr = load_training_cfg(config_dir, profile="mac_dev")
    assert tr["profile_name"] == "mac_dev"
    assert tr["max_samples"] == 500
    assert tr["mixed_precision"] is False
    assert tr["epochs"] == 10


def test_load_training_cfg_colab_full_profile(config_dir):
    tr = load_training_cfg(config_dir, profile="colab_full")
    assert tr["profile_name"] == "colab_full"
    assert tr["max_samples"] is None
    assert tr["mixed_precision"] is True
    assert tr["epochs"] == 300


def test_dyson_profile_env_var_overrides_active_profile(config_dir, monkeypatch):
    monkeypatch.setenv("DYSON_PROFILE", "mac_dev")
    tr = load_training_cfg(config_dir)  # no explicit profile arg
    assert tr["profile_name"] == "mac_dev"


def test_explicit_profile_beats_env_var(config_dir, monkeypatch):
    monkeypatch.setenv("DYSON_PROFILE", "mac_dev")
    tr = load_training_cfg(config_dir, profile="colab_full")
    assert tr["profile_name"] == "colab_full"


def test_load_training_cfg_unknown_profile_raises(config_dir):
    with pytest.raises(ValueError, match="not found"):
        load_training_cfg(config_dir, profile="nonexistent_profile_xyz")


def test_load_inference_cfg_has_required_fields(config_dir):
    inf = load_inference_cfg(config_dir)
    assert "runName" in inf
    assert "spectrum_basename" in inf
    assert "dataset_prefix" in inf


def test_load_all_returns_four_dicts(config_dir):
    result = load_all(config_dir)
    assert len(result) == 4
    paths, ds, tr, inf = result
    assert "project_dir" in paths
    assert "N" in ds
    assert "profile_name" in tr
    assert "runName" in inf


def test_missing_paths_json_raises_clear_error(tmp_path):
    cfg = tmp_path / "config"
    cfg.mkdir()
    with pytest.raises(FileNotFoundError, match="paths.example"):
        load_paths(cfg)


def test_colab_falls_back_to_paths_example_when_paths_json_absent(
    tmp_path, monkeypatch
):
    """On Colab, paths.json is optional — load_paths should fall back to
    paths.example.json (drive_root_colab is universal across users)."""
    cfg = tmp_path / "config"
    cfg.mkdir()
    # Write a minimal paths.example.json (no paths.json)
    (cfg / "paths.example.json").write_text(
        '{"drive_root_mac": "/Users/<u>/Library/CloudStorage/GoogleDrive-<e>/My Drive",'
        ' "drive_root_colab": "/content/drive/MyDrive",'
        ' "project_subdir": "Python/DysonianLineCNN",'
        ' "runs_subdir": "runs",'
        ' "data_subdir": "data"}'
    )

    # Force runtime detection to "colab" by monkey-patching the helper
    import dyson_cnn.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "_detect_platform", lambda: "colab")

    paths = load_paths(cfg)
    assert paths["runtime"] == "colab"
    assert paths["drive_root"] == "/content/drive/MyDrive"
    assert paths["project_dir"] == "/content/drive/MyDrive/Python/DysonianLineCNN"


def test_mac_does_not_fall_back_to_example_even_if_present(tmp_path, monkeypatch):
    """On Mac, paths.json is required (drive_root_mac is user-specific).
    The fallback must NOT trigger even when paths.example.json exists."""
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "paths.example.json").write_text(
        '{"drive_root_mac": "/Users/<u>/Library/CloudStorage/GoogleDrive-<e>/My Drive",'
        ' "drive_root_colab": "/content/drive/MyDrive",'
        ' "project_subdir": "Python/DysonianLineCNN",'
        ' "runs_subdir": "runs",'
        ' "data_subdir": "data"}'
    )

    import dyson_cnn.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "_detect_platform", lambda: "mac")

    with pytest.raises(FileNotFoundError, match="paths.example"):
        load_paths(cfg)


def test_doc_fields_are_stripped(config_dir):
    """Fields starting with underscore are human-readable docs and must
    be stripped from the returned dict so downstream code never sees them."""
    ds = load_dataset_cfg(config_dir)
    assert "_doc" not in ds
    tr = load_training_cfg(config_dir)
    assert "_doc" not in tr


# ---------------------------------------------------------------------------
# Per-set config loading
# ---------------------------------------------------------------------------


def test_load_dataset_cfg_with_set_name(config_dir):
    """load_dataset_cfg(set_name='set-1') should read config/sets/set-1.json."""
    ds = load_dataset_cfg(config_dir, set_name="set-1")
    assert ds["Npoints"] == 4096, "Npoints must stay 4096"
    assert ds["N"] > 0
    assert len(ds["B0Range_G"]) == 2


def test_load_dataset_cfg_without_set_name_falls_back_to_legacy(config_dir):
    """Without set_name, load_dataset_cfg reads config/dataset.json (legacy)."""
    ds = load_dataset_cfg(config_dir)
    assert ds["Npoints"] == 4096
    # Legacy config has dBRange [250, 350]
    assert ds["dBRange_G"] == [250, 350]


def test_load_dataset_cfg_nonexistent_set_raises(config_dir):
    with pytest.raises(FileNotFoundError):
        load_dataset_cfg(config_dir, set_name="set-99")


def test_load_paths_with_set_name_adds_set_paths(config_dir):
    paths = load_paths(config_dir, set_name="set-1")
    # All four per-set path keys must be present
    assert "set_name" in paths
    assert "set_cnn_dir" in paths
    assert "set_runs_dir" in paths
    assert "set_dataset_dir" in paths
    assert "set_project_dir" in paths  # legacy alias, kept for back-compat
    assert paths["set_name"] == "set-1"
    # All per-set paths must be rooted under project_dir/results/set-1/cnn/
    for key in ("set_cnn_dir", "set_runs_dir", "set_dataset_dir", "set_project_dir"):
        assert "set-1" in paths[key]
        assert "results" in paths[key]
        assert "cnn" in paths[key]
    # set_runs_dir ends with runs_subdir, set_dataset_dir ends with "dataset"
    assert paths["set_runs_dir"].endswith(paths["runs_subdir"])
    assert paths["set_dataset_dir"].endswith("dataset")
    # Legacy alias points at the cnn directory
    assert paths["set_project_dir"] == paths["set_cnn_dir"]


def test_load_paths_without_set_name_has_no_set_paths(config_dir):
    paths = load_paths(config_dir)
    for key in ("set_cnn_dir", "set_runs_dir", "set_dataset_dir", "set_project_dir"):
        assert key not in paths


def test_load_all_with_set_name(config_dir):
    paths, ds, tr, inf = load_all(config_dir, set_name="set-1")
    assert "set_cnn_dir" in paths
    assert "set_dataset_dir" in paths
    assert ds["Npoints"] == 4096


def test_inference_cfg_has_set_name(config_dir):
    inf = load_inference_cfg(config_dir)
    assert "set_name" in inf
    assert inf["set_name"].startswith("set-")
