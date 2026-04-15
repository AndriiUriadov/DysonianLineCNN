"""JSON config loading for DysonianLineCNN.

Single source of truth for paths, dataset generator inputs, CNN training
hyperparameters, and inference targets. See `config/README.md` for what
each field means.

All loaders accept `config_dir` as a `str` or `pathlib.Path` and return
plain dicts (training.py and friends work with dict access, not attribute
access, to stay serialization-friendly).
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Hint: if this is paths.json, copy config/paths.example.json "
            f"to config/paths.json and edit drive_root_mac for your machine."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _strip_docs(cfg: Any) -> Any:
    """Recursively remove `_doc` fields from a config dict for clean consumption."""
    if isinstance(cfg, dict):
        return {k: _strip_docs(v) for k, v in cfg.items() if k != "_doc"}
    if isinstance(cfg, list):
        return [_strip_docs(v) for v in cfg]
    return cfg


def _detect_platform() -> str:
    """Return `"colab"` if running inside Google Colab, `"mac"` otherwise.

    Colab detection is based on the `/content` directory, which is guaranteed
    to exist in every Colab runtime and nowhere else by convention.
    """
    if os.path.isdir("/content"):
        return "colab"
    if platform.system() == "Darwin":
        return "mac"
    # Fall back to mac for anything non-Colab; the user can still override
    # manually via paths.json by editing drive_root_mac.
    return "mac"


def load_paths(
    config_dir: str | Path,
    set_name: str | None = None,
) -> dict[str, Any]:
    """Load `paths.json` and resolve platform-specific roots.

    Returns a dict with the following resolved absolute paths on top of the
    raw fields from paths.json:

        drive_root:  the active drive root for this runtime (mac or colab)
        project_dir: drive_root / project_subdir
        runs_dir:    project_dir / runs_subdir
        data_dir:    project_dir / data_subdir

    When ``set_name`` is given (e.g. ``"set-1"``), extra paths are added,
    all rooted under ``project_dir/results/<set_name>/cnn/``:

        set_name:        the set identifier
        set_cnn_dir:     project_dir / results / <set> / cnn
                         (per-spectrum inputs/outputs: CSV, JSON, PNG)
        set_runs_dir:    project_dir / results / <set> / cnn / runs
                         (trained CNN models saved here)
        set_dataset_dir: project_dir / results / <set> / cnn / dataset
                         (synthetic training data from DysonGeneratorMix)
        set_project_dir: alias of set_cnn_dir (legacy name, still exposed)

    On Colab, `paths.json` is optional: if absent, the function falls back to
    the committed `paths.example.json`. This works because `drive_root_colab`
    is universal across users (`/content/drive/MyDrive`), so the example file
    is sufficient and no per-user editing is needed in Colab.

    On Mac (or any non-Colab runtime), `paths.json` is required because
    `drive_root_mac` is user-specific (depends on the Google Drive for
    Desktop mount path, which embeds the user's email).
    """
    config_dir = Path(config_dir)
    paths_file = config_dir / "paths.json"
    example_file = config_dir / "paths.example.json"
    runtime = _detect_platform()

    if paths_file.exists():
        raw = _strip_docs(_read_json(paths_file))
    elif runtime == "colab" and example_file.exists():
        # Colab fallback: drive_root_colab is universal, no per-user override needed.
        raw = _strip_docs(_read_json(example_file))
    else:
        # Re-read to surface the standard "missing paths.json" error message.
        raw = _strip_docs(_read_json(paths_file))

    if runtime == "colab":
        drive_root = raw["drive_root_colab"]
    else:
        drive_root = raw["drive_root_mac"]

    project_dir = Path(drive_root) / raw["project_subdir"]
    runs_dir = project_dir / raw["runs_subdir"]
    data_dir = project_dir / raw["data_subdir"]

    result = {
        **raw,
        "runtime": runtime,
        "drive_root": str(Path(drive_root)),
        "project_dir": str(project_dir),
        "runs_dir": str(runs_dir),
        "data_dir": str(data_dir),
    }

    if set_name is not None:
        # New layout (April 2026): all per-set outputs consolidated under
        # results/<set>/cnn/ on both local repo and Drive.
        set_cnn_dir = project_dir / "results" / set_name / "cnn"
        set_runs_dir = set_cnn_dir / raw["runs_subdir"]
        set_dataset_dir = set_cnn_dir / "dataset"
        result["set_name"] = set_name
        result["set_cnn_dir"] = str(set_cnn_dir)
        result["set_runs_dir"] = str(set_runs_dir)
        result["set_dataset_dir"] = str(set_dataset_dir)
        # Legacy alias — kept so older callers reading set_project_dir
        # still get a meaningful path (the per-set CNN working directory).
        result["set_project_dir"] = str(set_cnn_dir)

    return result


def load_dataset_cfg(
    config_dir: str | Path,
    set_name: str | None = None,
) -> dict[str, Any]:
    """Load dataset generation parameters.

    When ``set_name`` is given (e.g. ``"set-1"``), reads
    ``config/sets/set-1.json``. Otherwise falls back to the legacy
    ``config/dataset.json`` for backward compatibility.
    """
    config_dir = Path(config_dir)
    if set_name is not None:
        path = config_dir / "sets" / f"{set_name}.json"
    else:
        path = config_dir / "dataset.json"
    return _strip_docs(_read_json(path))


def load_training_cfg(
    config_dir: str | Path,
    profile: str | None = None,
) -> dict[str, Any]:
    """Load `training.json` and resolve a profile.

    Profile resolution order (highest priority first):
        1. explicit `profile` argument
        2. DYSON_PROFILE environment variable
        3. `active_profile` field in the JSON file

    Returns a flat dict with the resolved profile's fields plus a synthetic
    `profile_name` field identifying which profile was chosen.
    """
    raw = _strip_docs(_read_json(Path(config_dir) / "training.json"))

    resolved = profile or os.environ.get("DYSON_PROFILE") or raw.get("active_profile")
    if resolved is None:
        raise ValueError(
            "No profile specified: pass `profile=...`, set DYSON_PROFILE, or "
            "add `active_profile` to training.json."
        )

    profiles = raw.get("profiles", {})
    if resolved not in profiles:
        available = sorted(profiles.keys())
        raise ValueError(
            f"Profile '{resolved}' not found in training.json. "
            f"Available: {available}"
        )

    cfg = dict(profiles[resolved])
    cfg["profile_name"] = resolved
    return cfg


def load_inference_cfg(config_dir: str | Path) -> dict[str, Any]:
    """Load `inference.json`."""
    return _strip_docs(_read_json(Path(config_dir) / "inference.json"))


def load_all(
    config_dir: str | Path,
    profile: str | None = None,
    set_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Convenience: load all four configs in one call.

    When ``set_name`` is given, paths and dataset_cfg are resolved for that
    set (per-set Drive subdirectory and ``config/sets/<set_name>.json``).

    Returns (paths, dataset_cfg, training_cfg, inference_cfg).
    """
    return (
        load_paths(config_dir, set_name=set_name),
        load_dataset_cfg(config_dir, set_name=set_name),
        load_training_cfg(config_dir, profile=profile),
        load_inference_cfg(config_dir),
    )
