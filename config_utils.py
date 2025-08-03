"""
config_utils.py – YAML-based config loader with CLI override.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Any, Dict

import yaml


def _flatten(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """foo.bar=1 → {'foo': {'bar': 1}} (inverse happens in CLI parsing)."""
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def load_config(argv: list[str] | None = None) -> argparse.Namespace:
    """
    1. Parse `--config path.yaml` (default: configs/config.yaml).
    2. Load YAML into nested dict.
    3. Add *all* YAML keys as CLI args (`--foo.bar.baz value`) so any field
       can be overridden at runtime.
    4. Return *argparse.Namespace* with dot access (`cfg.model.name`).
    """
    # First pass – capture only --config
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, default="configs/config.yaml",
                             help="Path to YAML config with defaults.")
    known, _ = base_parser.parse_known_intermixed_args(argv)

    cfg_path = Path(known.config).expanduser()
    if not cfg_path.exists():
        sys.exit(f"❌ Config file not found: {cfg_path}")

    with cfg_path.open() as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f)

    flat = _flatten(raw_cfg)
    # Second pass – build a full parser *dynamically*
    parser = argparse.ArgumentParser(parents=[base_parser])
    for dotted_key, default in flat.items():
        arg_key = dotted_key.replace(".", "_")

        # bool → store_true / store_false
        if isinstance(default, bool):
            parser.add_argument(
                f"--{dotted_key}", dest=arg_key,
                action="store_true" if not default else "store_false"
            )
        # list / tuple / dict → ast.literal_eval on CLI string
        elif isinstance(default, (list, tuple, dict)):
            parser.add_argument(
                f"--{dotted_key}", dest=arg_key,
                type=lambda s: ast.literal_eval(s),
                default=default
            )
        # numbers → int / float
        elif isinstance(default, numbers.Number):
            parser.add_argument(
                f"--{dotted_key}", dest=arg_key,
                type=type(default), default=default
            )
        # everything else → str
        else:
            parser.add_argument(
                f"--{dotted_key}", dest=arg_key,
                type=str, default=default
            )
    args = parser.parse_args(argv)
    # ✨ new: auto-literal-eval any CLI value that *looks* like a list/dict/tuple
    for k, v in vars(args).items():
        if isinstance(v, str) and v and v[0] in "[{(":
            try:
                setattr(args, k, ast.literal_eval(v))
            except Exception:
                pass   # leave as string if it isn't valid Python literal
            
    # Third pass – rebuild nested dict from namespace
    merged: Dict[str, Any] = {}
    for dotted_key, _ in flat.items():
        segs = dotted_key.split(".")
        cur = merged
        for seg in segs[:-1]:
            cur = cur.setdefault(seg, {})
        arg_key = dotted_key.replace(".", "_")
        if hasattr(args, arg_key):
            value = getattr(args, arg_key)
            segs = dotted_key.split(".")
            cur = merged
            for seg in segs[:-1]:
                cur = cur.setdefault(seg, {})
            cur[segs[-1]] = value
        else:
            print(f"[config_utils] Skipping unknown config key: {dotted_key} → {arg_key}")

    # Convenience: expose as attribute-style access
    class DotDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    return argparse.Namespace(**{k: DotDict(v) if isinstance(v, dict) else v
                                 for k, v in merged.items()})
import ast, numbers, re

_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")

def _to_native(x):
    """Recursively cast str → int/float/bool/list/dict."""
    # If it's already not a string, recurse if it’s a container
    if isinstance(x, list):
        return [_to_native(i) for i in x]
    if isinstance(x, tuple):
        return tuple(_to_native(i) for i in x)
    if isinstance(x, dict):
        return {k: _to_native(v) for k, v in x.items()}
    if not isinstance(x, str):
        return x

    # It's a string → first see if it looks numeric
    if _NUM_RE.match(x):
        return int(x) if x.isdigit() else float(x)

    # Try literal_eval for '[…]', '{…}', 'True', 'False', etc.
    try:
        v = ast.literal_eval(x)
        return _to_native(v)
    except Exception:
        return x  # leave as-is
# config_utils.py  (append to the bottom -- after load_config definition)
# ──────────────────────────────────────────────────────────────────────
def _normalize_sft(sft: argparse.Namespace) -> None:
    """
    In-place fixes that your old cfg_init() did:
      • Ensure lists have length == num_epochs
      • Accept scalar → list
      • Fill default values
    """
    sft.unfreeze_ids = _to_native(getattr(sft, "unfreeze_ids", None))
    sft.do_eval = _to_native(getattr(sft, "do_eval", None))
    sft.max_step_per_epoch = _to_native(getattr(sft, "max_step_per_epoch", None))
    
    
    # 1. Unfreeze layers
    if sft.unfreeze_ids is None:
        sft.unfreeze_ids = [[0]] * sft.num_epochs
    elif not isinstance(sft.unfreeze_ids[0], list):
        sft.unfreeze_ids = [sft.unfreeze_ids]

    if len(sft.unfreeze_ids) < sft.num_epochs:
        pad = [sft.unfreeze_ids[-1]] * (sft.num_epochs - len(sft.unfreeze_ids))
        sft.unfreeze_ids += pad

    # 2. Max steps per epoch
    if sft.max_step_per_epoch is None:
        sft.max_step_per_epoch = [1000] * sft.num_epochs
    elif not isinstance(sft.max_step_per_epoch, list):
        sft.max_step_per_epoch = [sft.max_step_per_epoch]

    if len(sft.max_step_per_epoch) < sft.num_epochs:
        pad = [sft.max_step_per_epoch[-1]] * (sft.num_epochs - len(sft.max_step_per_epoch))
        sft.max_step_per_epoch += pad

    # 3. do_eval flags
    if sft.do_eval is None:
        sft.do_eval = [False] * sft.num_epochs
    elif not isinstance(sft.do_eval, list):
        sft.do_eval = [sft.do_eval]

    if len(sft.do_eval) < sft.num_epochs:
        sft.do_eval += [False] * (sft.num_epochs - len(sft.do_eval))

    # 4. init_max_CL fallback
def select(cfg: argparse.Namespace, section: str) -> argparse.Namespace:
    """Pull a top-level subsection like cfg.sft or cfg.eval."""
    if not hasattr(cfg, section):
        raise ValueError(f"Missing section '{section}' in config.")
    return getattr(cfg, section)
