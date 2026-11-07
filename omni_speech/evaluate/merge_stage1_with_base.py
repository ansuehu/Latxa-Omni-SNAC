"""Interpolate a Stage 1 checkpoint's LLM weights back toward base Latxa (WiSE-FT).

Every weight present in both models becomes alpha * stage1 + (1 - alpha) * base;
the speech encoder and projector exist only in Stage 1 and are copied unchanged.
The CLI writes one checkpoint per alpha (~17G each), in the same sharded layout,
loadable by create_model and by LlamaForCausalLM. The eval scripts instead merge
in memory with interpolate_model, given a "<checkpoint>@<alpha>" model spec.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def merge_tensor(stage1, base, alpha):
    if stage1.shape != base.shape:
        raise ValueError(f"shape mismatch: {tuple(stage1.shape)} vs {tuple(base.shape)}")
    return (alpha * stage1.float() + (1 - alpha) * base.float()).to(stage1.dtype)


def weight_map(model_dir):
    with open(Path(model_dir) / "model.safetensors.index.json", encoding="utf-8") as stream:
        return json.load(stream)["weight_map"]


def resolve_dir(model):
    return Path(model) if Path(model).is_dir() else Path(snapshot_download(model, local_files_only=True))


def parse_spec(spec):
    """'<checkpoint>@<alpha>' -> (checkpoint, alpha); a bare path has alpha None."""
    path, _, alpha = spec.rpartition("@")
    return (path, float(alpha)) if path else (spec, None)


@torch.no_grad()
def interpolate_model(model, base, alpha):
    """In place: parameters that base Latxa also has become alpha * model + (1 - alpha) * base."""
    base_dir = resolve_dir(base)
    base_map = weight_map(base_dir)
    params = dict(model.named_parameters())
    missing = sorted(set(base_map) - set(params))
    if missing:
        raise ValueError(f"{len(missing)} base weights missing from the model, e.g. {missing[:3]}")
    for name in sorted(set(base_map.values())):
        with safe_open(base_dir / name, "pt") as stream:
            for key in stream.keys():
                param = params[key]
                param.copy_(merge_tensor(param.data, stream.get_tensor(key).to(param.device), alpha))
    return model


def merge(stage1_dir, base_dir, alphas, output_dir):
    stage1_dir, base_dir = Path(stage1_dir), Path(base_dir)
    stage1_map, base_map = weight_map(stage1_dir), weight_map(base_dir)
    missing = sorted(set(base_map) - set(stage1_map))
    if missing:
        raise ValueError(f"{len(missing)} base weights missing from Stage 1, e.g. {missing[:3]}")
    outputs = {}
    for alpha in alphas:
        out = Path(output_dir) / f"alpha_{alpha:g}"
        out.mkdir(parents=True, exist_ok=True)
        for path in stage1_dir.iterdir():
            if path.is_file() and path.suffix != ".safetensors" and path.name != "README.md":
                shutil.copy2(path, out / path.name)
        outputs[alpha] = out
    base_files = {name: safe_open(base_dir / name, "pt") for name in set(base_map.values())}
    for shard in sorted(set(stage1_map.values())):
        tensors = load_file(stage1_dir / shard)
        base = {key: base_files[base_map[key]].get_tensor(key) for key in tensors if key in base_map}
        for alpha, out in outputs.items():
            merged = {
                key: merge_tensor(value, base[key], alpha) if key in base else value
                for key, value in tensors.items()
            }
            save_file(merged, out / shard, metadata={"format": "pt"})
        print(f"{shard}: merged {len(base)}, copied {len(tensors) - len(base)}", flush=True)
    return outputs


def main(args):
    merge(args.stage1, resolve_dir(args.base), args.alphas, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage1", help="weights-only Stage 1 checkpoint directory")
    parser.add_argument("--base", default="HiTZ/Latxa-Llama-3.1-8B-Instruct")
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    with torch.no_grad():
        main(parser.parse_args())
