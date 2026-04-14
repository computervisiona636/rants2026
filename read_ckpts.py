import os
import torch
from collections import defaultdict
from omegaconf import DictConfig, ListConfig

torch.serialization.add_safe_globals([DictConfig, ListConfig])

LAST_PATH = "/hdd/RANTS/RANTS-25/checkpoints/last.ckpt"
HAMER_PATH = "/data/phuongttn/rants/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"


def sizeof_tensor_mb(t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / (1024 ** 2)


def safe_torch_load(path):
    print(f"\n[LOAD] {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Loaded: {type(ckpt)}")
    return ckpt


def get_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"], "state_dict"
        if all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt, "root_dict_as_state_dict"
    return None, None


def summarize_top_level(ckpt, name, file_path):
    print("\n" + "=" * 120)
    print(f"SUMMARY: {name}")
    print("=" * 120)
    print(f"File: {file_path}")
    if os.path.exists(file_path):
        print(f"File size: {os.path.getsize(file_path) / (1024**3):.3f} GB")
    else:
        print("File size: path not found")

    print(f"Python type: {type(ckpt)}")

    if not isinstance(ckpt, dict):
        print("Checkpoint is not a dict, cannot inspect top-level keys.")
        return

    print(f"\nTop-level keys ({len(ckpt.keys())}):")
    for k in ckpt.keys():
        print(f"  - {k}")

    print("\nTop-level content summary:")
    for k, v in ckpt.items():
        if torch.is_tensor(v):
            print(f"  - {k}: tensor | shape={tuple(v.shape)} | dtype={v.dtype} | {sizeof_tensor_mb(v):.2f} MB")
        elif isinstance(v, dict):
            tensor_count = 0
            total_mb = 0.0
            nested_types = defaultdict(int)

            for vv in v.values():
                nested_types[type(vv).__name__] += 1
                if torch.is_tensor(vv):
                    tensor_count += 1
                    total_mb += sizeof_tensor_mb(vv)

            nested_types_str = ", ".join(f"{kk}:{vv}" for kk, vv in sorted(nested_types.items()))
            print(
                f"  - {k}: dict | len={len(v)} | tensor_items={tensor_count} | "
                f"tensor_total={total_mb:.2f} MB | value_types=[{nested_types_str}]"
            )
        elif isinstance(v, list):
            print(f"  - {k}: list | len={len(v)}")
        else:
            print(f"  - {k}: {type(v).__name__}")


def summarize_state_dict(sd, name):
    print("\n" + "-" * 120)
    print(f"STATE_DICT SUMMARY: {name}")
    print("-" * 120)

    if sd is None:
        print("No state_dict found.")
        return

    num_tensors = 0
    total_params = 0
    total_mb = 0.0
    dtype_stats = defaultdict(lambda: {"count": 0, "mb": 0.0})
    prefix_stats = defaultdict(lambda: {"count": 0, "mb": 0.0})

    for k, v in sd.items():
        if torch.is_tensor(v):
            num_tensors += 1
            total_params += v.numel()
            mb = sizeof_tensor_mb(v)
            total_mb += mb
            dtype_stats[str(v.dtype)]["count"] += 1
            dtype_stats[str(v.dtype)]["mb"] += mb

            prefix = k.split(".")[0]
            prefix_stats[prefix]["count"] += 1
            prefix_stats[prefix]["mb"] += mb

    print(f"Number of tensor keys: {num_tensors}")
    print(f"Total parameters: {total_params:,}")
    print(f"Approx tensor size: {total_mb / 1024:.3f} GB")

    print("\nDtype distribution:")
    for dtype, info in sorted(dtype_stats.items(), key=lambda x: -x[1]["mb"]):
        print(f"  - {dtype}: {info['count']} tensors | {info['mb'] / 1024:.3f} GB")

    print("\nTop module prefixes by size:")
    for prefix, info in sorted(prefix_stats.items(), key=lambda x: -x[1]["mb"])[:30]:
        print(f"  - {prefix}: {info['count']} tensors | {info['mb'] / 1024:.3f} GB")


def compare_top_level_keys(ckpt_a, name_a, ckpt_b, name_b):
    print("\n" + "=" * 120)
    print(f"TOP-LEVEL KEY COMPARISON: {name_a} vs {name_b}")
    print("=" * 120)

    if not isinstance(ckpt_a, dict) or not isinstance(ckpt_b, dict):
        print("At least one checkpoint is not a dict, cannot compare top-level keys.")
        return

    keys_a = set(ckpt_a.keys())
    keys_b = set(ckpt_b.keys())

    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    print(f"Common top-level keys ({len(common)}):")
    for k in common:
        print(f"  - {k}")

    print(f"\nOnly in {name_a} ({len(only_a)}):")
    for k in only_a:
        print(f"  - {k}")

    print(f"\nOnly in {name_b} ({len(only_b)}):")
    for k in only_b:
        print(f"  - {k}")


def compare_state_dicts(sd_a, name_a, sd_b, name_b, show_samples=80):
    print("\n" + "=" * 120)
    print(f"STATE_DICT COMPARISON: {name_a} vs {name_b}")
    print("=" * 120)

    if sd_a is None or sd_b is None:
        print("At least one checkpoint has no state_dict.")
        return

    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    print(f"Keys in {name_a}: {len(keys_a)}")
    print(f"Keys in {name_b}: {len(keys_b)}")
    print(f"Common keys: {len(common)}")
    print(f"Only in {name_a}: {len(only_a)}")
    print(f"Only in {name_b}: {len(only_b)}")

    print(f"\nSample keys only in {name_a}:")
    for k in only_a[:show_samples]:
        print(f"  - {k}")

    print(f"\nSample keys only in {name_b}:")
    for k in only_b[:show_samples]:
        print(f"  - {k}")

    shape_mismatch = []
    for k in common:
        va = sd_a[k]
        vb = sd_b[k]
        if torch.is_tensor(va) and torch.is_tensor(vb):
            if tuple(va.shape) != tuple(vb.shape):
                shape_mismatch.append((k, tuple(va.shape), tuple(vb.shape)))

    print(f"\nShape mismatches: {len(shape_mismatch)}")
    for item in shape_mismatch[:show_samples]:
        print(f"  - {item[0]} | {name_a}: {item[1]} | {name_b}: {item[2]}")


def inspect_non_state_heavy_parts(ckpt, name):
    print("\n" + "=" * 120)
    print(f"NON-STATE HEAVY PART INSPECTION: {name}")
    print("=" * 120)

    if not isinstance(ckpt, dict):
        print("Checkpoint is not a dict.")
        return

    candidates = []
    for k, v in ckpt.items():
        if k == "state_dict":
            continue

        total_mb = 0.0
        tensor_count = 0

        def walk(obj):
            nonlocal total_mb, tensor_count
            if torch.is_tensor(obj):
                tensor_count += 1
                total_mb += sizeof_tensor_mb(obj)
            elif isinstance(obj, dict):
                for vv in obj.values():
                    walk(vv)
            elif isinstance(obj, (list, tuple)):
                for vv in obj:
                    walk(vv)

        walk(v)
        candidates.append((k, tensor_count, total_mb))

    candidates.sort(key=lambda x: -x[2])

    print("Top non-state_dict parts by tensor size:")
    for k, count, mb in candidates[:30]:
        print(f"  - {k}: tensor_items={count} | {mb / 1024:.3f} GB")


def main():
    last_ckpt = safe_torch_load(LAST_PATH)
    hamer_ckpt = safe_torch_load(HAMER_PATH)

    summarize_top_level(last_ckpt, "last.ckpt", LAST_PATH)
    summarize_top_level(hamer_ckpt, "hamer.ckpt", HAMER_PATH)

    last_sd, last_sd_src = get_state_dict(last_ckpt)
    hamer_sd, hamer_sd_src = get_state_dict(hamer_ckpt)

    print("\n[STATE_DICT SOURCE]")
    print(f"last.ckpt  -> {last_sd_src}")
    print(f"hamer.ckpt -> {hamer_sd_src}")

    summarize_state_dict(last_sd, "last.ckpt")
    summarize_state_dict(hamer_sd, "hamer.ckpt")

    compare_top_level_keys(last_ckpt, "last.ckpt", hamer_ckpt, "hamer.ckpt")
    compare_state_dicts(last_sd, "last.ckpt", hamer_sd, "hamer.ckpt", show_samples=100)

    inspect_non_state_heavy_parts(last_ckpt, "last.ckpt")
    inspect_non_state_heavy_parts(hamer_ckpt, "hamer.ckpt")


if __name__ == "__main__":
    main()