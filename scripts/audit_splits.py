"""
Leakage Audit for VigiLens Splits.

Ensures scene-disjoint partitioning across train, val, calibration, and test splits.
VidLLVIP and surveillance datasets have extreme temporal continuity across clips from
the same scene/camera.  If frames or clips from the same scene appear in both
training/validation/calibration and test sets, model performance metrics are
invalidated by severe data leakage (Problem P8).

This script:
1. Discovers clips in train, val, calib, and test sets (from Data/, cache/, manifests).
2. Extracts scene identifiers (e.g. '01', '02', '13' or prefix).
3. Verifies zero scene overlap between {train, val/calib, test}.
4. Fails loudly with exit code 1 if any leakage is detected.
5. Documents clean splits to experiments/SPLITS.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)


class LeakageError(Exception):
    """Raised when scene leakage is detected across splits."""


def extract_scene_id(identifier: Union[str, Path]) -> str:
    """Extract canonical scene identifier from a filename or sequence ID.

    Examples:
        '01_0000_0005.mp4' -> '01'
        '13_0005_0010_rgb.mp4' -> '13'
        '09_0000_0005.pt' -> '09'
        'train_000_rgb.mp4' -> 'train'
        'normal_001.pt' -> 'normal'
    """
    stem = Path(identifier).stem
    # Remove common modality/type suffixes
    stem = re.sub(r'(_rgb|_ir|_features|_seg)$', '', stem, flags=re.IGNORECASE)
    parts = stem.split('_')
    return parts[0] if parts else stem


def discover_clips_from_directory(directory: Union[str, Path]) -> List[str]:
    """Find unique sequence IDs / clip names under a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    valid_extensions = {".mp4", ".avi", ".mkv", ".pt", ".json"}
    sequences = set()

    for file_path in dir_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            # Skip manifest files or sidecar JSONs when .pt exists
            if file_path.name in {"manifest.json", "cache_manifest.json", "results.json"}:
                continue
            stem = file_path.stem
            stem = re.sub(r'(_rgb|_ir)$', '', stem, flags=re.IGNORECASE)
            sequences.add(stem)

    return sorted(sequences)


def discover_clips_from_manifest(manifest_path: Union[str, Path]) -> List[str]:
    """Extract sequence IDs from a manifest JSON file."""
    path = Path(manifest_path)
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        logger.warning("Could not read manifest %s: %s", path, exc)
        return []

    sequences = []
    videos = data.get("videos", [])
    for entry in videos:
        seq_id = entry.get("sequence_id")
        if seq_id:
            sequences.append(seq_id)
        elif "rgb" in entry:
            stem = Path(entry["rgb"]).stem
            stem = re.sub(r'(_rgb|_ir)$', '', stem, flags=re.IGNORECASE)
            sequences.append(stem)

    return sorted(set(sequences))


def audit_splits(splits: Dict[str, Sequence[str]],
                 fail_on_overlap: bool = True) -> Dict:
    """Audit clip distributions and verify scene-level disjointness across splits.

    Args:
        splits: Dictionary mapping split names (e.g. 'train', 'val', 'calib', 'test')
                to lists of clip/sequence identifiers.
        fail_on_overlap: If True, raise LeakageError when any forbidden overlap is found.

    Returns:
        Audit report dictionary with per-split scene sets, clip counts, and overlap records.
    """
    split_scenes: Dict[str, Set[str]] = {}
    split_clips: Dict[str, List[str]] = {}
    scene_to_clips: Dict[str, Dict[str, List[str]]] = {}

    for split_name, clips in splits.items():
        unique_clips = sorted(set(clips))
        split_clips[split_name] = unique_clips
        scenes = set()
        for clip in unique_clips:
            scene_id = extract_scene_id(clip)
            scenes.add(scene_id)
            scene_to_clips.setdefault(split_name, {}).setdefault(scene_id, []).append(clip)
        split_scenes[split_name] = scenes

    # Pairwise overlap checks
    # Standard forbidden overlaps:
    # train vs test, train vs val, train vs calib, val vs test, calib vs test
    overlaps = []
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            name_a = split_names[i]
            name_b = split_names[j]
            # Val and calib can optionally share scenes if calib is sampled from val;
            # but train vs anything and test vs anything must have zero overlap.
            scenes_a = split_scenes[name_a]
            scenes_b = split_scenes[name_b]
            common = scenes_a & scenes_b
            if common:
                overlaps.append({
                    "split_a": name_a,
                    "split_b": name_b,
                    "overlapping_scenes": sorted(common),
                    "clips_a": {s: scene_to_clips[name_a][s] for s in common},
                    "clips_b": {s: scene_to_clips[name_b][s] for s in common},
                })

    has_leakage = len(overlaps) > 0
    report = {
        "passed": not has_leakage,
        "splits": {
            name: {
                "num_clips": len(split_clips[name]),
                "num_scenes": len(split_scenes[name]),
                "scenes": sorted(split_scenes[name]),
            }
            for name in split_names
        },
        "overlaps": overlaps,
    }

    if has_leakage and fail_on_overlap:
        msg_lines = ["\n[LEAKAGE AUDIT FAILED] Scene overlap detected across splits:"]
        for item in overlaps:
            msg_lines.append(
                f"  Overlap between '{item['split_a']}' and '{item['split_b']}': "
                f"scenes {item['overlapping_scenes']}"
            )
            for sc in item["overlapping_scenes"]:
                msg_lines.append(f"    Scene {sc} in {item['split_a']}: {item['clips_a'][sc][:5]}")
                msg_lines.append(f"    Scene {sc} in {item['split_b']}: {item['clips_b'][sc][:5]}")
        raise LeakageError("\n".join(msg_lines))

    return report


def generate_splits_markdown(audit_result: Dict,
                             title: str = "VigiLens Scene Partitioning & Leakage Audit",
                             notes: Optional[str] = None) -> str:
    """Format audit results as a readable Markdown document."""
    lines = [
        f"# {title}",
        "",
        f"**Audit Status**: {'PASSED (Zero Leakage)' if audit_result['passed'] else 'FAILED (Leakage Detected)'}",
        "",
    ]
    if notes:
        lines.extend([notes, ""])

    lines.append("## Split Summary")
    lines.append("")
    lines.append("| Split | Clips | Scenes | Scene IDs |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for name, info in audit_result["splits"].items():
        scenes_str = ", ".join(info["scenes"]) if info["scenes"] else "none"
        lines.append(f"| `{name}` | {info['num_clips']} | {info['num_scenes']} | `{scenes_str}` |")

    lines.append("")
    lines.append("## Pairwise Overlap Audit")
    lines.append("")
    if not audit_result["overlaps"]:
        lines.append("> [!NOTE]")
        lines.append("> All scene sets are strictly disjoint. No scene appears in more than one split.")
    else:
        lines.append("> [!CAUTION]")
        lines.append("> Overlap was found between the following splits:")
        for item in audit_result["overlaps"]:
            lines.append(f"- **{item['split_a']}** <-> **{item['split_b']}**: scenes `{item['overlapping_scenes']}`")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit VigiLens dataset splits for scene leakage")
    parser.add_argument("--data-dir", default=None, help="Path to Data directory (with train/, val/, test/)")
    parser.add_argument("--cache-dir", default=None, help="Path to cache directory (with train/, val/, eval/)")
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--calib-manifest", default=None)
    parser.add_argument("--test-manifest", default=None)
    parser.add_argument("--eval-manifest", default=None, help="Manifest for test / evaluation")
    parser.add_argument("--out-md", default="experiments/SPLITS.md", help="Path to write SPLITS.md")
    parser.add_argument("--no-fail", action="store_true", help="Do not exit with code 1 on leakage")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    splits: Dict[str, List[str]] = {}

    # Discover from data-dir or cache-dir
    if args.data_dir:
        data_path = Path(args.data_dir)
        for split in ["train", "val", "calib", "test"]:
            sub = data_path / split
            if sub.exists():
                clips = discover_clips_from_directory(sub)
                if clips:
                    splits[split] = clips

    if args.cache_dir:
        cache_path = Path(args.cache_dir)
        for split in ["train", "val", "calib", "test", "eval"]:
            sub = cache_path / split
            if sub.exists():
                clips = discover_clips_from_directory(sub)
                target_split = "test" if split == "eval" else split
                if clips:
                    splits.setdefault(target_split, []).extend(clips)

    if args.train_manifest:
        splits["train"] = discover_clips_from_manifest(args.train_manifest)
    if args.val_manifest:
        splits["val"] = discover_clips_from_manifest(args.val_manifest)
    if args.calib_manifest:
        splits["calib"] = discover_clips_from_manifest(args.calib_manifest)
    if args.test_manifest:
        splits["test"] = discover_clips_from_manifest(args.test_manifest)
    elif args.eval_manifest:
        splits["test"] = discover_clips_from_manifest(args.eval_manifest)

    # Fallback default check if nothing specified: check standard workspace locations
    if not splits:
        if Path("Data").exists():
            for s in ["train", "val", "test"]:
                if (Path("Data") / s).exists():
                    splits[s] = discover_clips_from_directory(Path("Data") / s)
        if Path("Evaluation/manifest.json").exists() and "test" not in splits:
            splits["test"] = discover_clips_from_manifest("Evaluation/manifest.json")

    if not splits:
        print("No splits found to audit. Please specify --data-dir, --cache-dir, or manifests.")
        sys.exit(0)

    try:
        result = audit_splits(splits, fail_on_overlap=not args.no_fail)
        print("\n[Audit Passed] Zero scene leakage across splits:")
        for name, info in result["splits"].items():
            print(f"  {name:6s}: {info['num_clips']} clips across {info['num_scenes']} scenes: {info['scenes']}")
    except LeakageError as err:
        print(str(err), file=sys.stderr)
        if args.out_md:
            # Still record the failing audit in markdown if requested
            result = audit_splits(splits, fail_on_overlap=False)
            md = generate_splits_markdown(result, notes="Audit detected leakage.")
            Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out_md).write_text(md, encoding="utf-8")
        sys.exit(1)

    if args.out_md:
        md = generate_splits_markdown(result)
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(md, encoding="utf-8")
        print(f"\nWrote audit report to {args.out_md}")


if __name__ == "__main__":
    main()
