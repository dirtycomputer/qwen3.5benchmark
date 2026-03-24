#!/usr/bin/env python3
"""
Download all Qwen3.5 natural language benchmark datasets.

Usage:
    pip install datasets huggingface_hub
    python download_benchmarks.py [--output-dir ./data] [--category Knowledge] [--benchmark MMLU-Pro]

This script downloads benchmark datasets from HuggingFace and prepares them
for local evaluation.
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
except ImportError:
    print("Please install required packages:")
    print("  pip install datasets huggingface_hub")
    sys.exit(1)


# ============================================================================
# Dataset Registry: all Qwen3.5 NL benchmarks
# ============================================================================

BENCHMARKS = {
    # ── Knowledge ──────────────────────────────────────────────────────────
    "MMLU-Pro": {
        "category": "Knowledge",
        "hf_id": "TIGER-Lab/MMLU-Pro",
        "split": "test",
    },
    "MMLU-Redux": {
        "category": "Knowledge",
        "hf_id": "edinburgh-dawg/mmlu-redux",
        "split": "test",
    },
    "SuperGPQA": {
        "category": "Knowledge",
        "hf_id": "m-a-p/SuperGPQA",
    },
    "C-Eval": {
        "category": "Knowledge",
        "hf_id": "ceval/ceval-exam",
        "split": "test",
    },

    # ── Instruction Following ──────────────────────────────────────────────
    "IFEval": {
        "category": "Instruction Following",
        "hf_id": "google/IFEval",
    },
    "IFBench": {
        "category": "Instruction Following",
        "hf_id": "allenai/IFBench_test",
    },
    "MultiChallenge": {
        "category": "Instruction Following",
        "hf_id": None,
        "github": "https://github.com/ekwinox117/multi-challenge",
        "note": "Clone from GitHub: git clone https://github.com/ekwinox117/multi-challenge",
    },

    # ── Long Context ──────────────────────────────────────────────────────
    "AA-LCR": {
        "category": "Long Context",
        "hf_id": "ArtificialAnalysis/AA-LCR",
    },
    "LongBench-v2": {
        "category": "Long Context",
        "hf_id": "THUDM/LongBench-v2",
    },

    # ── STEM ──────────────────────────────────────────────────────────────
    "GPQA": {
        "category": "STEM",
        "hf_id": "Idavidrein/gpqa",
    },
    "HLE": {
        "category": "STEM",
        "hf_id": "cais/hle",
    },
    "HLE-Verified": {
        "category": "STEM",
        "hf_id": "skylenage/HLE-Verified",
    },

    # ── Reasoning ─────────────────────────────────────────────────────────
    "LiveCodeBench-v6": {
        "category": "Reasoning",
        "hf_id": "livecodebench/code_generation",
    },
    "HMMT-Feb-25": {
        "category": "Reasoning",
        "hf_id": "MathArena/hmmt_feb_2025",
    },
    "HMMT-Nov-25": {
        "category": "Reasoning",
        "hf_id": "MathArena/hmmt_nov_2025",
    },
    "IMOAnswerBench": {
        "category": "Reasoning",
        "hf_id": "OpenEvals/IMO-AnswerBench",
    },
    "AIME26": {
        "category": "Reasoning",
        "hf_id": "MathArena/aime_2026",
    },

    # ── General Agent ─────────────────────────────────────────────────────
    "BFCL-V4": {
        "category": "General Agent",
        "hf_id": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
    },
    "TAU2-Bench": {
        "category": "General Agent",
        "hf_id": None,
        "github": "https://github.com/sierra-research/tau2-bench",
        "note": "Install: pip install tau-bench; data included in package",
    },
    "VITA-Bench": {
        "category": "General Agent",
        "hf_id": None,
        "github": "https://github.com/meituan-longcat/vitabench",
        "note": "Clone from GitHub: git clone https://github.com/meituan-longcat/vitabench",
    },
    "DeepPlanning": {
        "category": "General Agent",
        "hf_id": "Qwen/DeepPlanning",
    },
    "Tool-Decathlon": {
        "category": "General Agent",
        "hf_id": "hkust-nlp/Toolathlon-Trajectories",
        "github": "https://github.com/hkust-nlp/Toolathlon",
    },
    "MCP-Mark": {
        "category": "General Agent",
        "hf_id": None,
        "github": "https://github.com/eval-sys/mcpmark",
        "note": "Clone from GitHub: git clone https://github.com/eval-sys/mcpmark",
    },

    # ── Search Agent ──────────────────────────────────────────────────────
    "HLE-w-tool": {
        "category": "Search Agent",
        "hf_id": "cais/hle",
        "note": "Same dataset as HLE, evaluated with search/tool augmentation",
    },
    "BrowseComp": {
        "category": "Search Agent",
        "hf_id": None,
        "github": "https://github.com/openai/simple-evals",
        "note": "Inside openai/simple-evals repo, browsecomp directory",
    },
    "BrowseComp-zh": {
        "category": "Search Agent",
        "hf_id": "PALIN2018/BrowseComp-ZH",
    },
    "WideSearch": {
        "category": "Search Agent",
        "hf_id": "ByteDance-Seed/WideSearch",
    },
    "Seal-0": {
        "category": "Search Agent",
        "hf_id": "vtllms/sealqa",
    },

    # ── Multilingualism ───────────────────────────────────────────────────
    "MMMLU": {
        "category": "Multilingualism",
        "hf_id": "openai/MMMLU",
    },
    "MMLU-ProX": {
        "category": "Multilingualism",
        "hf_id": "li-lab/MMLU-ProX",
    },
    "NOVA-63": {
        "category": "Multilingualism",
        "hf_id": "zjy1298/NOVA-63",
    },
    "INCLUDE": {
        "category": "Multilingualism",
        "hf_id": None,
        "github": "https://github.com/nlp-uoregon/mlmm-evaluation",
        "note": "Clone from GitHub and follow setup instructions",
    },
    "Global-PIQA": {
        "category": "Multilingualism",
        "hf_id": "mrlbenchmarks/global-piqa-nonparallel",
    },
    "PolyMATH": {
        "category": "Multilingualism",
        "hf_id": "Qwen/PolyMath",
    },
    "WMT24++": {
        "category": "Multilingualism",
        "hf_id": "google/wmt24pp",
    },
    "MAXIFE": {
        "category": "Multilingualism",
        "hf_id": None,
        "url": "https://arxiv.org/abs/2506.01776",
        "note": "Check paper for official dataset release",
    },

    # ── Coding Agent ──────────────────────────────────────────────────────
    "SWE-bench-Verified": {
        "category": "Coding Agent",
        "hf_id": "SWE-bench/SWE-bench_Verified",
    },
    "SWE-bench-Multilingual": {
        "category": "Coding Agent",
        "hf_id": None,
        "github": "https://github.com/multi-swe-bench/multi-swe-bench",
        "note": "Clone from GitHub and follow setup instructions",
    },
    "SecCodeBench": {
        "category": "Coding Agent",
        "hf_id": None,
        "github": "https://github.com/alibaba/sec-code-bench",
        "note": "Clone from GitHub: git clone https://github.com/alibaba/sec-code-bench",
    },
    "Terminal-Bench-2": {
        "category": "Coding Agent",
        "hf_id": "zai-org/terminal-bench-2-verified",
    },
}


def download_hf_dataset(name: str, info: dict, output_dir: Path) -> bool:
    """Download a single HuggingFace dataset and save as JSONL."""
    hf_id = info["hf_id"]
    category = info["category"]
    save_dir = output_dir / category.replace(" ", "_") / name

    if save_dir.exists() and any(save_dir.iterdir()):
        print(f"  [SKIP] {name} already exists at {save_dir}")
        return True

    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [DOWN] {name} <- {hf_id}")
    try:
        split = info.get("split")
        if split:
            ds = load_dataset(hf_id, split=split, trust_remote_code=True)
            out_path = save_dir / f"{split}.jsonl"
            ds.to_json(out_path)
            print(f"         Saved {len(ds)} samples -> {out_path}")
        else:
            ds = load_dataset(hf_id, trust_remote_code=True)
            for split_name, split_ds in ds.items():
                out_path = save_dir / f"{split_name}.jsonl"
                split_ds.to_json(out_path)
                print(f"         Saved {split_name}: {len(split_ds)} samples -> {out_path}")
    except Exception as e:
        # Fallback: try snapshot_download for non-standard datasets
        print(f"  [WARN] load_dataset failed for {name}: {e}")
        print(f"  [RETRY] Trying snapshot_download...")
        try:
            local_path = snapshot_download(
                repo_id=hf_id,
                repo_type="dataset",
                local_dir=save_dir / "raw",
            )
            print(f"         Downloaded snapshot -> {local_path}")
        except Exception as e2:
            print(f"  [FAIL] {name}: {e2}")
            # Write error info
            (save_dir / "DOWNLOAD_FAILED.txt").write_text(
                f"Dataset: {hf_id}\nError (load_dataset): {e}\nError (snapshot): {e2}\n"
            )
            return False
    return True


def clone_github_repo(name: str, info: dict, output_dir: Path) -> bool:
    """Clone a GitHub repo for datasets not on HuggingFace."""
    github_url = info.get("github")
    if not github_url:
        return False

    category = info["category"]
    save_dir = output_dir / category.replace(" ", "_") / name

    if save_dir.exists() and any(save_dir.iterdir()):
        print(f"  [SKIP] {name} already exists at {save_dir}")
        return True

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [CLONE] {name} <- {github_url}")
    ret = os.system(f"git clone --depth 1 {github_url} {save_dir / 'repo'}")
    if ret != 0:
        print(f"  [FAIL] git clone failed for {name}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Download Qwen3.5 NL benchmark datasets")
    parser.add_argument("--output-dir", "-o", default="./data", help="Output directory (default: ./data)")
    parser.add_argument("--category", "-c", default=None, help="Only download a specific category")
    parser.add_argument("--benchmark", "-b", default=None, help="Only download a specific benchmark")
    parser.add_argument("--list", "-l", action="store_true", help="List all benchmarks and exit")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Filter benchmarks
    targets = {}
    for name, info in BENCHMARKS.items():
        if args.benchmark and name != args.benchmark:
            continue
        if args.category and info["category"] != args.category:
            continue
        targets[name] = info

    if args.list:
        print(f"\n{'Benchmark':<25} {'Category':<25} {'Source':<45}")
        print("=" * 95)
        for name, info in sorted(targets.items(), key=lambda x: x[1]["category"]):
            src = info.get("hf_id") or info.get("github", "N/A")
            print(f"{name:<25} {info['category']:<25} {src:<45}")
        print(f"\nTotal: {len(targets)} benchmarks")

        hf_count = sum(1 for v in targets.values() if v.get("hf_id"))
        gh_count = sum(1 for v in targets.values() if not v.get("hf_id") and v.get("github"))
        manual_count = sum(1 for v in targets.values() if not v.get("hf_id") and not v.get("github"))
        print(f"  HuggingFace: {hf_count} | GitHub: {gh_count} | Manual: {manual_count}")
        return

    if args.dry_run:
        print("\n[DRY RUN] Would download the following:\n")
        for name, info in targets.items():
            if info.get("hf_id"):
                print(f"  HF: {name} <- {info['hf_id']}")
            elif info.get("github"):
                print(f"  GH: {name} <- {info['github']}")
            else:
                print(f"  ??: {name} <- MANUAL ({info.get('note', 'no info')})")
        return

    # Download
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(targets)
    success = 0
    failed = []
    manual = []

    print(f"\nDownloading {total} benchmarks to {output_dir.resolve()}\n")
    print("=" * 70)

    for i, (name, info) in enumerate(targets.items(), 1):
        print(f"\n[{i}/{total}] {info['category']} / {name}")

        if info.get("hf_id"):
            if download_hf_dataset(name, info, output_dir):
                success += 1
            else:
                failed.append(name)
        elif info.get("github"):
            if clone_github_repo(name, info, output_dir):
                success += 1
            else:
                failed.append(name)
        else:
            note = info.get("note", "No download source available")
            print(f"  [MANUAL] {name}: {note}")
            manual.append((name, note))

    # Summary
    print("\n" + "=" * 70)
    print(f"\nDONE: {success}/{total} downloaded successfully")
    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")
    if manual:
        print(f"\nMANUAL DOWNLOAD REQUIRED ({len(manual)}):")
        for name, note in manual:
            print(f"  - {name}: {note}")

    # Save download manifest
    manifest = {
        "output_dir": str(output_dir.resolve()),
        "total": total,
        "success": success,
        "failed": failed,
        "manual": [{"name": n, "note": t} for n, t in manual],
    }
    manifest_path = output_dir / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
