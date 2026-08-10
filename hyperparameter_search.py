"""
hyperparameter_search.py
========================
Configurable Grid / Random Hyperparameter Search

Tunes:
  - learning_rate
  - batch_size
  - dropout
  - weight_decay
  - scheduler
  - warmup_epochs
  - gradient_clipping

Each trial trains for a fixed number of epochs and reports val_f1 as the
primary selection metric. Best config is saved as YAML for direct use.

Usage:
    # Random search (10 trials, 5 epochs each)
    python hyperparameter_search.py --mode random --n-trials 10 --search-epochs 5

    # Grid search over specific params
    python hyperparameter_search.py --mode grid --search-epochs 5

    # Load from config and run
    python hyperparameter_search.py --config configs/train_config.yaml --mode random --n-trials 5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("hparam_search")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from Genomic_Framework.pipeline.preprocessor import (
        GenomicDatasetBuilder,
        CLASS_LABELS,
        NUM_CLASSES,
    )
    from Genomic_Framework.engine import GenomicClassificationEngine
    from Genomic_Framework.training.advanced_trainer import AdvancedTrainer, set_seed, detect_device
    PROJECT_OK = True
except ImportError as e:
    logger.warning("Project modules not fully importable: %s. Running in DRY-RUN mode.", e)
    PROJECT_OK = False


# ---------------------------------------------------------------------------
# Search Space Definitions
# ---------------------------------------------------------------------------

GRID_SEARCH_SPACE: Dict[str, List[Any]] = {
    "learning_rate":       [1e-5, 5e-5, 1e-4, 5e-4],
    "batch_size":          [8, 16, 32],
    "dropout":             [0.1, 0.2, 0.3],
    "weight_decay":        [0.001, 0.01, 0.05],
    "scheduler":           ["cosine", "plateau"],
    "warmup_epochs":       [2, 5, 8],
    "gradient_clipping":   [1.0, 5.0],
}

RANDOM_SEARCH_SPACE: Dict[str, Any] = {
    "learning_rate":       ("log_uniform", 1e-5, 1e-3),
    "batch_size":          ("choice", [8, 16, 32]),
    "dropout":             ("uniform", 0.05, 0.40),
    "weight_decay":        ("log_uniform", 1e-4, 0.1),
    "scheduler":           ("choice", ["cosine", "plateau"]),
    "warmup_epochs":       ("choice", [0, 2, 3, 5, 8]),
    "gradient_clipping":   ("choice", [0.5, 1.0, 2.0, 5.0]),
}


def _sample_random_config(rng: random.Random) -> Dict[str, Any]:
    """Sample one random hyperparameter configuration."""
    config = {}
    for param, spec in RANDOM_SEARCH_SPACE.items():
        dist = spec[0]
        if dist == "choice":
            config[param] = rng.choice(spec[1])
        elif dist == "uniform":
            config[param] = rng.uniform(spec[1], spec[2])
        elif dist == "log_uniform":
            import math
            lo, hi = math.log(spec[1]), math.log(spec[2])
            config[param] = math.exp(rng.uniform(lo, hi))
        else:
            config[param] = spec[1]
    return config


def _grid_search_configs() -> List[Dict[str, Any]]:
    """Generate all grid search combinations."""
    keys = list(GRID_SEARCH_SPACE.keys())
    values = list(GRID_SEARCH_SPACE.values())
    configs = []
    for combo in product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


def _build_trial_config(base_config: dict, hparams: Dict[str, Any]) -> dict:
    """Merge hyperparameters into a training config dict."""
    import copy
    cfg = copy.deepcopy(base_config)

    cfg.setdefault("optimization", {})["learning_rate"]              = hparams["learning_rate"]
    cfg.setdefault("optimization", {})["weight_decay"]               = hparams["weight_decay"]
    cfg.setdefault("optimization", {})["gradient_clipping"]          = hparams["gradient_clipping"]
    cfg.setdefault("scheduler",    {})["name"]                       = hparams["scheduler"]
    cfg.setdefault("scheduler",    {})["warmup_epochs"]              = hparams["warmup_epochs"]
    cfg.setdefault("model",        {})["dropout"]                    = hparams["dropout"]
    cfg.setdefault("data",         {})["batch_size"]                 = hparams["batch_size"]

    return cfg


def run_trial(
    trial_id: int,
    hparams: Dict[str, Any],
    base_config: dict,
    search_epochs: int,
    seed: int,
    dataset_root: Path,
) -> Dict[str, Any]:
    """Run a single hyperparameter trial and return result dict."""
    logger.info("=" * 60)
    logger.info("TRIAL %03d STARTING", trial_id)
    for k, v in hparams.items():
        logger.info("  %-22s = %s", k, v)
    logger.info("=" * 60)

    trial_start = time.time()
    result = {
        "trial_id": trial_id,
        "hparams": hparams,
        "val_f1": None,
        "val_accuracy": None,
        "val_loss": None,
        "val_balanced_accuracy": None,
        "best_epoch": None,
        "duration_sec": None,
        "status": "FAILED",
        "error": None,
    }

    if not PROJECT_OK:
        # Dry run — simulate result
        val_f1 = random.uniform(0.3, 0.9)
        result.update({
            "val_f1": round(val_f1, 4),
            "val_accuracy": round(random.uniform(0.5, 0.9), 4),
            "val_loss": round(random.uniform(0.3, 1.5), 4),
            "val_balanced_accuracy": round(random.uniform(0.4, 0.85), 4),
            "best_epoch": random.randint(3, search_epochs),
            "duration_sec": round(random.uniform(30, 120), 1),
            "status": "DRY_RUN",
        })
        logger.info("DRY-RUN trial %03d completed. Simulated val_f1=%.4f", trial_id, val_f1)
        return result

    try:
        trial_config = _build_trial_config(base_config, hparams)
        trial_config["training"]["max_epochs"] = search_epochs
        trial_config["training"]["min_epochs"] = max(1, search_epochs // 3)

        set_seed(seed + trial_id)
        device = detect_device(trial_config.get("system", {}).get("device", "auto"))

        # Build dataloaders
        batch_size = int(hparams["batch_size"])
        builder = GenomicDatasetBuilder(
            dataset_root=dataset_root,
            seq_len=trial_config.get("model", {}).get("seq_len", 128),
            batch_size=batch_size,
            seed=seed + trial_id,
            stratified=trial_config.get("data", {}).get("stratified", True),
        )
        train_loader, val_loader, test_loader = builder.build_dataloaders()
        class_weights = builder.class_weights  # exposed as property after build_dataloaders()

        # Build model
        model_cfg = trial_config.get("model", {})
        model = GenomicClassificationEngine(
            seq_len=model_cfg.get("seq_len", 128),
            vocab_size=model_cfg.get("vocab_size", 4096),
            hyena_dim=model_cfg.get("hyena_dim", 256),
            hidden_dim=model_cfg.get("hidden_dim", 512),
            num_classes=NUM_CLASSES,
            max_mutations=model_cfg.get("max_mutations", 32),
            mutation_dim=model_cfg.get("mutation_dim", 64),
            dropout=float(hparams["dropout"]),
        )

        trainer = AdvancedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=trial_config,
            class_weights=class_weights,
        )

        history = trainer.train()

        val_metrics_final = history.get("test_metrics", {})
        best_val_f1 = max(history.get("val_f1", [0.0]))
        result.update({
            "val_f1": round(best_val_f1, 4),
            "val_accuracy": round(max(history.get("val_accuracy", [0.0])), 4),
            "val_loss": round(min(history.get("val_loss", [9.9])), 4),
            "val_balanced_accuracy": None,  # tracked if available in future
            "best_epoch": history.get("best_epoch", -1),
            "duration_sec": round(time.time() - trial_start, 1),
            "status": "SUCCESS",
        })

        logger.info("Trial %03d COMPLETED. Best val_f1=%.4f at epoch %s",
                    trial_id, best_val_f1, history.get("best_epoch"))

    except Exception as exc:
        result["error"] = str(exc)
        result["duration_sec"] = round(time.time() - trial_start, 1)
        logger.error("Trial %03d FAILED: %s", trial_id, exc)

    return result


def save_results(results: List[Dict[str, Any]], output_dir: Path) -> None:
    """Save all trial results to CSV, JSON, and best config YAML."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by val_f1 descending (None goes last)
    sorted_results = sorted(
        results,
        key=lambda r: r["val_f1"] if r["val_f1"] is not None else -1.0,
        reverse=True,
    )
    best = sorted_results[0] if sorted_results else None

    # 1. JSON
    json_path = output_dir / "all_trials.json"
    json_path.write_text(json.dumps(sorted_results, indent=2, default=str), encoding="utf-8")
    logger.info("Saved all trials JSON to %s", json_path)

    # 2. CSV comparison table
    csv_path = output_dir / "trial_comparison.csv"
    if results:
        fieldnames = ["trial_id", "status", "val_f1", "val_accuracy", "val_loss",
                      "best_epoch", "duration_sec"] + list(results[0]["hparams"].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted_results:
                row = {
                    "trial_id": r["trial_id"],
                    "status": r["status"],
                    "val_f1": r["val_f1"],
                    "val_accuracy": r["val_accuracy"],
                    "val_loss": r["val_loss"],
                    "best_epoch": r["best_epoch"],
                    "duration_sec": r["duration_sec"],
                }
                row.update(r.get("hparams", {}))
                writer.writerow(row)
        logger.info("Saved trial comparison CSV to %s", csv_path)

    # 3. Best config YAML
    if best and best["hparams"]:
        best_yaml_path = output_dir / "best_config.yaml"
        best_yaml_path.write_text(
            yaml.dump({"best_trial": best["trial_id"], "val_f1": best["val_f1"],
                       "hyperparameters": best["hparams"]}, default_flow_style=False),
            encoding="utf-8",
        )
        logger.info("Saved best config YAML to %s", best_yaml_path)
        logger.info("BEST TRIAL %03d | val_f1=%.4f | hparams=%s",
                    best["trial_id"], best["val_f1"] or 0, best["hparams"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hyperparameter Search for Genomic Sequence Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"),
                        help="Base training config YAML file.")
    parser.add_argument("--mode", choices=["random", "grid"], default="random",
                        help="Search mode: 'random' or 'grid'.")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Number of trials for random search.")
    parser.add_argument("--search-epochs", type=int, default=5,
                        help="Number of training epochs per trial.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset_augmented"),
                        help="Dataset root directory.")
    parser.add_argument("--output", type=Path, default=Path("hparam_results"),
                        help="Output directory for search results.")
    args = parser.parse_args()

    # Load base config
    base_config: dict = {}
    if args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f) or {}
        logger.info("Loaded base config from %s", args.config)
    else:
        logger.warning("Config file not found: %s. Using empty base config.", args.config)

    # Build trial list
    rng = random.Random(args.seed)
    if args.mode == "random":
        trial_hparams = [_sample_random_config(rng) for _ in range(args.n_trials)]
        logger.info("Random search: %d trials.", args.n_trials)
    else:
        trial_hparams = _grid_search_configs()
        logger.info("Grid search: %d total combinations.", len(trial_hparams))

    # Run trials
    all_results: List[Dict[str, Any]] = []
    for trial_id, hparams in enumerate(trial_hparams, 1):
        result = run_trial(
            trial_id=trial_id,
            hparams=hparams,
            base_config=base_config,
            search_epochs=args.search_epochs,
            seed=args.seed,
            dataset_root=args.dataset_root,
        )
        all_results.append(result)
        # Save incremental results after each trial
        save_results(all_results, args.output)

    logger.info("All %d trials completed.", len(all_results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
