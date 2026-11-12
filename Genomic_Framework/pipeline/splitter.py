"""Accession/isolate-level stratified split. Zero leakage across train/val/test."""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from Genomic_Framework.pipeline.labeling import CANONICAL_FIELDS
from Genomic_Framework.pipeline.taxonomy import INDETERMINATE_PATHOTYPE

logger = logging.getLogger(__name__)


class AccessionSplitter:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        split = config.get("split", {})
        self.train_ratio = float(split.get("train_ratio", 0.70))
        self.val_ratio = float(split.get("val_ratio", 0.15))
        self.test_ratio = float(split.get("test_ratio", 0.15))
        self.seed = int(split.get("seed", 42))
        self.group_by = split.get("group_by", "isolate_id")
        paths = config.get("paths", {})
        self.canonical_dir = Path(paths.get("canonical_dir", "dataset/canonical"))
        self.splits_dir = Path(paths.get("splits_dir", "dataset/splits"))

    def split(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in records:
            key = rec.get(self.group_by) or rec.get("insdc_accession")
            groups[key].append(rec)

        strata: Dict[str, List[str]] = defaultdict(list)
        for isolate, recs in groups.items():
            subtypes = sorted({r.get("subtype") or "NA" for r in recs})
            pathos = sorted({r.get("pathotype") or INDETERMINATE_PATHOTYPE for r in recs})
            patho = next((p for p in pathos if p != INDETERMINATE_PATHOTYPE), INDETERMINATE_PATHOTYPE)
            strata[f"{subtypes[0]}|{patho}"].append(isolate)

        rng = random.Random(self.seed)
        train_ids, val_ids, test_ids = set(), set(), set()
        for _, isolates in strata.items():
            ids = list(isolates)
            rng.shuffle(ids)
            n = len(ids)
            if n == 1:
                train_ids.update(ids)
                continue
            if n == 2:
                train_ids.add(ids[0])
                test_ids.add(ids[1])
                continue
            n_train = max(1, int(n * self.train_ratio))
            n_val = max(1, int(n * self.val_ratio))
            if n_train + n_val >= n:
                n_val = max(1, n - n_train - 1)
            train_ids.update(ids[:n_train])
            val_ids.update(ids[n_train : n_train + n_val])
            test_ids.update(ids[n_train + n_val :])

        assigned = []
        for rec in records:
            key = rec.get(self.group_by) or rec.get("insdc_accession")
            rec = dict(rec)
            if key in train_ids:
                rec["split"] = "train"
            elif key in val_ids:
                rec["split"] = "val"
            else:
                rec["split"] = "test"
            assigned.append(rec)

        leak = self.leakage_report(assigned)
        if leak["train_val"] or leak["train_test"] or leak["val_test"]:
            raise RuntimeError(f"Split leakage detected: {leak}")

        self._write(assigned, leak)
        logger.info(
            "Split isolates train=%d val=%d test=%d (records train=%d val=%d test=%d)",
            len(train_ids), len(val_ids), len(test_ids),
            sum(r["split"] == "train" for r in assigned),
            sum(r["split"] == "val" for r in assigned),
            sum(r["split"] == "test" for r in assigned),
        )
        return assigned

    @staticmethod
    def leakage_report(records: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        by_split: Dict[str, set] = defaultdict(set)
        for rec in records:
            acc = rec.get("insdc_accession")
            iso = rec.get("isolate_id")
            by_split[rec.get("split", "")].add(acc)
            by_split[f"{rec.get('split')}_iso"].add(iso)
        return {
            "train_val": sorted(by_split["train"] & by_split["val"]),
            "train_test": sorted(by_split["train"] & by_split["test"]),
            "val_test": sorted(by_split["val"] & by_split["test"]),
            "train_val_iso": sorted(by_split["train_iso"] & by_split["val_iso"]),
            "train_test_iso": sorted(by_split["train_iso"] & by_split["test_iso"]),
        }

    def _write(self, records: List[Dict[str, Any]], leak: Dict[str, Any]) -> None:
        self.canonical_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        path = self.canonical_dir / "manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CANONICAL_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        for split in ("train", "val", "test"):
            accs = sorted({r["insdc_accession"] for r in records if r["split"] == split})
            (self.splits_dir / f"{split}_accessions.txt").write_text("\n".join(accs) + "\n", encoding="utf-8")
        (self.splits_dir / "leakage_report.json").write_text(json.dumps(leak, indent=2), encoding="utf-8")
