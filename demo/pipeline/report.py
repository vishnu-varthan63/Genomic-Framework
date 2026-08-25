import json
import pandas as pd
from pathlib import Path
import yaml
import datetime

# Load config (for potential inclusion in report)
from ..config import load_config
config = load_config()

def build_report(input_seq: str,
                 cleaned_seq: str,
                 alignment_stats: dict,
                 mutations: dict,
                 annotations: list,
                 priorities: list,
                 prediction_probs: list,
                 shap_values: list | None = None) -> dict:
    """Assemble a complete report dictionary.
    Args:
        input_seq: raw user‑provided sequence string.
        cleaned_seq: sequence after QC.
        alignment_stats: dict from `alignment.align_sequence`.
        mutations: dict with keys 'positions' (list of floats) and 'types' (list of strings).
        annotations: list of annotation strings per mutation.
        priorities: list of priority scores per mutation.
        prediction_probs: list of class probabilities from the classifier.
        shap_values: optional list of SHAP values per class (same length as prediction_probs).
    Returns:
        dict: structured report ready for JSON/CSV/HTML export.
    """
    report = {
        "metadata": {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "config": config,
        },
        "input": {
            "raw_sequence": input_seq,
            "cleaned_sequence": cleaned_seq,
        },
        "alignment": alignment_stats,
        "mutations": {
            "positions": mutations.get("positions", []),
            "types": mutations.get("types", []),
            "annotations": annotations,
            "priorities": priorities,
        },
        "prediction": {
            "probabilities": prediction_probs,
            "class_labels": config.get("class_labels", []),
        },
    }
    if shap_values is not None:
        report["explainability"] = {"shap_values": shap_values}
    return report

def save_report(report: dict, output_dir: Path | str = None) -> Path:
    """Write the report to JSON, CSV and HTML files.
    Returns the path to the JSON file.
    """
    output_dir = Path(output_dir) if output_dir else (Path(__file__).resolve().parents[2] / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"report_{timestamp}.json"
    csv_path = output_dir / f"report_{timestamp}.csv"
    html_path = output_dir / f"report_{timestamp}.html"
    # JSON
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(report, jf, indent=2)
    # CSV – flatten mutation table and prediction
    rows = []
    mutations = report.get("mutations", {})
    n = len(mutations.get("positions", []))
    for i in range(n):
        rows.append({
            "mutation_index": i,
            "position": mutations["positions"][i],
            "type": mutations["types"][i],
            "annotation": mutations["annotations"][i] if i < len(mutations["annotations"]) else "",
            "priority": mutations["priorities"][i] if i < len(mutations["priorities"]) else "",
        })
    # Add a summary row for prediction
    pred = report.get("prediction", {})
    rows.append({
        "mutation_index": "prediction",
        "position": "",
        "type": "",
        "annotation": "",
        "priority": "",
        "probabilities": ", ".join([f"{lbl}:{p:.3f}" for lbl, p in zip(pred.get("class_labels", []), pred.get("probabilities", []))])
    })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    # Simple HTML using pandas styling
    html_content = df.to_html(index=False, justify='center')
    with open(html_path, "w", encoding="utf-8") as hf:
        hf.write(html_content)
    return json_path
