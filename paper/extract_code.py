"""Copy selected functions from src/ into paper/code/ for the code annex (keeps the paper in sync)."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "volforecast"
OUT = Path(__file__).resolve().parent / "code"
OUT.mkdir(exist_ok=True)
PICK = {
    "data_pipeline.py": ["_remove_bad_ticks", "liquidity_groups", "split_by_target_date"],
    "har.py": ["_fit_predict"],
    "egarch.py": ["egarch_features"],
    "lstm.py": ["VolatilityLSTM", "QLIKELoss", "train_one_seed"],
    "evaluation.py": ["dm_test"],
}
for fname, names in PICK.items():
    text = (SRC / fname).read_text()
    lines = text.splitlines()
    for node in ast.parse(text).body:
        if getattr(node, "name", None) in names:
            start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
            (OUT / f"{node.name.strip('_')}.py").write_text("\n".join(lines[start:node.end_lineno]) + "\n")
print(sorted(p.name for p in OUT.glob("*.py")))
