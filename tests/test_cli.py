import json
import subprocess
import sys
from pathlib import Path

import pytest

from protosnap import cli

ROOT = Path(__file__).resolve().parents[1]


def test_help_lists_every_command(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for name in cli.COMMANDS:
        assert name in out
    assert cli.main(["no-such-command"]) == 2


def test_every_command_resolves_to_a_main():
    import importlib
    for name, (target, _) in cli.COMMANDS.items():
        mod = importlib.import_module(target.split(":")[0])
        assert callable(getattr(mod, "main")), name


def test_module_entry_point_and_subcommand_forwarding():
    r = subprocess.run([sys.executable, "-m", "protosnap", "format", "check", "data/reviewed"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "conform" in r.stdout
    r = subprocess.run([sys.executable, "-m", "protosnap", "promote", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "--dry-run" in r.stdout


def test_reproduce_compare_logic():
    from protosnap.reproduce import compare
    base = {"precision": 0.95, "recall": 0.97, "f1": 0.96, "corner_error": 3.0, "tail": 5.0, "exact": 0.4, "glyphs": 45}
    exp = {"scores": {"classical": {"test": dict(base)}, "hybrid:m.pt": {"test": dict(base)}}}
    same = json.loads(json.dumps(exp))
    lines, failed, warned = compare(exp, same)
    assert not failed and not warned and all(l.startswith("PASS") for l in lines)
    drift = json.loads(json.dumps(exp))
    drift["scores"]["hybrid:m.pt"]["test"]["exact"] = 0.4 + 1 / 45          # one glyph: warning, not failure
    assert compare(exp, drift)[1:] == (False, True)
    drift["scores"]["hybrid:m.pt"]["test"]["exact"] = 0.4 + 3 / 45
    assert compare(exp, drift)[1] is True
    drift = json.loads(json.dumps(exp))
    drift["scores"]["classical"]["test"]["f1"] = 0.9601                      # the deterministic baseline must match exactly
    assert compare(exp, drift)[1] is True


@pytest.mark.skipif(not (ROOT / "models" / "expected_scores.json").exists(), reason="no recorded scores")
def test_data_fingerprint_matches_the_recorded_one():
    from protosnap.reproduce import fingerprint
    from protosnap.split import load_splits
    exp = json.loads((ROOT / "models" / "expected_scores.json").read_text())
    assert fingerprint(ROOT, load_splits(ROOT / "data" / "splits.json")) == exp["fingerprint"]
