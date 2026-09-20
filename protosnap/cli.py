"""One entry point for everything.

    protosnap <command> [options]          (or: python -m protosnap <command> [options])

Every command forwards its options to the module that implements it, so
`protosnap <command> --help` shows that command's own options.
"""
from __future__ import annotations

import importlib
import sys

COMMANDS: dict[str, tuple[str, str]] = {
    # name            (module,                        one-line description)
    "reproduce":      ("protosnap.reproduce",         "verify the data, then re-score the baseline and the shipped models against recorded values"),
    "infer":          ("protosnap.stage5:run",        "generate skeletons for every glyph that has none (writes generated/)"),
    "editor":         ("protosnap.editor",            "review and correct generated skeletons in the browser"),
    "acceptance":     ("protosnap.stage5:acceptance", "acceptance rate of a filled-in review CSV"),
    "promote":        ("protosnap.stage5:promote",    "publish accepted and corrected skeletons into skeletons/"),
    "train":          ("protosnap.learned.train",     "train the wedge detector"),
    "compare":        ("protosnap.retrain_report",    "compare groups of checkpoints over seeds"),
    "format":         ("protosnap.skeleton_io",       "check or convert skeleton CSVs to the original annotation layout"),
    "correct":        ("protosnap.corrections",       "repair an original annotation without modifying it"),
    "charts":         ("protosnap.charts",            "redraw the bar charts from saved results"),
    "tune":           ("protosnap.tune",              "tune the classical baseline"),
    "stage0":         ("protosnap.stage0",            "data audit: splits, stats, contact sheets"),
    "stage1":         ("protosnap.stage1",            "evaluation harness reference scores"),
    "stage2":         ("protosnap.stage2",            "classical baseline report"),
    "stage3":         ("protosnap.stage3",            "learned model report"),
    "stage4":         ("protosnap.stage4",            "hybrid ablation report"),
}
# commands implemented as a subcommand of another module's parser
SUBCOMMAND = {"infer": "run", "acceptance": "acceptance", "promote": "promote"}


def usage() -> str:
    width = max(len(c) for c in COMMANDS)
    lines = [__doc__.strip(), "", "commands:"]
    lines += [f"  {name:<{width}}  {desc}" for name, (_, desc) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"unknown command {name!r}\n\n{usage()}", file=sys.stderr)
        return 2
    module = COMMANDS[name][0].split(":")[0]
    if name in SUBCOMMAND:
        rest = [SUBCOMMAND[name], *rest]
    sys.argv = [f"protosnap {name}", *rest]          # so each module's --help names the command correctly
    return int(importlib.import_module(module).main(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
