"""Static and generated-file checks used by the paper build."""
from __future__ import annotations

import ast
import glob
import os
import re
import sys


HERE = os.path.dirname(os.path.abspath(__file__))


def check_labels():
    for path in glob.glob(os.path.join(HERE, "paper", "generated", "en", "*.tex")) + \
            glob.glob(os.path.join(HERE, "paper", "generated", "gr", "*.tex")):
        labels = re.findall(r"\\label\{(tab:[^}]+)\}", open(path, encoding="utf-8").read())
        if len(labels) != len(set(labels)):
            raise SystemExit(f"duplicate table label in {path}")


def check_writer_targets():
    path = os.path.join(HERE, "make_paper_assets.py")
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    targets = {}
    for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
        for node in ast.walk(function):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "write" or not node.args:
                continue
            names = [x.value for x in ast.walk(node.args[0])
                     if isinstance(x, ast.Constant) and isinstance(x.value, str)]
            if not names:
                continue
            target = names[-1]
            targets.setdefault(target, []).append((function.name, node.lineno))
    duplicate = {target: calls for target, calls in targets.items() if len(calls) != 1}
    if duplicate:
        raise SystemExit(f"paper asset writer collision: {duplicate}")


if __name__ == "__main__":
    check_labels()
    check_writer_targets()
    print("build checks passed")
