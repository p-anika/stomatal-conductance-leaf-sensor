"""T0.6: docs skeleton exists and stays in step with the spec."""
import re
from pathlib import Path

DOCS = ["equations.md", "assumptions_and_limitations.md", "measurement_protocol.md",
        "pi_setup.md", "design_notes.md"]


def test_docs_exist(repo_root: Path) -> None:
    missing = [d for d in DOCS if not (repo_root / "docs" / d).is_file()]
    assert missing == []


def test_equations_is_copy_of_appendix_a(repo_root: Path) -> None:
    spec = (repo_root / "docs/spec/08_appendix_A_equations.md").read_text(encoding="utf-8")
    copy = (repo_root / "docs/equations.md").read_text(encoding="utf-8")
    # equations.md may add a header note above the copied text, nothing else
    assert copy.endswith(spec)


def test_assumptions_register_has_a1_to_a17(repo_root: Path) -> None:
    text = (repo_root / "docs/assumptions_and_limitations.md").read_text(encoding="utf-8")
    found = set(re.findall(r"^\| (A\d+) \|", text, flags=re.MULTILINE))
    assert {f"A{i}" for i in range(1, 18)} <= found


def test_readme_has_status_table_and_commands(repo_root: Path) -> None:
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "pytest -q" in text
    assert "Untested on hardware" in text
