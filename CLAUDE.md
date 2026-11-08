# Contact-free stomatal conductance sensor (soybean, ISEF Plant Sciences)

Owner is a beginner programmer and a high-school researcher. Keep chat replies short.
Put explanations in the docs files named below, in plain language.

## Environment
- Target Python: 3.13 (Pi: 3.13.5 on Debian 13 trixie, aarch64; laptop: 3.13.12). Write syntax and use packages that run on 3.13. Run everything with .venv\Scripts\python.
- Code is written and tested on the laptop. Only files the spec marks [PI] may import hardware libraries
  (smbus2, adafruit_*, picamera2, RPi.*). Everything else must import and run on a laptop.
- You cannot run Pi code or touch sensors. Never claim a hardware step works; say it is untested until the owner pastes output.
- Run everything with .venv\Scripts\python (for example .venv\Scripts\python -m pytest -q --tb=short).

## Source of truth
- The plan is in docs/spec/. Read ONLY the files and task IDs I name, plus 02_layout_conventions.md and 08_appendix_A_equations.md when physics is involved.
- Do only the named task. Never edit anything in docs/spec/.
- If the spec looks wrong, unphysical or contradicts itself, STOP and tell me before coding.
- Never change a constant, config value or equation silently. Log every change in docs/decision_log.md.

## Code rules
- Constants and thresholds come from config.yaml, never hard-coded inside functions.
- Write tests first, then code. Run `pytest -q --tb=short`. A task is not done until tests pass.
- Hardware tests are marked `@pytest.mark.hardware` and skipped by default.
- Never edit or delete raw data files. Never invent a citation, DOI or coefficient: put it in docs/sources.md as UNVERIFIED.
- Derive, do not recall: for any physical coefficient, show the derivation and a units check in the explainer.
- Prefer small pure functions with type hints, explicit units in names (for example temp_c, rn_w_m2), and clear error messages.

## Documentation per task (research-paper material)
- Add one row to docs/decision_log.md: date | task ID | decision | why | alternatives rejected | assumptions | spec IDs.
- For modules with science or non-obvious logic, write docs/explainers/<module>.md from _TEMPLATE.md: what it does, how it works step by step, why, a worked numeric example, a draft Methods paragraph, and Limitations. Write for a beginner.
- Docstrings state the equation and units. Comments explain why, not what.
- Software versions: tell me to run `python scripts/snapshot_versions.py --label laptop` at the end of the first session and whenever dependencies change.

## Hardware tasks
- If a task needs the Pi, sensors, camera, wind sensor or reference cards, start your reply with the line REQUIRES HARDWARE.
- Then write docs/hardware/<task>.md from _TEMPLATE.md: parts needed, wiring or setup steps with pin names, exact commands to run on the Pi, what a good result looks like, and what failure looks like with the first thing to check.

## Finish every task with
1. Files created or changed. 2. Test results. 3. Assumptions and open questions. 4. Git commands for me to run (git add, git commit -m). Do NOT run git commit or git push yourself.
