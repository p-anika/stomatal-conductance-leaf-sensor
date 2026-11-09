"""Tests for acquisition/session_metadata.py and processing/data_io.py (T2.1-T2.3).

Owner decisions (2026-11-08) tested here:
- Appendix B columns for both files; the 11 T2.2 card columns are required, the 8 geometry
  columns optional.
- No migration: a session_metadata.csv with an old or unknown header is never rewritten;
  appending is refused with a message telling the user to move it aside.
- position and leaf_position are normalised to lowercase_with_underscores.
- dry_reference_type = card with a card_id missing from the manifest gives a warning only.
Every file is written inside pytest's tmp_path; data/raw is never touched.
"""
import csv
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from acquisition import session_metadata as sm
from processing.config import load_config
from processing.data_io import (
    CARD_OPTIONAL_COLUMNS,
    CARD_REQUIRED_COLUMNS,
    ReferenceCardError,
    load_reference_cards,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

APPENDIX_B_SESSION_COLUMNS = [
    "session_id", "timestamp_start", "plant_id", "genotype", "water_treatment",
    "reproductive_stage", "block", "position", "leaf_position", "light_source",
    "lamp_distance_mm", "window_blocked", "acclimation_minutes", "fan_state",
    "working_distance_mm", "incidence_angle_deg", "card_angle_deg", "lamp_angle_deg",
    "lamp_to_card_mm", "background", "card_id", "dry_reference_type",
    "wet_reference_present", "operator", "porometer_gs_mol", "notes",
]
APPENDIX_B_CARD_COLUMNS = [
    "card_id", "shape_group", "source", "material", "thickness_mm", "coating", "a_s_card",
    "emissivity", "area_mm2", "d_min_mm", "d_eq_mm", "aspect_ratio", "eccentricity",
    "solidity", "circularity", "edge_density", "geometry_source", "photo_path", "notes",
]
NOW = datetime(2026, 11, 8, 14, 30, 5, tzinfo=timezone(timedelta(hours=-5)))


@pytest.fixture
def cfg():
    return load_config(REPO_ROOT / "config.yaml")


def raw(**kw):
    """A complete, valid set of operator answers (strings, as typed or passed as flags)."""
    base = dict(plant_id="HK-03", genotype="Houjaku_Kuwazu", water_treatment="well_watered",
                reproductive_stage="R2", block="1", position="Bench A-3",
                leaf_position="Top Trifoliate", light_source="led_lamp",
                lamp_distance_mm="300", window_blocked="yes", acclimation_minutes="35",
                fan_state="low", working_distance_mm="200", incidence_angle_deg="10",
                card_angle_deg="10", lamp_angle_deg="0", lamp_to_card_mm="310",
                background="neutral_matte", card_id="card01", dry_reference_type="card",
                wet_reference_present="true", operator="AP", porometer_gs_mol="",
                notes="first test session")
    base.update(kw)
    return base


def write_manifest(path, rows=(), columns=APPENDIX_B_CARD_COLUMNS):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def card_row(**kw):
    base = dict(card_id="card01", shape_group="leaflet", source="traced_leaf", material="card stock",
                thickness_mm="0.3", coating="matte black", a_s_card="", emissivity="0.95",
                area_mm2="", photo_path="", notes="")
    base.update(kw)
    return base


# ---------- schema ----------

def test_session_columns_match_appendix_b():
    assert sm.FIELDNAMES == APPENDIX_B_SESSION_COLUMNS


def test_card_columns_match_appendix_b():
    assert list(CARD_REQUIRED_COLUMNS) + list(CARD_OPTIONAL_COLUMNS) != []
    assert set(CARD_REQUIRED_COLUMNS) | set(CARD_OPTIONAL_COLUMNS) == set(APPENDIX_B_CARD_COLUMNS)
    assert set(CARD_REQUIRED_COLUMNS) == {
        "card_id", "shape_group", "source", "material", "thickness_mm", "coating", "a_s_card",
        "emissivity", "area_mm2", "photo_path", "notes"}


def test_repo_manifest_is_header_only_and_loads():
    path = REPO_ROOT / "data" / "reference_cards.csv"
    with open(path, encoding="utf-8") as f:
        assert next(csv.reader(f)) == APPENDIX_B_CARD_COLUMNS
    assert load_reference_cards(path) == {}


def test_config_enumerations(cfg):
    e = cfg["enumerations"]
    assert e["reproductive_stage"] == ["VE", "VC"] + [f"V{n}" for n in range(1, 16)] + [f"R{n}" for n in range(1, 9)]
    assert e["background"] == ["neutral_matte", "other"]
    assert e["dry_reference_type"] == ["none", "card", "vaseline_leaf"]
    # YAML 1.1 reads a bare off/no/yes as a boolean; every choice must be a string.
    assert e["fan_state"] == ["off", "low", "high"]
    assert all(isinstance(v, str) for values in e.values() for v in values)


# ---------- validation: good values ----------

def test_valid_session_types_and_normalisation(cfg):
    s = sm.make_session(raw(), cfg, now=NOW)
    assert s.block == 1 and s.lamp_distance_mm == 300.0 and s.acclimation_minutes == 35.0
    assert s.window_blocked is True and s.wet_reference_present is True
    assert s.position == "bench_a_3" and s.leaf_position == "top_trifoliate"
    assert s.porometer_gs_mol is None
    assert s.timestamp_start == "2026-11-08T14:30:05-05:00"
    assert s.session_id == "HK-03_2026-11-08T14-30-05-05-00"


def test_timestamp_must_have_timezone(cfg):
    with pytest.raises(ValueError, match="time zone"):
        sm.make_session(raw(), cfg, now=datetime(2026, 11, 8, 14, 30))


def test_default_timestamp_has_timezone(cfg):
    s = sm.make_session(raw(), cfg)
    assert datetime.fromisoformat(s.timestamp_start).tzinfo is not None


@pytest.mark.parametrize("text, expected", [("yes", True), ("Y", True), ("1", True),
                                            ("true", True), ("no", False), ("0", False),
                                            ("False", False)])
def test_boolean_spellings(cfg, text, expected):
    assert sm.make_session(raw(window_blocked=text), cfg, now=NOW).window_blocked is expected


def test_window_light_needs_no_lamp_fields(cfg):
    s = sm.make_session(raw(light_source="window", lamp_distance_mm="", lamp_angle_deg="",
                            lamp_to_card_mm=""), cfg, now=NOW)
    assert s.lamp_distance_mm is None and s.lamp_angle_deg is None and s.lamp_to_card_mm is None


def test_optional_card_fields_blank(cfg):
    s = sm.make_session(raw(card_id="", card_angle_deg="", dry_reference_type="none"), cfg, now=NOW)
    assert s.card_id is None and s.card_angle_deg is None


@pytest.mark.parametrize("stage", ["VE", "VC", "V1", "V3", "V15", "R1", "R8"])
def test_vegetative_and_reproductive_stages_accepted(cfg, stage):
    assert sm.make_session(raw(reproductive_stage=stage), cfg, now=NOW).reproductive_stage == stage


def test_porometer_value(cfg):
    assert sm.make_session(raw(porometer_gs_mol="0.31"), cfg, now=NOW).porometer_gs_mol == 0.31


# ---------- validation: rejection with clear messages ----------

@pytest.mark.parametrize("field, bad", [
    ("genotype", "williams82"), ("genotype", "Williams 82"),
    ("water_treatment", "drought"), ("light_source", "LED"), ("fan_state", "medium"),
    ("reproductive_stage", "V"), ("reproductive_stage", "V16"), ("reproductive_stage", "v3"),
    ("reproductive_stage", "R9"),
    ("dry_reference_type", "vaseline"), ("background", "white"),
])
def test_enumeration_typo_rejected(cfg, field, bad):
    with pytest.raises(ValueError) as e:
        sm.make_session(raw(**{field: bad}), cfg, now=NOW)
    msg = str(e.value)
    assert field in msg and repr(bad) in msg
    for allowed in cfg["enumerations"][field]:
        assert allowed in msg


@pytest.mark.parametrize("field, bad, words", [
    ("block", "0", "block"), ("block", "1.5", "whole number"), ("block", "A", "whole number"),
    ("lamp_distance_mm", "-10", "greater than 0"), ("working_distance_mm", "0", "greater than 0"),
    ("working_distance_mm", "nan", "number"), ("incidence_angle_deg", "95", "0 and 90"),
    ("card_angle_deg", "-1", "0 and 90"), ("lamp_angle_deg", "abc", "number"),
    ("acclimation_minutes", "-5", "0 or more"), ("porometer_gs_mol", "-0.1", "0 or more"),
    ("window_blocked", "maybe", "true or false"), ("wet_reference_present", "", "required"),
    ("plant_id", "", "required"), ("plant_id", "HK 03", "plant_id"),
    ("plant_id", "HK/03", "plant_id"), ("operator", " ", "required"),
    ("position", "***", "position"), ("leaf_position", "", "required"),
])
def test_bad_values_rejected(cfg, field, bad, words):
    with pytest.raises(ValueError, match=words):
        sm.make_session(raw(**{field: bad}), cfg, now=NOW)


def test_lamp_fields_required_for_led(cfg):
    with pytest.raises(ValueError, match="lamp_distance_mm.*led_lamp"):
        sm.make_session(raw(lamp_distance_mm=""), cfg, now=NOW)


def test_lamp_fields_must_be_blank_without_lamp(cfg):
    with pytest.raises(ValueError, match="lamp_to_card_mm.*blank"):
        sm.make_session(raw(light_source="window", lamp_distance_mm="", lamp_angle_deg=""),
                        cfg, now=NOW)


def test_unknown_field_rejected(cfg):
    with pytest.raises(ValueError, match="unknown field"):
        sm.make_session(raw(colour="green"), cfg, now=NOW)


def test_all_errors_reported_together(cfg):
    with pytest.raises(ValueError) as e:
        sm.make_session(raw(genotype="x", fan_state="y"), cfg, now=NOW)
    assert "genotype" in str(e.value) and "fan_state" in str(e.value)


# ---------- writing: csv, session.json, refusal to touch an old file ----------

def test_start_session_writes_csv_and_json(cfg, tmp_path):
    manifest = write_manifest(tmp_path / "cards.csv", [card_row()])
    s = sm.start_session(raw(), cfg, root=tmp_path, manifest_path=manifest, now=NOW)
    csv_path = tmp_path / "data" / "raw" / "session_metadata.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == APPENDIX_B_SESSION_COLUMNS
    assert rows[0]["session_id"] == s.session_id
    assert rows[0]["window_blocked"] == "true" and rows[0]["porometer_gs_mol"] == ""
    js = json.loads((tmp_path / "data" / "raw" / s.session_id / "session.json").read_text("utf-8"))
    assert list(js.keys()) == APPENDIX_B_SESSION_COLUMNS
    assert js["block"] == 1 and js["porometer_gs_mol"] is None and js["window_blocked"] is True


def test_second_session_appends(cfg, tmp_path):
    manifest = write_manifest(tmp_path / "cards.csv", [card_row()])
    sm.start_session(raw(), cfg, root=tmp_path, manifest_path=manifest, now=NOW)
    sm.start_session(raw(plant_id="WI-01", genotype="Williams_82"), cfg, root=tmp_path,
                     manifest_path=manifest, now=NOW + timedelta(hours=1))
    with open(tmp_path / "data" / "raw" / "session_metadata.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2


@pytest.mark.parametrize("header", [
    ["session_id", "timestamp_start", "plant_id", "genotype", "water_treatment",
     "working_distance_mm", "incidence_angle_deg", "notes"],          # the old 8-column file
    ["something", "else"],                                            # unknown header
])
def test_old_or_unknown_header_is_refused_and_untouched(cfg, tmp_path, header):
    csv_path = tmp_path / "data" / "raw" / "session_metadata.csv"
    csv_path.parent.mkdir(parents=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(["x"] * len(header))
    before = csv_path.read_bytes()
    with pytest.raises(sm.MetadataFileError, match="move it aside"):
        sm.start_session(raw(), cfg, root=tmp_path, now=NOW)
    assert csv_path.read_bytes() == before                  # never rewritten
    assert not list(csv_path.parent.glob("*.bak"))           # no migration copy either
    assert [p.name for p in csv_path.parent.iterdir()] == ["session_metadata.csv"]  # no session folder


def test_empty_existing_file_is_refused(cfg, tmp_path):
    csv_path = tmp_path / "data" / "raw" / "session_metadata.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("", encoding="utf-8")
    with pytest.raises(sm.MetadataFileError):
        sm.start_session(raw(), cfg, root=tmp_path, now=NOW)


def test_existing_session_json_never_overwritten(cfg, tmp_path):
    s = sm.make_session(raw(), cfg, now=NOW)
    folder = tmp_path / "data" / "raw" / s.session_id
    folder.mkdir(parents=True)
    (folder / "session.json").write_text("{}", encoding="utf-8")
    with pytest.raises(sm.MetadataFileError, match="already exists"):
        sm.start_session(raw(), cfg, root=tmp_path, now=NOW)
    assert (folder / "session.json").read_text(encoding="utf-8") == "{}"
    assert not (tmp_path / "data" / "raw" / "session_metadata.csv").exists()


def test_invalid_values_write_nothing(cfg, tmp_path):
    with pytest.raises(ValueError):
        sm.start_session(raw(genotype="bad"), cfg, root=tmp_path, now=NOW)
    assert not (tmp_path / "data").exists()


# ---------- card manifest warning ----------

def test_card_not_in_manifest_warns_only(cfg, tmp_path, caplog):
    manifest = write_manifest(tmp_path / "cards.csv", [card_row(card_id="card02")])
    with caplog.at_level(logging.WARNING, logger="acquisition.session_metadata"):
        s = sm.start_session(raw(card_id="card01"), cfg, root=tmp_path,
                             manifest_path=manifest, now=NOW)
    assert s.card_id == "card01"
    assert "card01" in caplog.text and "not in the reference-card manifest" in caplog.text


def test_card_in_manifest_no_warning(cfg, tmp_path, caplog):
    manifest = write_manifest(tmp_path / "cards.csv", [card_row()])
    with caplog.at_level(logging.WARNING, logger="acquisition.session_metadata"):
        sm.start_session(raw(), cfg, root=tmp_path, manifest_path=manifest, now=NOW)
    assert "manifest" not in caplog.text


def test_missing_manifest_warns_only(cfg, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="acquisition.session_metadata"):
        sm.start_session(raw(), cfg, root=tmp_path, manifest_path=tmp_path / "none.csv", now=NOW)
    assert "manifest" in caplog.text


def test_no_manifest_check_unless_dry_reference_is_card(cfg, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="acquisition.session_metadata"):
        sm.start_session(raw(dry_reference_type="none", card_id="zzz"), cfg, root=tmp_path,
                         manifest_path=tmp_path / "none.csv", now=NOW)
    assert caplog.text == ""


# ---------- command line and prompts ----------

def _argv(values):
    out = []
    for k, v in values.items():
        out += ["--" + k.replace("_", "-"), v]
    return out


def test_cli_flags_cover_every_operator_field():
    parser = sm.build_parser()
    dests = {a.dest for a in parser._actions if a.dest not in ("help", "root", "config")}
    assert dests == set(sm.OPERATOR_FIELDS)
    assert set(sm.OPERATOR_FIELDS) == set(APPENDIX_B_SESSION_COLUMNS) - {"session_id", "timestamp_start"}


def test_cli_all_flags_no_prompts(cfg, tmp_path):
    def no_input(prompt):
        raise AssertionError(f"should not prompt: {prompt}")
    write_manifest(tmp_path / "cards.csv", [card_row()])
    s = sm.main(_argv(raw()) + ["--root", str(tmp_path)], input_fn=no_input,
                manifest_path=tmp_path / "cards.csv")
    assert s.plant_id == "HK-03"
    assert (tmp_path / "data" / "raw" / "session_metadata.csv").exists()


def test_cli_prompts_only_for_missing_and_reprompts_bad(cfg, tmp_path):
    given = raw()
    missing = {"genotype": given.pop("genotype"), "operator": given.pop("operator")}
    answers = iter(["Williams82", missing["genotype"], missing["operator"]])
    asked = []

    def fake_input(prompt):
        asked.append(prompt)
        return next(answers)
    s = sm.main(_argv(given) + ["--root", str(tmp_path)], input_fn=fake_input,
                manifest_path=write_manifest(tmp_path / "cards.csv", [card_row()]))
    assert s.genotype == "Houjaku_Kuwazu" and s.operator == "AP"
    assert len(asked) == 3 and "genotype" in asked[0] and "genotype" in asked[1]


def test_prompt_skips_lamp_fields_without_lamp(cfg, tmp_path):
    given = raw(light_source="window")
    for k in ("lamp_distance_mm", "lamp_angle_deg", "lamp_to_card_mm"):
        given.pop(k)

    def no_input(prompt):
        raise AssertionError(f"should not prompt: {prompt}")
    s = sm.main(_argv(given) + ["--root", str(tmp_path)], input_fn=no_input,
                manifest_path=write_manifest(tmp_path / "cards.csv", [card_row()]))
    assert s.lamp_distance_mm is None


def test_bad_flag_value_exits_with_message(cfg, tmp_path, capsys):
    with pytest.raises(SystemExit):
        sm.main(_argv(raw(fan_state="medium")) + ["--root", str(tmp_path)],
                input_fn=lambda p: "")
    assert "fan_state" in capsys.readouterr().err


def test_prompt_for_session_keeps_logger_signature():
    # combined_logger.py calls prompt_for_session() with no arguments.
    import inspect
    params = inspect.signature(sm.prompt_for_session).parameters
    assert all(p.default is not inspect.Parameter.empty for p in params.values())


# ---------- reference-card loader (T2.2) ----------

def test_loader_reads_rows(tmp_path):
    p = write_manifest(tmp_path / "c.csv", [card_row(), card_row(card_id="card02", a_s_card="0.94",
                                                                 area_mm2="1500", d_min_mm="30")])
    cards = load_reference_cards(p)
    assert list(cards) == ["card01", "card02"]
    assert cards["card01"].a_s_card is None and cards["card01"].emissivity == 0.95
    assert cards["card02"].a_s_card == 0.94 and cards["card02"].d_min_mm == 30.0
    assert cards["card02"].solidity is None


def test_loader_accepts_missing_optional_columns(tmp_path):
    p = write_manifest(tmp_path / "c.csv", [card_row()], columns=list(CARD_REQUIRED_COLUMNS))
    assert load_reference_cards(p)["card01"].geometry_source is None


@pytest.mark.parametrize("col", ["card_id", "a_s_card", "notes", "photo_path"])
def test_loader_requires_columns(tmp_path, col):
    cols = [c for c in APPENDIX_B_CARD_COLUMNS if c != col]
    p = write_manifest(tmp_path / "c.csv", [], columns=cols)
    with pytest.raises(ReferenceCardError, match=col):
        load_reference_cards(p)


def test_loader_rejects_unknown_column(tmp_path):
    p = write_manifest(tmp_path / "c.csv", [], columns=APPENDIX_B_CARD_COLUMNS + ["colour"])
    with pytest.raises(ReferenceCardError, match="colour"):
        load_reference_cards(p)


@pytest.mark.parametrize("kw, words", [
    ({"card_id": ""}, "card_id"), ({"shape_group": ""}, "shape_group"),
    ({"source": "drawn"}, "source"), ({"geometry_source": "scanner"}, "geometry_source"),
    ({"a_s_card": "1.2"}, "a_s_card"), ({"emissivity": "0"}, "emissivity"),
    ({"thickness_mm": "-1"}, "thickness_mm"), ({"area_mm2": "abc"}, "area_mm2"),
    ({"eccentricity": "1.5"}, "eccentricity"), ({"solidity": "0"}, "solidity"),
])
def test_loader_rejects_bad_values_with_row_number(tmp_path, kw, words):
    p = write_manifest(tmp_path / "c.csv", [card_row(), card_row(**{"card_id": "card02", **kw})])
    with pytest.raises(ReferenceCardError, match=f"row 3.*{words}|{words}.*row 3"):
        load_reference_cards(p)


def test_loader_rejects_duplicate_ids(tmp_path):
    p = write_manifest(tmp_path / "c.csv", [card_row(), card_row()])
    with pytest.raises(ReferenceCardError, match="duplicate"):
        load_reference_cards(p)


def test_loader_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_reference_cards(tmp_path / "nope.csv")
