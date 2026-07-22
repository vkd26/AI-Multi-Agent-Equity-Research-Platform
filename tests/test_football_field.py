"""football_field.py의 방법론별 밸류에이션 범위 집계를 검증한다."""
from src.valuation.football_field import build_football_field


def test_build_football_field_computes_range_and_midpoint():
    valuations = {"DCF": (100, 200), "Comps": (300, 500)}
    table = build_football_field(valuations)

    dcf_row = table[table["method"] == "DCF"].iloc[0]
    assert dcf_row["range"] == 100
    assert dcf_row["midpoint"] == 150

    comps_row = table[table["method"] == "Comps"].iloc[0]
    assert comps_row["range"] == 200
    assert comps_row["midpoint"] == 400


def test_build_football_field_sorts_by_midpoint_ascending():
    valuations = {"High method": (500, 700), "Low method": (10, 30), "Mid method": (100, 200)}
    table = build_football_field(valuations)
    assert list(table["method"]) == ["Low method", "Mid method", "High method"]
