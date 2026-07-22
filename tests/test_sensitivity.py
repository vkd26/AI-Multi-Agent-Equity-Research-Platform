"""sensitivity.py의 DCF 민감도 격자 계산을 검증한다."""
import pytest

from src.valuation.dcf import run_dcf
from src.valuation.sensitivity import dcf_sensitivity_table


def test_dcf_sensitivity_table_shape_and_axis_labels():
    table = dcf_sensitivity_table(
        base_fcf=100, growth_rate=0.05, years=3, net_debt=0, shares_outstanding=10,
        wacc_range=[0.10, 0.12], terminal_growth_range=[0.02, 0.03],
    )
    assert table.shape == (2, 2)
    assert list(table.index) == [0.10, 0.12]
    assert list(table.columns) == [0.02, 0.03]


def test_dcf_sensitivity_table_matches_direct_run_dcf_call():
    table = dcf_sensitivity_table(
        base_fcf=100, growth_rate=0.05, years=3, net_debt=0, shares_outstanding=10,
        wacc_range=[0.10], terminal_growth_range=[0.03],
    )
    direct = run_dcf(base_fcf=100, wacc=0.10, growth_rate=0.05, terminal_growth_rate=0.03, years=3, shares_outstanding=10)
    assert table.loc[0.10, 0.03] == pytest.approx(direct["implied_share_price"])


def test_dcf_sensitivity_table_is_monotonic_in_expected_directions():
    table = dcf_sensitivity_table(
        base_fcf=100, growth_rate=0.05, years=5, net_debt=0, shares_outstanding=10,
        wacc_range=[0.10, 0.15, 0.20], terminal_growth_range=[0.01, 0.03, 0.05],
    )
    # WACC가 오를수록 내재가치는 낮아져야 한다 (할인율 상승)
    assert table.loc[0.10, 0.03] > table.loc[0.15, 0.03] > table.loc[0.20, 0.03]
    # 터미널성장률이 오를수록 내재가치는 높아져야 한다
    assert table.loc[0.10, 0.01] < table.loc[0.10, 0.03] < table.loc[0.10, 0.05]


def test_dcf_sensitivity_table_leaves_none_for_diverging_combinations():
    # wacc <= terminal_growth 인 조합은 발산하므로 None으로 남아야 한다 (에러로 죽으면 안 됨)
    table = dcf_sensitivity_table(
        base_fcf=100, growth_rate=0.05, years=3, net_debt=0, shares_outstanding=10,
        wacc_range=[0.02], terminal_growth_range=[0.05],
    )
    assert table.loc[0.02, 0.05] is None
