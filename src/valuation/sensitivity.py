"""⑦ Valuation Engine — Sensitivity Analysis. WACC × 터미널 성장률을 격자로 바꿔가며 DCF 내재주가가
얼마나 민감한지 보여준다.
"""
import pandas as pd

from src.valuation.dcf import run_dcf


def dcf_sensitivity_table(
    base_fcf, growth_rate, years, net_debt, shares_outstanding, wacc_range, terminal_growth_range,
):
    """WACC(행) x 터미널성장률(열) 격자로 DCF 내재주가를 계산한 표를 반환한다.

    WACC가 터미널성장률보다 작거나 같은 조합은 계산이 발산해서(dcf.terminal_value가 ValueError를
    던짐) 그 칸은 None으로 남긴다.
    """
    rows = {}
    for wacc in wacc_range:
        row = {}
        for terminal_growth in terminal_growth_range:
            try:
                result = run_dcf(
                    base_fcf=base_fcf, wacc=wacc, growth_rate=growth_rate,
                    terminal_growth_rate=terminal_growth, years=years, net_debt=net_debt,
                    shares_outstanding=shares_outstanding,
                )
                row[terminal_growth] = result.get("implied_share_price")
            except ValueError:
                row[terminal_growth] = None
        rows[wacc] = row

    table = pd.DataFrame(rows).T
    table.index.name = "wacc"
    table.columns.name = "terminal_growth_rate"
    return table
