"""⑦ Valuation Engine — Football Field. 여러 밸류에이션 방법론(DCF, Comps, 52주 가격범위 등)의 결과를
한 표로 모아 방법론 간 비교를 쉽게 한다.
"""
import pandas as pd


def build_football_field(valuations):
    """valuations: {"방법론 이름": (저가, 고가)} 형태의 dict를 받아 중간값(midpoint) 기준으로 정렬한
    요약 표를 만든다.

    예: {"DCF (Bear-Bull)": (150, 250), "Comps (EV/EBITDA)": (400, 600), "52주 가격범위": (164, 237)}
    """
    rows = [{"method": name, "low": low, "high": high} for name, (low, high) in valuations.items()]
    table = pd.DataFrame(rows)
    table["range"] = table["high"] - table["low"]
    table["midpoint"] = (table["low"] + table["high"]) / 2
    return table.sort_values("midpoint").reset_index(drop=True)
