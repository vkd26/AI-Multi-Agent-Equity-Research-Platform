"""프로젝트 전역 설정. 분석 대상 기업을 바꾸고 싶으면 이 파일의 TICKER/COMPANY_NAME만 수정하면 된다."""
import os
from datetime import date, timedelta

# notebooks/에서 실행하든 프로젝트 루트에서 실행하든 항상 같은 data/ 폴더를 가리키도록 절대경로로 고정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def market_of(ticker):
    """티커의 상장 시장을 판별한다. ".KS"(코스피)/".KQ"(코스닥)면 한국, 접미사가 없으면 미국으로 본다."""
    return "KR" if ticker.endswith((".KS", ".KQ")) else "US"


TICKER = "005930.KS"  # 한국은 "005930.KS"(코스피)/".KQ"(코스닥), 미국은 "NVDA"처럼 접미사 없이 표기
COMPANY_NAME = "삼성전자"

MARKET = market_of(TICKER)  # "KR" 또는 "US" — collector가 dart.py/sec.py 중 어느 쪽을 쓸지 이 값으로 분기
DART_STOCK_CODE = TICKER.split(".")[0] if MARKET == "KR" else None  # OpenDART용 6자리 종목코드, 미국 종목이면 해당 없음

LOOKBACK_DAYS = 90
END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=LOOKBACK_DAYS)

DATA_DIR_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_DIR_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
