"""실적발표 컨퍼런스콜 대본 수집 — Motley Fool(fool.com)이 무료로 공개하는 대본 페이지를 스크래핑한다.

verbatim Q&A까지 포함된 완전한 대본을 무료로 주는 공식 API가 마땅치 않다 (FMP/API Ninjas의 해당
엔드포인트는 유료 전용). fool.com은 페이월 없이 대본 전문을 공개하므로 이를 파싱해서 쓴다 — 공식
API가 아니라 페이지 구조가 바뀌면 깨질 수 있다.

커버리지가 두 가지 이유로 제한적이다: (1) find_transcript_url이 검색하는 "최근 발표" 인덱스
(_INDEX_URL)는 그 시점 기준 최근 20개 정도만 보여주는 좁은 롤링 윈도우라, 오래된 분기는 밀려나 못
찾는다. (2) 더 근본적으로, fool.com은 애초에 모든 상장사의 대본을 만들지 않는다 — TSMC/NVIDIA 같은
대형·주목주 위주라, Navitas Semiconductor(NVTS, 시총 약 30억 달러)처럼 상대적으로 작은 종목은 그
회사 자체 Fool 페이지에도 대본 링크가 아예 없는 걸 확인했다(직접 테스트, 날짜를 추정한 URL 탐색으로도
못 찾음). 미국 상장 종목만 커버한다(한국 기업은 fool.com에 대본이 없다).

화자별 발언(speaker/text)뿐 아니라, chunking.py의 Q&A 인식 병합에 필요한 구조 정보도 함께 뽑는다:

- speaker_type/speaker_role: fool.com이 "CALL PARTICIPANTS" 섹션에 경영진 이름과 직함을 구조화해
  제공하므로("Chief Financial Officer - Wendell Huang") 경영진(management)은 신뢰도 높게 분류된다.
  이 목록에 없는 화자는 애널리스트(analyst)로 간주한다.
- organization: 애널리스트 소속 증권사. 진행자가 "Sunny Lin from UBS" 식으로 소개하는 문장을 정규식으로
  파싱한 best-effort 값이다 — 콜마다 소개 문구 스타일이 달라(예: "question comes from X with Y") 못
  잡는 경우가 많다는 걸 감안할 것. 못 찾으면 None.
- section: "prepared_remarks" 또는 "qa". 첫 번째 진짜 애널리스트 발언이 나오는 시점을 기준으로 나눈다.
  처음엔 "Q&A 시작을 알리는 문구"(예: "question and answer session")로 텍스트 휴리스틱을 시도했는데,
  실제로는 진행자가 콜 초반에 "이따가 Q&A 시간을 열겠다"고 미리 언급하는 문장에도 걸려서 오탐이 심했다
  (예: TSMC 콜에서 진짜 Q&A 시작보다 20문단 이상 앞에서 잘못 전환됨). 화자 유형(speaker_type) 기준이
  훨씬 안정적이라 이쪽으로 바꿨다 — 경영진 발언이 아무리 길어도 진짜 애널리스트가 등장하기 전까진
  prepared_remarks로 남는다.
- fiscal_quarter: 대상 분기(예: "Q2 2026"). 본문이 아니라 URL 슬러그(".../q2-2026-earnings-call-.../")에서
  뽑는다 — fool.com이 URL을 이 형식으로 일관되게 생성해서 본문 텍스트 파싱보다 훨씬 안정적이다.
"""
import os
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import DATA_DIR_RAW

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
_INDEX_URL = "https://www.fool.com/earnings-call-transcripts/"
_LINK_RE = re.compile(r'href="(/earnings/call-transcripts/[^"]+)"')
_QUARTER_RE = re.compile(r"-q(\d)-(\d{4})-earnings-call-transcript")

_ANALYST_INTRO_RE = re.compile(
    r"([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+)+)\s+(?:from|with)\s+([A-Z][\w&.,'\-\s]+?)[,.]"
)
_ROLE_KEYWORDS = [
    ("chief executive officer", "CEO"),
    ("chief financial officer", "CFO"),
    ("chief operating officer", "COO"),
    ("investor relations", "IR"),
    ("chairman", "Chairman"),
    ("president", "President"),
]


def find_transcript_url(ticker):
    """최근 실적발표 대본 목록에서 해당 티커의 URL을 찾는다 (최근 발표분만 커버됨)."""
    resp = requests.get(_INDEX_URL, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    links = set(_LINK_RE.findall(resp.text))

    ticker_slug = f"-{ticker.lower()}-q"
    matches = [link for link in links if ticker_slug in link.lower()]
    if not matches:
        raise ValueError(
            f"최근 실적발표 목록에서 '{ticker}' 대본을 찾을 수 없다 — 아직 발표 전이거나 목록에서 밀려났을 수 있다."
        )
    return "https://www.fool.com" + sorted(matches)[-1]


def _parse_fiscal_quarter(url):
    """URL 슬러그(예: ".../tsm-tsm-q2-2026-earnings-call-transcript/")에서 분기를 뽑는다. fool.com이
    URL을 이 형식으로 일관되게 생성하므로 본문 텍스트를 파싱하는 것보다 훨씬 신뢰도가 높다."""
    m = _QUARTER_RE.search(url)
    return f"Q{m.group(1)} {m.group(2)}" if m else None


def _classify_role(title):
    title_lower = title.lower()
    for keyword, label in _ROLE_KEYWORDS:
        if keyword in title_lower:
            return label
    return "Other"


def _parse_call_participants(soup):
    """"CALL PARTICIPANTS" 섹션에서 경영진 이름 -> 직함을 뽑는다. 못 찾으면 빈 dict."""
    heading = soup.find(lambda tag: tag.name in ("h2", "h3") and "PARTICIPANT" in tag.get_text().upper())
    if heading is None:
        return {}
    ul = heading.find_next_sibling("ul")
    if ul is None:
        return {}

    participants = {}
    for li in ul.find_all("li"):
        text = li.get_text(" ", strip=True)
        if " - " in text:
            title, name = text.rsplit(" - ", 1)
            participants[name.strip()] = title.strip()
    return participants


def _find_analyst_organization(preceding_text, speaker_name):
    """직전 문맥(주로 진행자의 소개 발언)에서 "이름 from/with 소속" 패턴으로 애널리스트 소속을 추정한다."""
    for m in _ANALYST_INTRO_RE.finditer(preceding_text[-500:]):
        if m.group(1).strip() == speaker_name:
            return m.group(2).strip()
    return None


def download_transcript(ticker, url=None, use_cache=True):
    """실적발표 컨퍼런스콜 대본 전문을 문단 단위로 받아온다.

    반환 컬럼: speaker, speaker_role(CEO/CFO/Analyst 등), speaker_type(management/analyst),
    organization(애널리스트 소속, best-effort), section(prepared_remarks/qa), fiscal_quarter(예: "Q2 2026"),
    text. url을 지정하지 않으면 find_transcript_url로 최근 목록에서 자동 검색한다.
    """
    cache_path = os.path.join(DATA_DIR_RAW, f"transcript_{ticker.upper()}.csv")
    if use_cache and os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    url = url or find_transcript_url(ticker)
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    body = soup.find(class_="transcript-content")
    if body is None:
        raise ValueError(f"{url}에서 대본 본문을 찾을 수 없다 — 페이지 구조가 바뀌었을 수 있다.")

    participants = _parse_call_participants(soup)
    fiscal_quarter = _parse_fiscal_quarter(url)

    rows = []
    speaker = None
    preceding_text = ""
    for p in body.find_all("p"):
        strong = p.find("strong")
        # 문단이 <strong>화자명:</strong>으로 시작하면 화자 라벨로 보고 갱신한다 (본문에서는 제거)
        if strong is not None and p.contents and p.contents[0] == strong:
            label = strong.get_text(strip=True)
            if label.endswith(":"):
                speaker = label.rstrip(":").strip()
                strong.extract()

        text = p.get_text(" ", strip=True)
        if not text:
            continue

        if speaker is None:
            speaker_type, speaker_role, organization = None, None, None
        elif speaker in participants:
            speaker_type, speaker_role, organization = "management", _classify_role(participants[speaker]), None
        elif speaker.strip().lower() == "operator":
            # 다이얼인 안내를 읽는 자동 진행자 — 경영진도 애널리스트도 아니라서 Q&A 시작 판정에서 제외한다
            speaker_type, speaker_role, organization = "operator", "Operator", None
        else:
            speaker_type = "analyst"
            speaker_role = "Analyst"
            organization = _find_analyst_organization(preceding_text, speaker)

        rows.append({
            "speaker": speaker,
            "speaker_role": speaker_role,
            "speaker_type": speaker_type,
            "organization": organization,
            "fiscal_quarter": fiscal_quarter,
            "text": text,
        })
        preceding_text += " " + text

    df = pd.DataFrame(
        rows, columns=["speaker", "speaker_role", "speaker_type", "organization", "fiscal_quarter", "text"]
    )
    if df.empty:
        raise ValueError(f"{url}에서 대본 문단을 하나도 추출하지 못했다.")

    # 진짜 애널리스트가 처음 등장하는 행부터 "qa" — 경영진 발언이 아무리 길어도 그 전까진 prepared_remarks.
    analyst_rows = df.index[df["speaker_type"] == "analyst"]
    qa_start = analyst_rows[0] if len(analyst_rows) else len(df)
    df["section"] = ["qa" if i >= qa_start else "prepared_remarks" for i in df.index]

    os.makedirs(DATA_DIR_RAW, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df
