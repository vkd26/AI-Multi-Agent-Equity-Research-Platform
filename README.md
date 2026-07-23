# AI Multi-Agent Equity Research Platform

공시·실적발표·뉴스·재무데이터를 자동 수집해 Financial RAG로 근거를 검색하고, LLM으로 실적발표 분석·뉴스
논조분석·투자 Thesis·PDF 리포트까지 생성하는 AI 기반 Equity Research 파이프라인이다. 한국(DART)·미국(SEC)
상장 종목을 모두 지원한다. TSMC(TSM)로 end-to-end 데모한다.

## 파이프라인

| # | 컴포넌트 | 내용 |
|---|---|---|
| ① | Data Collector | DART(한국)/SEC(미국) 공시, yfinance 시세·기업정보, 네이버/Finnhub 뉴스, Motley Fool 실적발표 대본(미국) 자동 수집·캐싱 |
| ② | Document Processor | 실적발표 대본을 Q&A 경계·화자 역할을 인식하며 청킹 |
| ③ | Financial RAG | BGE-M3 임베딩 + FAISS/BM25 하이브리드 검색 → BGE-Reranker 재정렬 → 출처(citation) 부착 |
| ④ | Earnings Call Analyzer | 키워드 언급 QoQ 비교, 화자 역할별 발언 필터링, RAG 기반 Guidance/Risk/CapEx/Margin/Demand 요약 |
| ⑤ | News Analyzer | 뉴스 논조(Positive/Negative/Neutral) + 영향도(High/Medium/Low) 분류 |
| ⑥ | Financial Analyzer | Margin/ROE/PER/PEG/EV-EBITDA/Debt·Equity/FCF/Revenue Growth |
| ⑦ | Valuation Engine | DCF(WACC·무위험금리·ERP·성장률 전부 실데이터 기반, 성장률은 5년에 걸쳐 터미널성장률로 선형 수렴), Comparable Company Analysis(Gemini 검색+yfinance 교차검증으로 피어 자동 선정, 한국·미국 상장만), Sensitivity, Football Field |
| ⑧ | Investment Thesis Generator | Valuation·재무·실적발표·뉴스 근거를 종합해 Thesis/Bear Case/Catalyst/Risk/Target Price 생성 (Target Price는 DCF 60%·Comps 40% 가중평균 기준점에 근거) |
| ⑨ | Portfolio Monitor | 공시/뉴스/실적발표/목표주가 변동 4가지를 매일 체크, 트리거 시에만 비싼 LLM 재생성 유도 |
| ⑩ | PDF Report | 위 결과 + LLM 생성 Company Overview/Business Analysis + 투자의견(매수/매도/중립)을 Jinja2+WeasyPrint로 최종 투자보고서 PDF 생성 |

**+ Tool-calling Agent** (`src/agents/`) — 위 컴포넌트들을 "도구"로 등록해, 자연어 질문 하나로 LLM이 필요한
도구만 스스로 판단해 호출한다. LangGraph 같은 프레임워크 대신 `planner.py`에 도구 호출 루프(판단→실행→
결과반영→반복)를 직접 구현했다 — 이유는 해당 모듈 독스트링 참고.

## 폴더 구조

```
ai-equity-research-agent/
├── main.py                     # CLI 진입점 — python main.py "자연어 질문" (Tool-calling Agent)
├── notebooks/
│   └── 01_end_to_end_demo.ipynb   # 전체 파이프라인 실행 결과 (recruiter용 메인 딜리버러블)
├── src/
│   ├── config.py            # 대상기업/시장(KR·US) 판별
│   ├── collector/            # ① dart.py, sec.py, market.py, news.py, transcript.py
│   ├── rag/                  # ②③ chunking.py, embedding.py, vector_store.py, retrieval.py
│   ├── analytics/             # ④⑤⑥ earnings_call_analyzer.py, news_analyzer.py, financial_analyzer.py
│   ├── valuation/             # ⑦ dcf.py, comps.py, sensitivity.py, football_field.py
│   ├── agents/                # ⑧⑨ + Tool-calling Agent: planner.py, tools.py, research_agent.py, thesis_agent.py, portfolio_monitor.py
│   └── report/                # ⑩ report_generator.py
├── templates/
│   └── report_template.html   # PDF 보고서 템플릿
├── data/
│   ├── raw/                  # 수집 원본 캐시 (최초 실행 시 자동 생성)
│   └── processed/            # Portfolio Monitor 상태 등
├── output/reports/            # 생성된 PDF
├── tests/                     # 단위 테스트 196개 (네트워크·LLM 호출은 모킹, 실 데이터 검증은 아래 참고)
├── requirements.txt
└── pytest.ini
```

## API 키 설정

`.env.example`을 참고해 필요한 키를 프로젝트 루트의 `.env` 또는 홈 디렉토리(`~/.env`)에 저장한다.

- `DART_API_KEY`: [opendart.fss.or.kr](https://opendart.fss.or.kr) 무료 발급
- `NAVER_CLIENT_ID` / `NAVER_CLIENT_SECRET`: [네이버 개발자센터](https://developers.naver.com) 무료 발급
- `FINNHUB_API_KEY`: [finnhub.io](https://finnhub.io) 무료 발급
- `GEMINI_API_KEY`: [aistudio.google.com](https://aistudio.google.com) 무료 발급
- `SEC_USER_AGENT` (선택): SEC EDGAR는 키가 없지만 신원을 밝히는 User-Agent를 요구한다 — 기본값도 동작하지만 운영 시 실제 연락처로 바꾸는 걸 권장
- `KRX_API_KEY` (선택): [openapi.krx.co.kr](https://openapi.krx.co.kr) 회원가입 후 "국채전문유통시장 일별매매정보" 서비스를 별도 신청(승인까지 약 1영업일)하면 발급된다 — 한국 무위험금리(국고채 10년 지표금리)를 실시간 조회하는 데 쓴다. 없으면 확인 시점 값으로 자동 폴백한다.

임베딩(BGE-M3)·리랭커(BGE-Reranker) 모델은 API 키 없이 로컬에서 실행되며, 첫 실행 시 허깅페이스에서
자동 다운로드된다(약 2GB).

## 실행 방법

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_end_to_end_demo.ipynb
```

Tool-calling Agent에 자연어로 바로 질문하려면(cmd/PowerShell 등 터미널에서):

```bash
python main.py "TSM 최근 재무비율이랑 뉴스 논조 같이 알려줘"
python main.py "005930.KS DCF 밸류에이션 해줘" --verbose   # --verbose: 호출된 도구 로그도 함께 출력
```

코드에서 바로 쓰려면:

```python
from src.agents.research_agent import run_equity_research_agent
answer, log = run_equity_research_agent("TSM 최근 재무비율이랑 뉴스 논조 같이 알려줘")
```

단위 테스트:

```bash
pytest
```

## 알려진 제한사항

- **한국 종목은 실적발표 대본 미지원** — Motley Fool(fool.com)이 한국 기업 대본을 제공하지 않는다. DART/
  yfinance 기반 컴포넌트(공시, 재무비율, DCF, Comps 등)는 한국 종목도 전부 지원한다.
- **실적발표 대본 커버리지가 제한적** — 두 가지 이유. (1) fool.com의 "최근 발표" 인덱스가 페이지당
  약 20개씩 페이지네이션되어 있다 — `find_transcript_url()`이 기본 10페이지(약 200개)까지 뒤지지만,
  그보다 더 과거로 밀려난 분기는 못 찾는다(처음엔 1페이지만 보고 포기하도록 짜여 있었는데, 실제로
  페이지네이션이 있다는 걸 뒤늦게 발견하고 고쳤다). (2) 더 근본적으로 fool.com 자체가 모든 상장사의
  대본을 만들지 않는다 — TSMC/NVIDIA 같은 대형·주목주 위주라, Navitas Semiconductor(NVTS, 시총 약
  30억 달러)처럼 상대적으로 작은 종목은 10페이지 끝까지 확인해도 대본 자체가 없는 걸 확인했다(페이지네이션
  범위 문제가 아니라 애초에 안 만드는 것). 두 경우 다 에러 없이 건너뛰도록 처리되어 있다.
- **애널리스트 소속 증권사 추출은 best-effort** — 진행자 소개 문구 스타일이 회사마다 달라(예: "X from Y"
  vs "question comes from X with Y") 일부는 못 잡는다.
- **ROIC은 의도적으로 제외** — yfinance의 `operatingMargins` 필드가 다른 마진 필드(`totalRevenue`,
  `profitMargins`)와 달리 TTM이 아니라 최신 분기 단독치라, `operating_margin × totalRevenue`로 영업이익을
  역산하면 시점이 안 맞아 값이 부풀려진다(TSM 실 데이터로 검증: 계산값 70.3% vs 실제 재무제표 기준
  54.2%). 바로잡으려면 분기별 재무제표를 별도로 받아야 해서, 지표 하나의 정밀도 대비 비용이 크다고
  판단해 뺐다.
- **도메인 적응 없음** — 임베딩(BGE-M3)·리랭커·LLM(Gemini) 전부 범용 사전학습 모델을 그대로 쓴다. 금융
  특화는 파인튜닝이 아니라 RAG(검색된 근거를 프롬프트에 제공)로 확보한다.
- **LangGraph 미사용** — 워크플로가 거의 선형(분기 1곳뿐)이라 그래프 프레임워크의 이점이 적어서, 대신
  도구 호출 루프를 직접 구현했다(`src/agents/planner.py` 참고).
- **재무제표 통화 환산은 실행 시점 환율(spot rate)을 쓴다** — TSM처럼 `currency`(주가 통화)와
  `financialCurrency`(재무제표 보고 통화)가 다른 ADR 종목은 `market.normalize_financial_currency()`가
  재무제표 절대금액 필드를 실시간 환율로 환산한다. 다만 이건 "지금 환율"이지 그 재무제표가 실제로
  보고된 시점의 환율은 아니라서, 환율 변동이 컸던 기간엔 약간의 오차가 있을 수 있다.
- **피어 그룹 자동 선정은 실행마다 결과가 조금씩 다를 수 있다** — Gemini 검색 기반이라 호출 시점에 따라
  다른 후보를 찾을 수 있다(예: TSM 피어가 어떤 실행에선 GFS/UMC/TSEM/000990.KS/005930.KS, 다른
  실행에선 일부만 겹침). 전부 파운드리/반도체 동종업계라는 방향성은 일관되고, yfinance로 실존 여부는
  항상 교차검증한다 — 한국·미국 상장 종목만 허용한다(`comps.py`의 `_is_kr_or_us_ticker` 참고).

## 현재 상태

10개 컴포넌트(①~⑩) + Tool-calling Agent 구현 완료. TSM(TSMC)·NVDA·005930.KS(삼성전자) 실 데이터로
전 구간 검증했고, `notebooks/01_end_to_end_demo.ipynb`에 TSM 기준 end-to-end 실행 결과(154개 대본 문단
수집 → 56개 청크 인덱싱 → 하이브리드 검색 → Guidance/Risk/CapEx/Margin/Demand 요약 → 뉴스 241건
논조분류 → 재무비율 → DCF(내재주가 $240.76)/Comps(피어 GFS·UMC·TSEM·000990.KS·005930.KS 자동 선정,
내재주가 $230~$519)/Sensitivity/Football Field → Investment Thesis/Bear Case/Catalyst/Risk/Target
Price($724.89, DCF·Comps 가중평균 근거) → Portfolio Monitor → PDF 리포트(투자의견 포함) → 자연어
질의응답까지)가 저장되어 있다. 단위 테스트 207개 통과.

최근에 고친 실제 버그:
- yfinance가 TSM처럼 주가는 USD, 재무제표는 원래 보고통화(TWD)로 반환하는 케이스를 모르고 그대로
  계산에 써서 DCF 내재주가가 실제 주가의 20~30배로 부풀려졌던 문제
- DCF의 FCF 성장률을 5년 내내 고정값으로 유지하다가 터미널가치 계산 시점(6년차)에 터미널성장률로
  갑자기 뚝 떨어뜨리던 문제 — 실제 성장은 절벽처럼 안 꺾이고 점진적으로 둔화되므로 이 방식은 가치를
  부풀리는 쪽으로 편향됐다. 이제 1년차 성장률에서 마지막 해(5년차) 터미널성장률까지 선형으로
  점감(fade)하도록 고쳐서, VEEV 실 데이터 기준 내재주가가 $281 → $226.64로 조정됨
(`market.normalize_financial_currency()`로 수정, 위 "알려진 제한사항" 참고).
