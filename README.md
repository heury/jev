# Jev와 결정 모델: 학습부터 검증까지

딥러닝 경험자와 **NVIDIA GPU 서버**를 위한 한국어 **59장 Slidev 강의(기본 110분 + 선택 심화 Q&A)** 및 독립 실행 가능한 Jupyter 노트북 3개.
[참고 글](https://discuss.pytorch.kr/t/unsloth-llm-laya-clef-jev/12158)과 공식 자료를 바탕으로 작성했다.

Jev는 API 모델이다. 이 실습에서는 **Unsloth로 Qwen을 Jev 호환 결정 모델로 학습**한다.

| 파일 | 내용 | 실행 환경 |
|---|---|---|
| [slides.md](slides.md) | 개념, 구조, 손실, 평가, 학습, 서빙; 발표자 노트 포함 | Node.js 22.12+ |
| [01_probabilities_and_metrics.ipynb](output/jupyter-notebook/01_probabilities_and_metrics.ipynb) | 합성 로짓, NLL/Brier/ECE, 온도 보정, 보류 정책 | Python CPU |
| [02_train_decision_model.ipynb](output/jupyter-notebook/02_train_decision_model.ipynb) | 실제 데이터, Qwen + LoRA + 결정 헤드 학습, 보정, 저장·재로딩 검증 | Linux NVIDIA GPU |
| [03_test_decision_api.ipynb](output/jupyter-notebook/03_test_decision_api.ipynb) | 응답 계약, HTTP 추론, 한국어 smoke 평가, 지연시간 | CPU 클라이언트 + 실제 모델 서버 |
| [sources.md](sources.md) | 출처, 코드 revision, 결과 해석과 라이선스 확인 위치 | 문서 |

## 강의 실행

```bash
npm ci
npm run dev
```

브라우저에서 `http://localhost:3030`을 연다. 발표자 화면은 `/presenter/`다.

```bash
npm run build                 # dist/에 정적 사이트
npm run export                # artifacts/jev-course.pdf
```

PDF는 빈 페이지 내보내기를 피하도록 페이지별 렌더링(`--per-slide --wait 700 --dark`)으로 생성한다.
PDF export에 필요한 Chromium이 없으면 `npx playwright install chromium`을 먼저 실행한다.
프런트엔드 패키지 버전은 package.json 및 package-lock.json에 고정했다.
한국어 글꼴은 로컬 Pretendard / Noto Sans KR 또는 시스템 sans-serif를 사용한다.

## CPU 실습 환경

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-notebooks.txt
.venv/bin/jupyter lab output/jupyter-notebook
```

01은 바로 전체 실행할 수 있다. 03은 기본적으로 **합성 fixture만 검사**한다.
실제 추론을 하려면 서버를 준비하고 `LIVE_API=True`로 변경한다. API 키는 환경변수 `TYPESAFE_API_KEY`에서 읽는다.

## NVIDIA GPU 서버 실습

1. Linux + Python 3.11 또는 3.12의 전용 환경을 만든다.
2. [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)로 드라이버/CUDA에 맞는 PyTorch를 설치한다.
3. 같은 환경에 `requirements-gpu.txt`를 설치한다. 기존 커널에 설치했다면 재시작한다.
4. `CUDA_VISIBLE_DEVICES=0 jupyter lab`으로 단일 GPU를 지정하고 02를 연다.
5. 위에서 아래로 실행한다. Unsloth는 transformers/peft보다 먼저 import한다.

```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
# 여기서 서버에 맞는 CUDA PyTorch를 먼저 설치
python -m pip install -r requirements-gpu.txt
CUDA_VISIBLE_DEVICES=0 jupyter lab output/jupyter-notebook
```

시작 계획으로 VRAM 16GB 이상, 디스크 30GB 이상 여유를 권장한다. 실측 최소 요구량을 보장하는 수치는 아니다.
기본값은 `unsloth/Qwen3.5-0.8B`, 4-bit, 길이 2048, batch 1, accumulation 8이다.
`SMOKE=True`는 원천 train 240행을 168/36/36행으로 나누고 test 40행, 60 steps를 실행한다.
`SMOKE=False`는 전체 데이터와 2 epochs를 사용한다. 4B 실험은 `MODEL_ID`를 변경한다.

Unsloth/Unsloth Zoo는 `2026.10.4`로 지정했지만 CUDA 전체 환경을 검증한 lockfile은 아니다.
환경에 따른 설치 문제는 공식 설치 문서와 오류 메시지를 확인한다. 02는 실제 설치 버전과 모델/데이터 SHA를 기록한다.
학습 완료 예상 시간과 정확도는 제시하지 않는다. 실제 실행에서 측정한다.

### 학습 결과

실행한 커널의 현재 디렉터리에 `artifacts/qwen-decisions-<UTC timestamp>/`가 생성된다.

- `run_manifest.json`: seed, 모델/데이터 SHA, 분할 해시, 학습 설정과 변환 보고서
- `pip-freeze.txt`: 설치 버전
- `metrics.json`: 학습 전후 validation, 보정 보고서, 최종 test
- `adapter/`: LoRA 및 결정 헤드
- `reload_check.json`: 저장 전/재로딩 후 확률 비교
- `merged/`: `EXPORT_MERGED=True`일 때 저장하는 16-bit 모델

원본 train/test state 중복을 검사하고, train/validation/calibration/test를 분리한다.
모든 노트북은 학습 체크포인트 업로드나 유료 API 호출을 자동으로 수행하지 않는다.

## 모델 서빙과 테스트

학습한 모델은 02에서 merged export 후 GPU 서버 터미널에서 실행한다.

```bash
UNSLOTH_SYSTEMONE_MODEL=/absolute/path/to/merged unsloth studio -H 127.0.0.1 -p 8888
```

Studio의 Settings → API → Decision API에서 Serve requests와 선택 모델을 확인한다.
학습 없이 API 실습만 하려면 Studio에서 Laya multilingual을 선택할 수 있다.
노트북 03의 `BASE_URL`, `MODEL`, `LIVE_API=True`를 설정한 뒤 실행한다.
`jev-latest`는 로컬 서버의 선택 모델 별칭이며 실제 Jev 서비스 호출을 뜻하지 않는다.

노트북이 다른 컴퓨터에서 실행된다면:

```bash
ssh -L 8888:127.0.0.1:8888 user@gpu-server
```

03은 합성 계약 검사, 실제 모델 추론, 지연시간 측정을 각각 구분해 출력한다.
6건 한국어 예시는 연결 확인용이며 모델의 일반 성능을 대표하지 않는다.

## 검증

```bash
.venv/bin/python scripts/verify_notebooks.py
npm run build
```

실제 공개 데이터 전처리 확인은 추가 의존성과 네트워크가 필요하다.

```bash
uv pip install --python .venv/bin/python 'datasets>=3,<5' 'huggingface_hub>=0.34,<2'
.venv/bin/python scripts/verify_data.py
```

2026-10-13 작성 환경(macOS, CPU)에서 확인한 범위:

| 확인 | 결과 |
|---|---|
| 세 노트북 nbformat 및 모든 Python 셀 구문 | 통과 |
| 01 전체 실행 | 통과 |
| 03 오프라인 전체 실행 | 통과 |
| 응답 이상 사례 6개 거부 | 통과 |
| 로컬 HTTP fixture 서버 요청/인증 헤더/401 처리 | 통과 — 실제 모델 아님 |
| 실제 typed-decisions 다운로드 및 02의 전처리 셀 | 통과 — train 1,200 / test 400행, exact-state 교차 중복 없음 |
| Slidev production build | 통과 |
| NVIDIA GPU 학습·보정·모델 재로딩 | **미실행** |
| 실제 Laya/Qwen Decision API 추론 | **미실행** |

CPU 실행본과 데이터 검증 기록은 `artifacts/verification/`에 저장된다. GPU/실제 API 검증은 해당 서버에서 수행해야 한다.

노트북 원본 내용은 `scripts/build_notebooks.py`로 재생성할 수 있다. 실행하면 기존 셀과 출력을 초기화한다.
개별 `.ipynb`를 직접 편집한 경우 재생성 전에 변경을 스크립트에 반영한다.

현재 [강의 PDF](artifacts/jev-course.pdf)는 전체 59장을 포함한다.
슬라이드 19–32는 이 세션의 심화 Q&A 14장으로, LLM→결정 모델 변환, 새 질문 일반화, 백본·헤드 구조, 층 선택, JSON 입력·토큰 구간·logits·응답 ID 연결을 다룬다.
슬라이드 28–30은 본체 순전파 1회와 질문별 채점, 고정 벡터 차원과 가변 선택지 수, 공유 채점기·선택지 분리 구현을 예제로 설명한다.
기본 110분에서는 필요한 장만 선택하고, 모두 설명하면 약 16분을 추가한다. 심화 자료의 코드·연구 출처는 `sources.md`에 정리했다.

현재 고정된 Slidev 53.0.0 의존성 조합은 개발/내보내기 콘솔에
`Failed to patch FloatingVue` 메시지를 출력한다. upstream Twoslash tooltip의 내부 패치에서 발생하며,
이 강의에서 사용한 일반 코드 하이라이트·수식·슬라이드 탐색·PDF 생성은 동작 확인했다.
강의는 Twoslash 기능을 사용하지 않는다.
