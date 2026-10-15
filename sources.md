# 출처와 구현 기준

확인일: **2026-10-13 (Asia/Seoul)**. 이 자료의 설명은 한국어 참고 글을 출발점으로 공식 문서, 실제 구현, 데이터 카드를 교차 확인해 작성했다. 문서와 main 브랜치는 변경될 수 있다.

| 자료 | 이 강의에서 확인한 내용 |
|---|---|
| [PyTorch 한국 사용자 모임 참고 글](https://discuss.pytorch.kr/t/unsloth-llm-laya-clef-jev/12158) | 주제, Jev/Laya/Clef 비교, 학습과 서빙 개요 |
| [TypeSafe: Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | Jev와 System One 개념 |
| [Unsloth: Train your own Decision Model](https://unsloth.ai/docs/new/train-your-own-decision-model-with-unsloth) | FastDecisionModel, DecisionTrainer, 학습·보정·저장·평가 API |
| [Unsloth: Local Decision API](https://unsloth.ai/docs/models/decision-laya) | `/v1/systemone`, 응답 형태, Laya와 모델 별칭 |
| [typed-decisions 데이터 카드](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | soft gold, train/test, 교사 자기 일치도와 평가 한계 |
| [Slidev 공식 가이드](https://sli.dev/guide/) | 프로젝트 실행과 빌드 |
| [PyTorch 설치 선택기](https://pytorch.org/get-started/locally/) | 서버 CUDA 환경별 PyTorch 설치 |

## 고정한 코드 참조

- Unsloth commit: `1e7d78415b93e4b9a036f07fdb953de11363e2dd`
  - [decision.py](https://github.com/unslothai/unsloth/blob/1e7d78415b93e4b9a036f07fdb953de11363e2dd/unsloth/models/decision.py)
  - [decision_from_lm.py](https://github.com/unslothai/unsloth/blob/1e7d78415b93e4b9a036f07fdb953de11363e2dd/unsloth/models/decision_from_lm.py)
  - [clef.py](https://github.com/unslothai/unsloth/blob/1e7d78415b93e4b9a036f07fdb953de11363e2dd/unsloth/models/clef.py): `build_layout`, `batched_logits` — 질문·선택지 span 평균 pooling, 상태당 본체 순전파 1회, 질문 필드의 TransformerDecoder 공동 디코딩
- 공식 notebook commit: `26a618143bb36ecb36050b1bcb38b602b64a2ee5`
  - [Qwen3.5 4B Decision notebook](https://github.com/unslothai/notebooks/blob/26a618143bb36ecb36050b1bcb38b602b64a2ee5/nb/Qwen3_5_(4B)-Decision.ipynb)
- PyPI 조회 당시 `unsloth`, `unsloth_zoo`: `2026.10.4`.
  코드 참조 SHA는 API 검토 기준이며 PyPI wheel과 동일하다는 주장은 아니다.
- 학습 시 모델/데이터 revision과 실제 패키지 버전을 `run_manifest.json`, `pip-freeze.txt`에 기록한다.

## 해석상 구분

1. Jev 자체와 Jev 호환 모델은 다르다. 이 실습은 Qwen 기반 결정 모델을 학습한다.
2. 기사/공식 문서의 성능 수치는 보고값이다. 이 프로젝트의 실행 결과가 아니다.
3. 데이터 카드의 `0.735`는 교사 자기 일치도이며 모든 모델의 이론적 정확도 상한으로 단정하지 않는다.
4. 새 무작위 결정 헤드 baseline은 Qwen 채팅 모델의 zero-shot 분류 성능이 아니다.
5. `confidence`와 선택지 확률을 구분하고, API 교체 시 동일 confidence 임계값의 재사용을 가정하지 않는다.
6. 01의 hard-label 지표와 02의 공식 soft-gold 지표는 정의를 확인하기 전까지 직접 비교하지 않는다.
7. GPU 학습 및 실제 모델 서빙은 작성 환경에서 실행하지 않았다. 검증 범위는 README에 기록한다.

## 라이선스 확인 위치

확인한 Unsloth 결정 모델 소스 파일에는 `SPDX-License-Identifier: AGPL-3.0-only`가 표기되어 있다.
배포 시에는 [해당 파일](https://github.com/unslothai/unsloth/blob/1e7d78415b93e4b9a036f07fdb953de11363e2dd/unsloth/models/decision.py),
[저장소 라이선스](https://github.com/unslothai/unsloth/blob/1e7d78415b93e4b9a036f07fdb953de11363e2dd/LICENSE),
선택 모델과 데이터의 각 라이선스를 확인한다. 이 프로젝트는 upstream 구현 파일을 복제하지 않고 API 호출 예제를 제공한다.

## 세션 Q&A 보충 자료 (2026-10-15)

슬라이드 19–32에 변환·일반화·백본/헤드·층 선택·입출력 연결을 추가했다. 아래 구현은 위와 같은 고정 커밋 기준이다.

- [Cloudflare 원본을 포함한 joint_schema_model.py](https://github.com/unslothai/unsloth/blob/1e7d78415b93e4b9a036f07fdb953de11363e2dd/unsloth/_vendor/clef/joint_schema_model.py)
  - `encode_record`: JSON을 입력 토큰과 질문·선택지 span/ID로 변환
  - `EvidenceRoutingLayer`, `JointSchemaHead`: 라우팅·공동 질문 디코더·선택지 채점
  - `ClefModel.forward`: 본체의 last_hidden_state를 헤드로 전달
  - `systemone_answer`, `systemone`: 질문별 softmax, ID 연결, 타입별 JSON 응답 구성
- [Layer by Layer: Uncovering Hidden Representations in Language Models](https://arxiv.org/abs/2502.02013)
  - 텍스트 임베딩 과제에서 중간 층이 마지막 층보다 유리할 수 있다는 연구 근거.
  - Clef에서의 층별 우열을 검증한 결과는 아니다. 다른 층을 사용할 때는 해당 표현에 맞춰 헤드를 다시 학습하고 동일 조건에서 평가해야 한다.

보충 슬라이드의 logits·확률은 설명용이며 실측 예측이 아니다. `choice`의 confidence 및 `score`의 변환은 참조 `systemone_answer` 코드 기준이며 모든 Jev 호환 모델에 같은 정의를 가정하지 않는다. 학습된 Clef 체크포인트를 로딩하는 경우와 일반 LLM에 새 무작위 헤드를 추가하는 경우도 구분한다.

가변 출력 보충(슬라이드 28–30): `clef.py`의 `forward_per_record`에서 `option_counts`, `owner`, `torch.split`로 질문별 선택지 표현을 관리하고 공유 `residual_scorer`로 선택지당 점수를 계산한다. 기본 최적화 경로는 `batched_logits`에서 패딩·마스킹·인덱싱하고 `ClefLogits`에서 각 질문의 유효 선택지 수만큼 복원한다. 슬라이드의 `Linear(W,1)`은 가변 행 수를 설명하는 최소 예시이며 실제 전체 채점식이 아니다.
