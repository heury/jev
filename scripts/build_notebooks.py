"""Rebuild lesson cells while preserving the skill scaffold's notebook metadata."""
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/jupyter-notebook'

def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(s).strip() + '\n'}

def code(s):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': textwrap.dedent(s).strip() + '\n'}

def save(name, cells):
    path = OUT / name
    nb = json.loads(path.read_text())
    for i, cell in enumerate(cells):
        cell['id'] = f'lesson-{i:03d}'
    nb['cells'] = cells
    nb['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb['metadata']['language_info'] = {'name': 'python', 'version': '3.11'}
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')

METRICS = '''
import math

def softmax(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = [v / temperature for v in logits]
    exps = [math.exp(v - max(z)) for v in z]
    return [v / sum(exps) for v in exps]

def metrics(probabilities, labels, bins=5):
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("non-empty aligned predictions and labels required")
    if bins < 1:
        raise ValueError("bins must be positive")
    for p, y in zip(probabilities, labels):
        if (not p or not all(math.isfinite(v) and 0 <= v <= 1 for v in p)
                or not math.isclose(sum(p), 1, abs_tol=1e-6)
                or not isinstance(y, int) or not 0 <= y < len(p)):
            raise ValueError("invalid distribution or label")
    predicted = [max(range(len(p)), key=p.__getitem__) for p in probabilities]
    correct = [int(a == b) for a, b in zip(predicted, labels)]
    confidence = [max(p) for p in probabilities]
    n = len(labels)
    nll = -sum(math.log(max(p[y], 1e-12)) for p, y in zip(probabilities, labels)) / n
    # Multiclass Brier: sum over classes, then mean over samples (NOT divided by K).
    brier = sum(sum((v - int(k == y)) ** 2 for k, v in enumerate(p))
                for p, y in zip(probabilities, labels)) / n
    reliability = []
    for b in range(bins):
        ids = [i for i, c in enumerate(confidence) if min(int(c * bins), bins - 1) == b]
        if ids:
            acc = sum(correct[i] for i in ids) / len(ids)
            conf = sum(confidence[i] for i in ids) / len(ids)
            reliability.append((b, len(ids), acc, conf))
    ece = sum(count / n * abs(acc - conf) for _, count, acc, conf in reliability)
    return dict(accuracy=sum(correct)/n, nll=nll, brier=brier, ece=ece,
                reliability=reliability)
'''

save('01_probabilities_and_metrics.ipynb', [
md('''
# 01 · 결정 모델의 확률과 평가
**대상:** 딥러닝 경험자 · **시간:** 15분 · **환경:** Python 3.11+ CPU, 표준 라이브러리만 사용.

목표: `noul` / `choice` / `score` 해석, accuracy와 calibration 구분, validation에서 보류 임계값 선택.
이 노트북의 모든 로짓·확률은 **교육용 합성 값**이다. Jev 또는 학습 모델의 추론 결과가 아니다.

순서: 1. 확률 변환 → 2. 질문 유형 → 3. 평가 → 4. 온도 보정 → 5. 보류 정책.
'''), code(METRICS),
md(r'''## 1. 선택지 로짓을 확률로
$p_k=\exp(z_k/T)/\sum_j\exp(z_j/T)$. 양의 스칼라 온도는 argmax를 바꾸지 않지만 확률의 크기를 바꾼다.
훈련에서 soft label $q$를 쓰면 CE는 $-\sum_k q_k\log p_k$다.'''),
code('''
labels = ['billing', 'technical', 'sales']
logits = [2.2, 0.8, -0.3]
for temperature in [0.5, 1.0, 2.0]:
    print(temperature, dict(zip(labels, [round(p, 4) for p in softmax(logits, temperature)])))
q = [0.7, 0.2, 0.1]
p = softmax(logits)
print('soft-target CE:', -sum(a * math.log(b) for a, b in zip(q, p)))
'''),
md('''## 2. 세 질문 유형
`noul=0.8`은 yes 확률이다. `score`는 순서 있는 등급의 **기댓값**이며 argmax 등급과 다르다.
서비스의 `confidence` 필드는 모델별 정의가 다르므로 선택지 확률과 혼동하지 않는다.'''),
code('''
p_yes = 0.8
urgency = {'0': 0.1, '1': 0.2, '2': 0.7}
expected_score = sum(int(k) * v for k, v in urgency.items())
print('noul:', p_yes, '| no:', 1 - p_yes)
print('score expectation:', expected_score, '| most likely level:', max(urgency, key=urgency.get))
assert math.isclose(expected_score, 1.6)
'''),
md('''## 3. Accuracy만 같으면 같은 모델일까?
아래 두 예측은 모두 75% 정확도지만 잘못된 예측의 확신이 다르다.
여기서는 hard label NLL, 클래스 합 방식 Brier, top-label ECE(5개 고정 폭 bin)를 사용한다.
공식 데이터의 soft-gold 지표와 수치를 직접 비교하지 않는다. 작은 표본의 ECE는 특히 불안정하다.'''),
code('''
y = [0, 1, 0, 1]
moderate = [[.7, .3], [.3, .7], [.7, .3], [.7, .3]]
overconfident = [[.99, .01], [.01, .99], [.99, .01], [.99, .01]]
for name, probs in [('moderate', moderate), ('overconfident', overconfident)]:
    print(name, metrics(probs, y))
assert metrics(overconfident, y)['nll'] > metrics(moderate, y)['nll']
'''),
md('''## 4. 보정 데이터에서만 온도 선택
별도 calibration 데이터의 NLL로 T를 선택하고, 고정한 T를 test에 적용한다.
합성 예시에서 개선되더라도 실제 데이터에서의 개선을 보장하지 않는다.
Unsloth의 실제 보정은 02에서 `FastDecisionModel.calibrate`로 수행한다.'''),
code('''
cal_logits = [[4., 0.], [0., 4.], [4., 0.], [4., 0.]]
cal_labels = [0, 1, 0, 1]
test_logits = [[3., 0.], [0., 3.], [3., 0.], [0., 3.]]
test_labels = [0, 1, 1, 1]
candidates = [0.5, 1., 1.5, 2., 3., 4., 6.]
best_t = min(candidates, key=lambda t: metrics([softmax(z, t) for z in cal_logits], cal_labels)['nll'])
print('calibration-selected T:', best_t)
for t in [1., best_t]:
    print('test, T=', t, metrics([softmax(z, t) for z in test_logits], test_labels))
'''),
md('''## 5. 실습: coverage와 선택적 정확도
확률이 임계값보다 작은 입력은 사람 검토로 보낸다고 가정한다. 다음 함수에서 coverage는 자동 처리 비율,
selective accuracy는 자동 처리된 사례에서의 정답률이다. 한 건도 없으면 정확도는 `None`이다.

**문제:** validation에서 정확도 90% 이상을 만족하면서 coverage가 가장 큰 임계값을 고르자.
이 보장은 작은 validation 표본 안에서만 성립한다. 실제 서비스에는 독립 테스트와 충분한 표본이 필요하다.'''),
code('''
def selective_metrics(probs, labels, threshold):
    ids = [i for i, p in enumerate(probs) if max(p) >= threshold]
    correct = sum(max(range(len(probs[i])), key=probs[i].__getitem__) == labels[i] for i in ids)
    return {'coverage': len(ids)/len(probs), 'accuracy': correct/len(ids) if ids else None}

validation_probs = [[.95,.05], [.1,.9], [.8,.2], [.4,.6], [.55,.45]]
validation_labels = [0,1,1,1,0]
results = [(t, selective_metrics(validation_probs, validation_labels, t)) for t in [.5,.7,.85,.95,1.]]
print(results)
# 해설: 통과 후보 중 coverage 최대. 테스트 라벨을 이 선택에 사용하지 않는다.
feasible = [(t, r) for t, r in results if r['accuracy'] is not None and r['accuracy'] >= .9]
chosen = max(feasible, key=lambda item: item[1]['coverage']) if feasible else None
print('selected:', chosen)
assert chosen[0] == .85
assert selective_metrics(validation_probs, validation_labels, 1.)['accuracy'] is None
'''),
md('''## 다음 단계
02에서 합성 로짓 대신 실제 Qwen 결정 모델을 학습한다. 분포 이동, 한국어 데이터, 선택지 순서를 바꾸어
accuracy와 calibration이 함께 유지되는지 확인하자.

출처: [Unsloth 결정 모델](https://unsloth.ai/docs/new/train-your-own-decision-model-with-unsloth),
[typed-decisions 데이터 카드](https://huggingface.co/datasets/LocalLLaMA/typed-decisions).
''')])

save('02_train_decision_model.ipynb', [
md('''
# 02 · Qwen을 Jev 호환 결정 모델로 학습
**대상:** 딥러닝 경험자 · **환경:** Linux, Python 3.11/3.12, 단일 NVIDIA CUDA GPU.
**권장 시작 환경:** VRAM 16GB 이상, 디스크 30GB 이상 여유. 이는 실습 계획값이며 실제 사용량은 아래에서 측정한다.
Jev 자체의 비공개 가중치를 학습하는 것이 아니다. Unsloth가 공개 Qwen 본체에 결정 헤드를 붙인다.

순서: 환경 → 원본 행 분할 → LoRA/헤드 학습 → calibration → 독립 test → 저장 → 재로딩 → 예측 비교.
기본값은 0.8B / 60 steps의 동작 확인용이다. 공식 벤치마크 재현 실험이 아니다.
이 파일 하나만 서버로 옮겨도 실행된다. 결과는 현재 작업 폴더의 `artifacts/` 아래에 저장된다.

이 자료 작성 환경에는 NVIDIA GPU가 없어 **실제 학습은 미실행**이다. API와 셀 구문은 공식 소스에 대조했다.
'''),
md('''## 1. 설치 (필요할 때 한 번)
먼저 [PyTorch 공식 선택기](https://pytorch.org/get-started/locally/)에 맞는 CUDA PyTorch를 설치한다.
아래 패키지 버전은 2026-10-13에 확인한 배포 버전이며 GPU 전체 의존성 lock은 아니다.
새 환경에서 `INSTALL=True`로 실행한 후 **커널을 재시작하고 False로 복원**한다.
Unsloth는 transformers/peft보다 먼저 import해야 한다. 기존 환경을 덮어쓰기보다 전용 환경을 사용한다.'''),
code('''
INSTALL = False
if INSTALL:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           'unsloth==2026.10.4', 'unsloth_zoo==2026.10.4',
                           'datasets>=3,<5', 'huggingface_hub>=0.34,<2'])
    # 현재 CUDA PyTorch를 사용해 최적화 convolution 커널을 빌드한다.
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           '--no-build-isolation', '--no-deps', 'causal-conv1d==1.7.0'])
    print('설치 완료. 커널을 재시작한 뒤 INSTALL=False로 실행하세요.')
'''),
code('''
# transformers / peft / torch보다 먼저 Unsloth를 import한다.
import importlib.util
if importlib.util.find_spec('unsloth') is None:
    raise RuntimeError('Unsloth가 없습니다. 설치 셀 또는 requirements-gpu.txt를 사용하세요.')
from unsloth import FastDecisionModel, DecisionTrainer, is_bfloat16_supported
import torch
import inspect, json, os, random, platform, subprocess, sys, time, hashlib, gc
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import TrainingArguments, set_seed

if not torch.cuda.is_available():
    raise RuntimeError('이 학습 노트북은 NVIDIA CUDA GPU가 필요합니다. CPU 실습은 01을 사용하세요.')
required = ['build_dataset', 'evaluate', 'calibrate', 'predict', 'for_training']
assert all(hasattr(FastDecisionModel, name) for name in required), '결정 모델 API가 없는 Unsloth 버전'
assert 'processing_class' in inspect.signature(DecisionTrainer.__init__).parameters or any(
    p.kind == p.VAR_KEYWORD for p in inspect.signature(DecisionTrainer.__init__).parameters.values())
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
print('torch / CUDA:', torch.__version__, torch.version.cuda)
print('Unsloth:', version('unsloth'))
'''),
md('''## 2. 실행 설정과 재현 정보
`SMOKE=True`는 입력 240행, 테스트 40행, 60 steps. `False`는 전체 train/test와 2 epochs.
기본 본체는 0.8B이며 `MODEL_ID`를 4B로 바꿀 수 있다. 메모리는 batch=1부터 확인한다.
여러 GPU가 보이더라도 이 실습은 단일 GPU 기준이다. 서버에서 `CUDA_VISIBLE_DEVICES=0`으로 커널을 시작한다.
재현 시 아래 revision 값을 이전 `run_manifest.json`의 SHA로 채운다.'''),
code('''
SEED = 3407
SMOKE = True
MODEL_ID = 'unsloth/Qwen3.5-0.8B'
MODEL_REVISION = None  # 이전 manifest의 model_revision SHA를 넣으면 같은 가중치를 사용
DATASET_ID = 'LocalLLaMA/typed-decisions'
DATASET_REVISION = None
MAX_SEQ_LENGTH = 2048
set_seed(SEED)
api = HfApi()
model_revision = MODEL_REVISION or api.model_info(MODEL_ID).sha
resolved_dataset_revision = DATASET_REVISION or api.dataset_info(DATASET_ID).sha
RUN_DIR = Path('artifacts') / ('qwen-decisions-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
RUN_DIR.mkdir(parents=True, exist_ok=False)
manifest = dict(seed=SEED, smoke=SMOKE, model=MODEL_ID, model_revision=model_revision,
                dataset=DATASET_ID, dataset_revision=resolved_dataset_revision,
                max_seq_length=MAX_SEQ_LENGTH, python=platform.python_version(),
                gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                versions={p: version(p) for p in ['torch','unsloth','unsloth_zoo','transformers','datasets','peft']})
(RUN_DIR / 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
print('outputs:', RUN_DIR.resolve())
'''),
md('''## 3. 데이터 로드와 누수 검사
데이터는 `state`, `questions`, `gold`를 포함한다. JSON 문자열인 필드를 파싱한다.
`gold`, `factors`, `label_agreement`는 모델 입력에 넣지 않는다.
동일 state의 모든 질문은 같은 분할에 둔다. exact-state 중복을 제거하지만 의미상 유사 문장까지 검출하지는 않는다.
train/test 사이 state가 중복되면 학습을 중단한다. 커스텀 데이터는 고객/문서/시간 그룹 단위 분할도 필요하다.'''),
code('''
def parsed(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value

def state_key(row):
    canonical = json.dumps(parsed(row['state']), ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()

def unique_rows(rows):
    found = {}
    for row in rows:
        found.setdefault(state_key(row), row)
    return list(found.values())

raw_train = load_dataset(DATASET_ID, 'all', split='train', revision=resolved_dataset_revision)
raw_test = load_dataset(DATASET_ID, 'all', split='test', revision=resolved_dataset_revision)
train_rows, test_rows = unique_rows(raw_train), unique_rows(raw_test)
assert not ({state_key(r) for r in train_rows} & {state_key(r) for r in test_rows}), 'train/test state leakage'
print('deduplicated:', len(raw_train)-len(train_rows), len(raw_test)-len(test_rows))
rng = random.Random(SEED)
rng.shuffle(train_rows)
rng.shuffle(test_rows)
if SMOKE:
    train_rows, test_rows = train_rows[:240], test_rows[:40]
assert len(train_rows) >= 30 and test_rows, '분할하기에 데이터가 부족합니다.'
a, b = int(len(train_rows)*.7), int(len(train_rows)*.85)
raw_splits = {'train': train_rows[:a], 'validation': train_rows[a:b],
              'calibration': train_rows[b:], 'test': test_rows}
keys = {name: {state_key(r) for r in rows} for name, rows in raw_splits.items()}
for i, name in enumerate(keys):
    for other in list(keys)[i+1:]:
        assert keys[name].isdisjoint(keys[other]), (name, other)
manifest['split_state_hashes'] = {name: sorted(values) for name, values in keys.items()}
print({name: len(rows) for name, rows in raw_splits.items()})
sample = raw_splits['train'][0]
print('state preview:', str(parsed(sample['state']))[:250])
print('questions:', json.dumps(parsed(sample['questions']), ensure_ascii=False, indent=2)[:1500])
print('gold:', str(parsed(sample['gold']))[:500])
'''),
md('''## 4. 모델과 LoRA
4-bit는 본체 저장/계산 메모리를 줄인다. LoRA는 본체의 업데이트를 학습하고 새 결정 헤드도 학습한다.
새 헤드이므로 아래 학습 전 수치는 **무학습 결정 헤드 baseline**이다. Qwen의 채팅/JSON zero-shot 성능이 아니다.
실제 trainable parameter 수와 메모리를 출력해 가정을 확인한다.'''),
code('''
model, tokenizer = FastDecisionModel.from_pretrained(
    model_name=MODEL_ID, revision=model_revision,
    max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True,
)
model = FastDecisionModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0,
    use_gradient_checkpointing='unsloth', random_state=SEED,
)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print({'trainable_parameter_elements': trainable, 'all_parameter_elements': total})
print('4-bit 파라미터의 저장 표현 때문에 위 수는 원본 모델의 정확한 parameter count와 다를 수 있습니다.')
'''),
md('''## 5. 토큰화와 변환 보고서
Unsloth 변환기는 잘못된 질문을 건너뛸 수 있다. 조용히 성능 비교에서 빠지지 않게 skipped가 있으면 중단한다.
truncated는 별도로 기록한다. 길이에 걸린 state 끝의 정답 근거가 사라졌는지 확인한다.
행 수와 질문 수는 다르다. Clef식 항목은 한 행에 여러 질문을 묶을 수 있다.'''),
code('''
items, reports = {}, {}
for name, rows in raw_splits.items():
    items[name], reports[name] = FastDecisionModel.build_dataset(rows, tokenizer, model)
    print(name, 'items=', len(items[name]), 'report=', reports[name])
    assert items[name], f'{name}: no usable items'
    assert reports[name]['skipped'] == 0, f'{name}: inspect skipped questions before continuing'
manifest['build_reports'] = reports
(RUN_DIR / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
FastDecisionModel.for_inference(model)
before = FastDecisionModel.evaluate(model, tokenizer, items['validation'], batch_size=1)
print('validation / random head:', before)
'''),
md('''## 6. 실제 학습
본체 LoRA LR=2e-4, 헤드 LR=1e-4를 명시한다. batch=1, accumulation=8은 단일 GPU에서 유효 batch 8.
validation으로 진행을 관찰하고 calibration/test는 학습에 넣지 않는다.
OOM이면 길이·batch·모델 크기를 낮추고, 변경 내용을 manifest에 기록한다.
아래 셀은 실제 GPU 계산을 시작한다.'''),
code('''
FastDecisionModel.for_training(model)
args = TrainingArguments(
    output_dir=str(RUN_DIR / 'checkpoints'),
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    max_steps=60 if SMOKE else -1, num_train_epochs=2,
    warmup_steps=0.1, learning_rate=2e-4, lr_scheduler_type='cosine', weight_decay=0.01,
    bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
    eval_strategy='steps', eval_steps=20, logging_steps=10,
    save_strategy='steps', save_steps=20, save_total_limit=2,
    report_to='none', seed=SEED, data_seed=SEED,
)
trainer = DecisionTrainer(
    model=model, processing_class=tokenizer,
    train_dataset=items['train'], eval_dataset=items['validation'],
    head_learning_rate=1e-4, args=args,
)
manifest['training_args'] = args.to_dict()
manifest['head_learning_rate'] = 1e-4
(RUN_DIR / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
train_result = trainer.train()
torch.cuda.synchronize()
print('elapsed seconds:', round(time.perf_counter() - started, 1))
print('peak reserved GiB:', round(torch.cuda.max_memory_reserved() / 2**30, 2))
print(train_result.metrics)
'''),
md('''## 7. 학습 전후 → 보정 → 최종 test
보정 함수가 calibration split에서 추정한 온도를 모델 설정에 저장한다.
공식 evaluate 지표의 이름/정의는 설치한 버전을 따른다. 결과 dict를 그대로 저장한다.
최종 test는 이 단계에서만 읽고, test 결과를 보고 설정을 고르면 별도 최종 데이터가 다시 필요하다.'''),
code('''
FastDecisionModel.for_inference(model)
after = FastDecisionModel.evaluate(model, tokenizer, items['validation'], batch_size=1)
calibration = FastDecisionModel.calibrate(model, tokenizer, items['calibration'], batch_size=1)
calibrated_validation = FastDecisionModel.evaluate(model, tokenizer, items['validation'], batch_size=1)
final_test = FastDecisionModel.evaluate(model, tokenizer, items['test'], batch_size=1)
results = {'validation_before_random_head': before, 'validation_after_training': after,
           'calibration_fit_report': calibration, 'validation_after_calibration': calibrated_validation,
           'test_after_calibration': final_test, 'train': train_result.metrics}
print(json.dumps(results, indent=2, default=str))
(RUN_DIR / 'metrics.json').write_text(json.dumps(results, indent=2, default=str))
'''),
md('''## 8. 새 입력 추론과 저장
한국어 예시는 정량 평가가 아닌 전이 동작 관찰이다. 영어 합성 학습 데이터에서의 성능이 한국어로 보장되지는 않는다.
어댑터 저장에는 결정 헤드도 포함된다. tokenizer는 Unsloth의 결정 모델 저장 경로가 처리한다.
merged export는 16-bit 모델로 커지므로 필요한 경우에만 실행한다.'''),
code('''
questions = {
    'team': {'type':'choice', 'instructions':'담당 팀을 선택하세요.',
             'criteria':{'billing':'결제 및 환불', 'technical':'장애 및 오류', 'sales':'가격 및 계약'}},
    'refund': {'type':'noul', 'instructions':'고객이 환불을 요청합니까?'},
    'urgency': {'type':'score', 'instructions':'처리 긴급도', 'criteria':['여유 있음','곧 처리','오늘 처리']},
}
state = '어제 결제가 두 번 되었습니다. 오늘 중복 결제 금액을 환불해 주세요.'
prediction_before_save = FastDecisionModel.predict(model, tokenizer, state, questions)
print(json.dumps(prediction_before_save, ensure_ascii=False, indent=2))
ADAPTER_DIR = RUN_DIR / 'adapter'
model.save_pretrained(str(ADAPTER_DIR))
EXPORT_MERGED = False
if EXPORT_MERGED:
    model.save_pretrained_merged(str(RUN_DIR / 'merged'))
print('saved:', ADAPTER_DIR.resolve())
'''),
md('''## 9. GPU 메모리 해제 후 재로딩 검증
두 모델을 동시에 GPU에 두지 않는다. 어댑터 로딩은 원본 본체를 다시 가져오므로 원본 모델 접근이 필요하다.
양자화/직렬화로 작은 차이가 생길 수 있다. 확률 최대 차이와 선택 결과를 출력하고 허용오차를 검사한다.
실패하면 통과로 간주하지 말고 온도 저장, revision, dtype과 패키지 버전을 확인한다.'''),
code('''
del trainer, model, tokenizer
gc.collect()
torch.cuda.empty_cache()
model, tokenizer = FastDecisionModel.from_pretrained(
    model_name=str(ADAPTER_DIR), max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True,
)
FastDecisionModel.for_inference(model)
prediction_after_load = FastDecisionModel.predict(model, tokenizer, state, questions)
max_delta = 0.0
for name, previous in prediction_before_save.items():
    current = prediction_after_load[name]
    assert previous['answer'] == current['answer'], f'{name}: answer changed after reload'
    assert set(previous['probabilities']) == set(current['probabilities'])
    max_delta = max(max_delta, max(abs(float(p)-float(current['probabilities'][k]))
                                  for k, p in previous['probabilities'].items()))
print('max probability delta:', max_delta)
assert max_delta < 0.02, '저장/재로딩 확률 차이를 조사하세요.'
(RUN_DIR / 'reload_check.json').write_text(json.dumps({'max_delta':max_delta, 'passed':True}))
'''),
md('''## 10. 과제와 실험 설계
1. 같은 데이터/revision/seed에서 60 steps와 2 epochs를 비교한다.
2. 한국어 문의와 정답을 별도 수집해 test를 만든다. 영어 test 결과와 나눠 보고한다.
3. 선택지 순서를 바꾼 요청의 확률을 키로 정렬해 비교한다.
4. 최소 3개 seed와 더 큰 test에서 변동성을 확인한다. 40행 smoke test로 모델 우열을 주장하지 않는다.

**해설 가이드:** 검증 지표로 설정을 선택 → calibration split으로 보정 → 최종 test 한 번 →
accuracy, soft-gold loss, ECE와 샘플 수, GPU/지연시간을 함께 보고한다.

### 로컬 서빙으로 연결
`EXPORT_MERGED=True`로 저장한 뒤 GPU 서버 터미널에서 (실제 절대경로로 변경):
```bash
UNSLOTH_SYSTEMONE_MODEL=/absolute/path/to/merged unsloth studio -H 127.0.0.1 -p 8888
```
Studio Settings → API → Decision API에서 Serve requests를 켠다. 이어서 03을 실행한다.
다른 PC의 노트북에서 접속할 때 `ssh -L 8888:127.0.0.1:8888 user@gpu-server`로 터널을 연다.
기존 Laya 선택과 새 학습 모델 선택은 별도 작업이며 03에서는 실제 응답 model도 기록한다.

출처: [학습 가이드](https://unsloth.ai/docs/new/train-your-own-decision-model-with-unsloth),
[공식 노트북](https://github.com/unslothai/notebooks/blob/26a618143bb36ecb36050b1bcb38b602b64a2ee5/nb/Qwen3_5_(4B)-Decision.ipynb),
[decision.py](https://github.com/unslothai/unsloth/blob/1e7d78415b93e4b9a036f07fdb953de11363e2dd/unsloth/models/decision.py),
[데이터 카드](https://huggingface.co/datasets/LocalLLaMA/typed-decisions).
''')])

API_HELPERS = '''
import json, math, os, time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

def validate_answers(response, questions):
    answers = response.get('answers')
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError('missing or unexpected answer keys')
    def probability(value):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('probability must be finite and in [0,1]')
    for name, question in questions.items():
        answer = answers[name]
        kind = question['type']
        if answer.get('type') != kind:
            raise ValueError(f'{name}: wrong answer type')
        if kind == 'noul':
            probability(answer['noul'])
            continue
        probs = answer['probabilities']
        expected = set(question['criteria']) if kind == 'choice' else {str(i) for i in range(len(question['criteria']))}
        if set(probs) != expected:
            raise ValueError(f'{name}: wrong probability keys')
        for p in probs.values():
            probability(p)
        if not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-3):
            raise ValueError(f'{name}: probabilities do not sum to one')
        if kind == 'choice':
            chosen = answer['choice']
            if chosen not in probs or probs[chosen] < max(probs.values()) - 1e-6:
                raise ValueError(f'{name}: choice does not match argmax')
        elif kind == 'score':
            score = answer['score']
            expected_score = sum(int(k) * p for k, p in probs.items())
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not math.isclose(score, expected_score, abs_tol=1e-3):
                raise ValueError(f'{name}: score is not the expected level')
        else:
            raise ValueError(f'unsupported question type: {kind}')
    return answers

def request_decision(base_url, model, state, questions, api_key='', timeout=180):
    body = json.dumps({'model':model, 'state':state, 'questions':questions}, ensure_ascii=False).encode()
    headers = {'Content-Type':'application/json'}
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    request = Request(base_url.rstrip('/') + '/v1/systemone', data=body, headers=headers, method='POST')
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as result:
            response = json.load(result)
    except HTTPError as exc:
        raise RuntimeError(f'HTTP {exc.code}: API 활성화, 모델 및 access token을 확인하세요.') from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError('서버 주소/SSH 터널/모델 로딩 시간/timeout을 확인하세요.') from exc
    elapsed = time.perf_counter() - started
    validate_answers(response, questions)
    return response, elapsed
'''

save('03_test_decision_api.ipynb', [
md('''
# 03 · Jev 호환 HTTP API 테스트
**환경:** Python 3.11+ 표준 라이브러리, 로컬 Unsloth Studio 서버 (Laya 또는 02에서 학습한 모델).
**목표:** 요청/응답 계약, 실제 추론, cold/warm 지연시간, 소규모 라벨 평가와 보류 정책.

기본 `LIVE_API=False`에서는 **합성 fixture 계약 검사만** 수행한다. 실제 모델을 호출하려면 서버를 켜고 True로 바꾼다.
서버가 없을 때 가짜 추론 결과로 대체하지 않는다. 이 파일 하나만 옮겨 실행할 수 있다.

### 준비
1. Unsloth Studio → Settings → API → Decision API → Serve requests.
2. Laya multilingual 또는 02의 merged 모델을 선택한다.
3. Access token을 환경변수 `TYPESAFE_API_KEY`에 설정한다. 노트북에 키를 저장하지 않는다.
4. GPU 서버가 원격이면 SSH 터널을 열고 base URL은 localhost로 둔다.

02의 모델을 쓰려면 GPU 서버에서:
```bash
UNSLOTH_SYSTEMONE_MODEL=/absolute/path/to/merged unsloth studio -H 127.0.0.1 -p 8888
```
`jev-latest`는 이 로컬 서버에서 선택 모델에 대한 별칭이다. 실제 TypeSafe Jev 호출이라는 뜻이 아니다.
'''), code(API_HELPERS),
code('''
LIVE_API = False
BASE_URL = os.environ.get('TYPESAFE_BASE_URL', 'http://127.0.0.1:8888')
API_KEY = os.environ.get('TYPESAFE_API_KEY', '')
MODEL = 'jev-latest'
questions = {
    'team': {'type':'choice', 'instructions':'담당 팀은?',
             'criteria':{'billing':'결제와 환불', 'technical':'장애와 오류', 'sales':'가격과 계약'}},
    'refund': {'type':'noul', 'instructions':'환불 요청이 있는가?'},
    'urgency': {'type':'score', 'instructions':'긴급도는?', 'criteria':['일반','곧 처리','오늘 처리']},
}
state = '결제가 두 번 됐어요. 오늘 중복 금액을 환불해 주세요.'
print({'mode': 'LIVE MODEL' if LIVE_API else 'OFFLINE CONTRACT CHECK ONLY', 'base_url':BASE_URL, 'model_alias':MODEL})
'''),
md('''## 1. 오프라인 계약 검사
fixture는 직접 작성한 교육용 확률이다. 합계, 선택 키, score 기댓값과 noul 범위를 검사한다.
실제 모델 정확도와는 관련 없다. confidence는 공급자별 의미가 달라 라우팅 계산에 쓰지 않는다.'''),
code('''
fixture = {'model':'synthetic-fixture-NOT-A-MODEL', 'answers':{
    'team': {'type':'choice','choice':'billing','probabilities':{'billing':.8,'technical':.15,'sales':.05}},
    'refund': {'type':'noul','noul':.9},
    'urgency': {'type':'score','score':1.6,'probabilities':{'0':.1,'1':.2,'2':.7}},
}}
validate_answers(fixture, questions)
print('합성 응답 계약 검사 통과. 실제 추론은 아직 수행하지 않았습니다.')
'''),
md('''## 2. 실제 한 번의 요청
첫 요청은 모델 로딩 시간을 포함할 수 있다. 서버가 이미 준비된 상태라면 cold-start 측정은 아니다.
HTTP 에러는 그대로 실패시킨다. 선택된 모델 이름은 응답에서 확인한다.'''),
code('''
if LIVE_API:
    response, first_latency = request_decision(BASE_URL, MODEL, state, questions, API_KEY)
    print('first request seconds:', round(first_latency, 3))
    print(json.dumps(response, ensure_ascii=False, indent=2))
else:
    print('SKIPPED: 실제 추론을 하려면 LIVE_API=True로 변경하세요.')
'''),
md('''## 3. Warm latency와 소규모 정확도
아래 직접 작성한 한국어 문의는 세 팀마다 두 건이다. 통계적 벤치마크가 아니라 연결/동작 확인용이다.
측정은 client end-to-end 순차 요청 기준이고 서버 큐/네트워크/토큰화를 포함한다.
같은 입력 반복은 캐시 영향을 받을 수 있다. 공식 성능 수치와 직접 비교하지 않는다.'''),
code('''
cases = [
    ('중복 청구된 금액을 환불해 주세요.', 'billing'),
    ('이번 달 청구서 금액이 잘못됐습니다.', 'billing'),
    ('로그인하면 오류 코드가 나오고 접속이 안 됩니다.', 'technical'),
    ('업데이트 후 앱이 계속 종료됩니다.', 'technical'),
    ('기업용 요금제와 계약 조건을 알려 주세요.', 'sales'),
    ('팀 계정 50개의 신규 견적을 받고 싶습니다.', 'sales'),
]
if LIVE_API:
    import statistics
    records = []
    for text, gold in cases:
        result, seconds = request_decision(BASE_URL, MODEL, text, questions, API_KEY)
        answer = result['answers']['team']
        records.append({'gold':gold, 'prediction':answer['choice'],
                        'probabilities':answer['probabilities'], 'latency_s':seconds,
                        'served_model':result.get('model')})
    ordered = sorted(r['latency_s'] for r in records)
    summary = {'n':len(records), 'accuracy':sum(r['gold']==r['prediction'] for r in records)/len(records),
               'p50_s':statistics.median(ordered), 'p95_nearest_rank_s':ordered[math.ceil(.95*len(ordered))-1]}
    print(summary)
    print('6건의 p95는 사실상 최대값이므로 안정적인 성능 지표가 아닙니다.')
    from pathlib import Path
    from datetime import datetime, timezone
    output = Path('artifacts') / ('api-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({'summary':summary,'records':records}, ensure_ascii=False, indent=2))
    print('saved:', output.resolve())
else:
    print('SKIPPED: 라이브 라벨 평가와 지연시간 측정')
'''),
md('''## 4. 보류 정책과 연습
여기서 .85는 정책 흐름 설명용이다. 실제 기준값은 별도의 validation으로 선택하고 test에서는 고정한다.
이 예제는 라우팅 결과를 출력할 뿐 환불 등의 외부 작업을 실행하지 않는다.

**문제:** '환불은 원하지 않고 오류만 고쳐 주세요'를 추가하고 noul과 team이 어떻게 달라지는지 확인하자.
질문이나 선택지 설명을 바꿀 때 동일한 test 사례로 회귀 검증한다.'''),
code('''
def route(answer, threshold=.85):
    selected = answer['choice']
    return selected if answer['probabilities'][selected] >= threshold else 'human_review'

if LIVE_API:
    print('route:', route(response['answers']['team']))
    extra, _ = request_decision(BASE_URL, MODEL, '환불은 원하지 않고 오류만 고쳐 주세요.', questions, API_KEY)
    print('연습 관찰:', extra['answers'])
else:
    print('합성 fixture 정책 결과:', route(fixture['answers']['team']))
    assert route(fixture['answers']['team']) == 'human_review'
'''),
md('''## 해석과 문제 해결
- connection refused: Studio 실행, 포트, SSH 터널을 확인한다.
- 401/403: Access token 또는 localhost keyless 설정을 확인한다.
- timeout: 최초 모델 로딩/GPU 점유를 확인하고 timeout을 늘린다.
- 선택지 누락/NaN/합계 오류: 응답 계약 실패다. 임의 보정으로 숨기지 않는다.
- 한국어 정확도 저하: multilingual 모델 선택과 별도 한국어 평가 데이터를 확인한다.
- 02로 학습한 모델을 테스트하려면 merged 경로 및 Studio 선택 모델을 확인한다.

확장 과제: 실제 업무의 독립 test 파일로 cases를 교체하고 팀별 confusion matrix,
보류 비율, 유형별 정확도 및 latency 분포를 보고하자.

출처: [Unsloth 로컬 Decision API](https://unsloth.ai/docs/models/decision-laya),
[TypeSafe Jev 소개](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
''')])
print('Built 3 standalone notebooks')
