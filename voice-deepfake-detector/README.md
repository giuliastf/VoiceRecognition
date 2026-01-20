# Voice Deepfake Detector

Minimal project to classify real vs fake speech using log-mel spectrograms and a compact CNN.

## Setup

```bash
pip install -r requirements.txt
```

## Create split manifests

```bash
export PYTHONPATH=src
python -m vdd.hf.build_id_splits --config configs/base.yaml
```

For a streaming-only quick split (no full download), set in `configs/base.yaml`:

- `splits.use_streaming: true`
- `splits.max_audio_ids: 2000` (or whatever smaller number is needed)

## Train baseline

```bash
export PYTHONPATH=src
python -m vdd.train --config configs/experiments/baseline.yaml
```

Embedding baseline (wav2vec2 + logistic regression):

```bash
export PYTHONPATH=src
python -m vdd.embeddings.train_embed --config configs/experiments/embeddings_baseline.yaml
```

Extended baseline run:

```bash
export PYTHONPATH=src
python -m vdd.train --config configs/experiments/overnight_baseline.yaml
```

Extended baseline v2 (no augmentation, lower LR):

```bash
export PYTHONPATH=src
python -m vdd.train --config configs/experiments/overnight_baseline_v2.yaml
```

## Train cross-generator holdout

```bash
export PYTHONPATH=src
python -m vdd.train --config configs/experiments/crossgen.yaml
```

Embedding crossgen:

```bash
export PYTHONPATH=src
python -m vdd.embeddings.train_embed --config configs/experiments/embeddings_crossgen.yaml
```

Extended crossgen run:

```bash
export PYTHONPATH=src
python -m vdd.train --config configs/experiments/overnight_crossgen.yaml
```

Extended crossgen v2 (lower LR):

```bash
export PYTHONPATH=src
python -m vdd.train --config configs/experiments/overnight_crossgen_v2.yaml
```

## Evaluate

```bash
export PYTHONPATH=src
python -m vdd.eval --config configs/experiments/baseline.yaml
python -m vdd.eval --config configs/experiments/crossgen.yaml
```

Pretrained HF deepfake model evaluation:

```bash
export PYTHONPATH=src
python -m vdd.hf_models.eval_hf_model --config configs/experiments/hf_garystafford_baseline.yaml
python -m vdd.hf_models.eval_hf_model --config configs/experiments/hf_garystafford_crossgen.yaml
```

Embedding evaluation:

```bash
export PYTHONPATH=src
python -m vdd.embeddings.eval_embed --config configs/experiments/embeddings_baseline.yaml
python -m vdd.embeddings.eval_embed --config configs/experiments/embeddings_crossgen.yaml
```

## Real-time demo

```bash
export PYTHONPATH=src
python -m vdd.realtime.infer_live --checkpoint runs/exp005_overnight_baseline_v2/best_model.pt
```

Optional flags:
- `--window 4.0`
- `--hop 0.5`
- `--threshold 0.6`
- `--playback` (press `p` to play last audio)
- `--silence-rms 0.01` (treat low RMS as silence)
- `--bias -0.1` (shift probabilities to reduce false "fake")

## Outputs

- Run artifacts: `runs/expXXX_*`
- Cached mels: `data/cache/mels/`
- Split manifests: `data/manifests/`
- Training curves: `runs/expXXX_*/training_curves.png`
- ROC curve: `runs/expXXX_*/roc_curve.png`
- PR curve: `runs/expXXX_*/pr_curve.png`
