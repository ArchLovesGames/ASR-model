# ASR Fine-Tuning Activity Report

## 1. Objective

Automatic Speech Recognition (ASR) converts speech audio into text. The objective was to follow Mozilla AI's Speech-to-Text fine-tuning blueprint end-to-end: load a small real speech dataset, run baseline Whisper transcription, fine-tune a pretrained Whisper checkpoint, evaluate the result, compare predictions, and validate the Gradio interface.

## 2. Mozilla Blueprint

The activity followed the Mozilla AI Blueprint "Finetune an ASR model using Common Voice data" and the official repository at `mozilla-ai/speech-to-text-finetune`. The repository supports local Common Voice exports, Hugging Face-hosted Common Voice, and local custom datasets. Because Common Voice Hugging Face access requires account authorization and the activity environment had no token, I used the repository's bundled real custom WAV dataset, which exercises the same preprocessing, Whisper training, evaluation, and Gradio paths.

Sources inspected:

- https://blueprints.mozilla.ai/all-blueprints/finetune-an-asr-model-using-common-voice-data
- https://mozilla-ai.github.io/speech-to-text-finetune/
- https://github.com/mozilla-ai/speech-to-text-finetune

## 3. Technologies Used

- Whisper: pretrained encoder-decoder ASR model; `openai/whisper-tiny` was used for a CPU-compatible short run.
- Dataset: the repository's `example_data/custom` real WAV/transcript dataset, formatted as `train/text.csv`, `test/text.csv`, and `clips/rec_<index>.wav`.
- Hugging Face Transformers: loaded Whisper, performed generation, and ran `Seq2SeqTrainer`.
- PyTorch: executed model inference and fine-tuning on CPU.
- Gradio: repository UI for uploading or recording audio and transcribing with a Hugging Face or local model.
- WER: Word Error Rate, lower is better; computed with `evaluate`/`jiwer`.

## 4. Environment

- OS: macOS Darwin 25.4.0 arm64.
- Python: 3.13.4 in `.venv313`.
- ffmpeg: installed at `/opt/homebrew/bin/ffmpeg`.
- Device: CPU. CUDA was false and this Torch build reported MPS false.
- Model checkpoint: `openai/whisper-tiny`.
- Dataset: `example_data/custom`, English.
- Samples available: 5 train, 5 test.
- Samples used: 3 train, 2 held-out test.
- Available duration: 30.97 seconds train, 33.29 seconds test.

## 5. Dataset Preparation

The custom dataset loader reads CSV transcripts and matches them with sorted WAV files under each split's `clips` directory. The preprocessing step casts audio with Hugging Face `Audio` to Whisper's sample rate, extracts log-Mel input features through `WhisperProcessor`, tokenizes transcripts into labels, filters audio longer than 30 seconds, filters labels longer than 448 tokens, and saves the processed dataset at `example_data/custom/processed_version`.

Dataset summary was saved to `results/dataset_summary.json`.

## 6. Baseline Testing

Baseline inference used `openai/whisper-tiny` on two held-out WAV files and saved results to `results/baseline_results.json`.

| Sample | Reference excerpt | Baseline prediction excerpt |
| --- | --- | --- |
| 0 | D'AVRIGNY UNABLE TO BEAR THE SIGHT... | Davini, unable to bear the sight... |
| 1 | FOR SOME TIME NOTHING WAS HEARD... | For some time, nothing was heard... |

Baseline sample WER: `7.9365%`.

## 7. Fine-Tuning Configuration

Config file: `activity_config.yaml`.

- Base model: `openai/whisper-tiny`
- Output: `artifacts/activity-whisper-tiny-custom`
- Train samples: 3
- Test samples: 2
- Max steps: 3
- Batch size: 1
- Learning rate: `1e-5`
- Warmup steps: 0
- FP16: false
- Gradient checkpointing: false
- Save steps: 3
- Eval strategy: steps
- Generation max length: 96
- Push to Hub: false

Training ran successfully on CPU. Reported training runtime was `5.937` seconds inside the Trainer, and the wrapper runtime in `results/evaluation.json` was `13.777` seconds.

## 8. Results

Evaluation was saved to `results/evaluation.json`; sample comparison was saved to `results/comparison.json`.

| Metric | Baseline | Fine-tuned |
| --- | ---: | ---: |
| Eval loss | 3.0168 | 2.6568 |
| Normalized WER | 7.9365 | 7.9365 |
| Normalized CER | 2.1333 | 2.1333 |
| Orthographic CER | 84.8000 | 84.5333 |

| Sample | Baseline prediction | Fine-tuned prediction |
| --- | --- | --- |
| 0 | Davini, unable to bear the sight... | Davini, unable to bear the sight... |
| 1 | For some time, nothing was heard... | For some time nothing was heard... |

The short run reduced evaluation loss and slightly changed punctuation on one output, but did not improve normalized WER on the two held-out samples. This is expected for a 3-step CPU demonstration on only 3 training examples.

## 9. Challenges and Fixes

- Python 3.14 was the default interpreter and was too new to use confidently. Python 3.13.4 was used because a matching cached Torch wheel was available and the project declares Python 3.10+.
- Network speed was very slow for PyTorch and model dependencies. I used a cached Torch wheel for Python 3.13 and installed the remaining dependencies in smaller groups.
- Training initially attempted Hugging Face HEAD requests even after baseline cached the model. Rerunning with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` made the training use the local cache.
- The `cer` metric was not cached and could not be fetched offline. I added a fallback to compute WER/CER with local `jiwer`.
- The original training script did not explicitly save the final model at the end of training. I added `trainer.save_model`, processor saving, and metric saving.
- Gradio startup failed in this managed environment with `OSError: Cannot find empty port in range 7860-7959`, and also for explicit `8060-8159`. The app was still validated by direct import and function-level transcription.
- The Gradio local model loader needed an update for Transformers 4.46.3: it now passes `tokenizer` and `feature_extractor` explicitly.

## 10. Conclusion

The full ASR fine-tuning pipeline was completed on a real local speech dataset: setup, dataset preprocessing, baseline inference, actual fine-tuning, evaluation, saved model output, prediction comparison, and Gradio function validation. The fine-tuned model did not improve WER in this tiny run, but it did reduce eval loss from `3.0168` to `2.6568`, showing that training occurred and changed the model.

## 11. Reproduction Steps

From the repository root:

```bash
/opt/homebrew/bin/python3.13 -m venv .venv313
.venv313/bin/pip install /tmp/torch-2.12.0-cp313-cp313-macosx_14_0_arm64.whl
.venv313/bin/pip install 'transformers==4.46.3' 'datasets[audio]==3.1.0' 'accelerate==1.1.1' 'evaluate==0.4.3' 'gradio==5.9.1' 'jiwer==3.0.5' 'loguru==0.7.3' 'tensorboard==2.18.0' 'soundfile' 'spaces'
.venv313/bin/pip install -e . --no-deps
.venv313/bin/python scripts/activity_dataset_summary.py
.venv313/bin/python scripts/activity_transcribe.py --model openai/whisper-tiny --output results/baseline_results.json --limit 2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv313/bin/python scripts/run_activity_training.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv313/bin/python scripts/activity_transcribe.py --model artifacts/activity-whisper-tiny-custom --output results/finetuned_results.json --limit 2
.venv313/bin/python scripts/activity_compare_results.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv313/bin/python -c "import sys; sys.path.insert(0, 'demo'); import transcribe_app; print(transcribe_app.transcribe('', '', 'artifacts/activity-whisper-tiny-custom', 'example_data/custom/test/clips/rec_1.wav', False))"
```

To try the Gradio app on a normal local machine where ports are available:

```bash
GRADIO_SERVER_PORT=8060 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv313/bin/python demo/transcribe_app.py
```
