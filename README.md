# SegreSmart

Offline wet/dry waste sorting prototype for Raspberry Pi 4 (2 GB), USB webcam, MobileNetV3-Small TensorFlow Lite inference, and a servo diverter.

## Architecture

```text
USB webcam -> ROI -> RGB/224px preprocessing -> MobileNetV3-Small TFLite
                                      -> temporal confidence filter
                                      -> wet/dry/reject/unknown state machine
                                      -> servo diverter -> WET or DRY bin
```

V1 assumes one object in the sorting area. Reject, unknown, low-confidence, and empty predictions never actuate the servo.

## Hardware and safety

Use a Raspberry Pi 4, USB webcam, servo, external regulated 5 V servo supply, and common ground. BCM GPIO18 is the servo signal (physical header pin 12); “GPIO18” is not physical pin 18.

| Servo wire | Connect to |
|---|---|
| Signal | Pi BCM GPIO18 |
| V+ | External regulated 5 V supply |
| GND | External supply GND, shared with Pi GND |

Never power a servo from a GPIO pin or assume the Pi 3.3 V rail can supply it. Poor servo power can cause voltage dips, resets, camera disconnects, jitter, or SD corruption.

## PC setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/Pi: source .venv/bin/activate
pip install -r requirements-training.txt
python scripts/inspect_dataset.py
```

Place manually downloaded datasets under `datasets/raw/`. API credentials are optional and are not used by the runtime. Run `python scripts/inspect_dataset.py` before conversion; it writes `outputs/dataset_inspection.json`.

## Dataset mapping and preparation

Organic/biodegradable is treated as `wet` because the project means kitchen/biodegradable waste, not surface liquid. Paper, cardboard, plastic, glass, metal, cloth, recyclable, and non-recyclable map to `dry`. Hazardous and e-waste map to `reject`; they are never silently relabeled as ordinary dry waste.

```bash
python scripts/prepare_dataset.py
```

The converter handles class folders and YOLO text annotations, crops boxes with configurable padding, removes tiny/corrupt images, converts EXIF images to RGB, and writes `outputs/dataset_manifest.csv` and `outputs/dataset_report.json`. Keep all crops from one original image together when extending the split logic; the manifest records `group_id` for this purpose.

Capture 300–500 empty-chamber images in the real bin, under different lighting:

```bash
python scripts/capture_empty_background.py --count 400
```

Copy these into `datasets/processed/empty/` before training if using the empty class.

## Training and export

```bash
python scripts/train.py --epochs 20
python scripts/evaluate.py
python scripts/export_tflite.py
```

Training uses MobileNetV3-Small transfer learning at configurable 224px input. The current starter training script freezes the backbone; fine-tuning can be enabled after the first run by unfreezing final layers and recompiling at a lower learning rate. Do not claim success from accuracy alone: inspect `outputs/evaluation.json`, then add confusion-matrix/history plotting for your experiment run.

`models/labels.txt` must match the model output order. TFLite inference uses RGB conversion and handles float/uint8 inputs and quantized outputs. INT8 export should only be added with a representative calibration set from the real camera domain.

## Pi tests and runtime

Install only deployment dependencies on the Pi:

```bash
pip install -r requirements-pi.txt
python tools/camera_test.py
python tools/servo_test.py
python tools/inference_test.py --image example.jpg
python src/pi/main.py --headless
```

For camera troubleshooting: `ls /dev/video*` and `v4l2-ctl --list-devices`. Set `system.simulation_mode: true` while developing on Windows. Calibrate `servo.wet_angle`, `servo.dry_angle`, and `servo.neutral_angle` in `config.yaml`; they are mechanical starting points only.

The live preview overlays prediction, confidence, and latency. Set `system.show_preview: false` or use `--headless` for deployment. Logs are `logs/system.log` and `logs/predictions.csv`.

## Optional boot service

Edit the paths and username in `deploy/segresmart.service`, copy it to `/etc/systemd/system/`, then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now segresmart
journalctl -u segresmart -f
```

## Limitations and future work

This prototype handles one object, requires a trained local model, and needs Pi hardware validation for camera timing, GPIO permissions, servo angles, power integrity, and model latency. Future work includes grouped deterministic train/validation/test splitting in the training loader, calibrated INT8 benchmarking, object-presence detection, and mechanical sensors.
