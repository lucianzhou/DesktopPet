#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"
swift run DesktopPetCoreChecks

PYTHON="${PYTHON:-/Users/popwind/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3}"
"$PYTHON" Scripts/validate-wake.py Assets/baomihua-wake.png \
  --json-out Art/QA/wake-row-v2-validation.json
"$PYTHON" Scripts/validate-gaze-atlas.py Assets/baomihua-gaze-v8-uniform.png \
  --allow-pose-silhouette \
  --canonical Art/Approved/baomihua-canonical-front-crouch-cell.png \
  --json-out Art/QA/gaze-v8-uniform/validation.json


# The shipped interaction is a quality-gated 8×6 / 48-frame sequence.  Its
# validator deterministically rebuilds the atlas from the one storyboard and
# rejects frame reordering, hidden patching, duplicate action frames, scale
# pops, and non-identical neutral anchors.
"$PYTHON" Scripts/validate-interaction-v5.py \
  --storyboard Art/Generated/baomihua-interaction-v5-storyboard.png \
  --neutral Assets/baomihua-neutral.png \
  --atlas Assets/baomihua-interaction-v5.png \
  --rows 6 \
  --columns 8 \
  --prepare-report Art/QA/interaction-v5/registration.json \
  --json-out Art/QA/interaction-v5/validation.json

# The approved interaction-v6 rise segment contains complete generated cats
# only. Registration applies one uniform transform to each whole cat, locks the
# head scale and paw baseline, and forbids local warps or cross-frame parts.
# The validator intentionally exits non-zero for the single reviewed tail-matte
# metric warning; the recorded resolution is checked immediately afterwards.
set +e
"$PYTHON" Scripts/validate-interaction-proportions.py \
  --atlas Art/Approved/interaction-v6/rise-v1/atlas.png \
  --canonical Art/Approved/interaction-v6/rise-v1/frames/00-F00.png \
  --half-rise-master Art/Approved/interaction-v6/rise-v1/frames/08-F08.png \
  --rows 1 \
  --columns 9 \
  --mode-map Art/QA/interaction-v6/rise-v1/mode-map.json \
  --json-out Art/QA/interaction-v6/rise-v1/proportions.json
rise_status=$?
set -e
[[ $rise_status -eq 1 ]]
"$PYTHON" -c '
import json
from pathlib import Path

validation = json.loads(Path("Art/QA/interaction-v6/rise-v1/proportions.json").read_text())
resolution = json.loads(Path("Art/QA/interaction-v6/rise-v1/metric-resolution.json").read_text())
assert validation["failures"] == resolution["failed_checks"]
assert resolution["decision"] == "accept" and resolution["severity"] == "minor"
'
