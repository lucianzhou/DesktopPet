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
