#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
"$ROOT/Scripts/build-app.sh"
open "$ROOT/dist/DesktopPet.app"
