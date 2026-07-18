#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
APP_NAME="DesktopPet"
APP_DIR="$ROOT/dist/$APP_NAME.app"
CONTENTS="$APP_DIR/Contents"

cd "$ROOT"
swift build -c release --product DesktopPet

rm -rf "$APP_DIR"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources/Assets"

cp "$ROOT/.build/release/DesktopPet" "$CONTENTS/MacOS/DesktopPet"
cp "$ROOT/Support/Info.plist" "$CONTENTS/Info.plist"
cp "$ROOT/Assets/baomihua-interaction-v5.png" "$CONTENTS/Resources/Assets/"
cp "$ROOT/Assets/baomihua-sleep-v3.png" "$CONTENTS/Resources/Assets/"
cp "$ROOT/Assets/baomihua-wake-v7.png" "$CONTENTS/Resources/Assets/"
cp "$ROOT/Assets/baomihua-neutral.png" "$CONTENTS/Resources/Assets/"
cp "$ROOT/Assets/baomihua-gaze-v8-uniform.png" "$CONTENTS/Resources/Assets/"

codesign --force --deep --sign - "$APP_DIR"
echo "$APP_DIR"
