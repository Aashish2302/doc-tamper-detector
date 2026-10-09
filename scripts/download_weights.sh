#!/usr/bin/env bash
# Download the model + DINOv3 backbone weights from the GitHub Release into ./weights/
# Usage: scripts/download_weights.sh              (uses the repo's own release)
#        REPO=owner/name TAG=weights-v1 scripts/download_weights.sh
set -euo pipefail
REPO="${REPO:-Aashish2302/doc-tamper-detector}"
TAG="${TAG:-weights-v1}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/weights"
mkdir -p "$DIR"
echo "Downloading weights from $REPO release $TAG -> $DIR"
for f in model.pt dinov3_vitb16.pth; do
  if ! curl -fL --retry 3 "https://github.com/$REPO/releases/download/$TAG/$f" -o "$DIR/$f"; then
    echo "curl failed for $f; trying the gh CLI instead"
    gh release download "$TAG" --repo "$REPO" --pattern "$f" --dir "$DIR" --clobber
  fi
done
echo "Done. Files:"; ls -la "$DIR"
