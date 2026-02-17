#!/usr/bin/env bash
set -euo pipefail

URL="${1:-https://drive.google.com/drive/folders/1M90NDSPkg_0gvr7wwz0gYKTX_lo-DcSp}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

python3 - <<'PY' >/dev/null 2>&1 || python3 -m pip install -q gdown
import importlib.util; raise SystemExit(0 if importlib.util.find_spec("gdown") else 1)
PY

python3 - "$URL" "$TMP" <<'PY'
import sys, gdown
gdown.download_folder(url=sys.argv[1], output=sys.argv[2], quiet=False, use_cookies=False)
PY

for m in ares lmms sensembert; do
  src="$(find "$TMP" -type f \( -name "${m}_context_features.pt" -o -iname "*${m}*context*features*.pt" \) | head -n1 || true)"
  [[ -n "$src" ]] || { echo "missing: ${m}_context_features.pt"; exit 1; }
  cp -f "$src" "src/${m}_expert/${m}_context_features.pt"
  echo "ok: src/${m}_expert/${m}_context_features.pt"
done
