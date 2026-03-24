#!/usr/bin/env bash
set -e

echo "🔨  Building barq-mesh-web WASM package..."
wasm-pack build \
    --target web \
    --out-dir pkg \
    --features wasm \
    -- -Z build-std=panic_abort,std

echo "✅  Done — output in ./pkg"
