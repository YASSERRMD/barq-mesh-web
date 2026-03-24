#!/bin/bash
set -e

echo "Building barq-mesh-web for WebAssembly..."
wasm-pack build --target web --out-dir pkg -- --features wasm
echo "Build complete."
