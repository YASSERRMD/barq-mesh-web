#!/bin/bash
set -e

echo "Building barq-mesh-web for WebAssembly..."
cargo build --target wasm32-unknown-unknown --release --features wasm
echo "Build complete."
