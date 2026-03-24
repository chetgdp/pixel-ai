#!/bin/bash
set -e

# compile to wasm
wasm-pack build --target no-modules --out-dir pkg_nomodules/
# weave with histos
histos config.yaml -o pixel.ai.html


