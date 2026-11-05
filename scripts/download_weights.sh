#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
mkdir -p weights/bert-base-uncased

download() {
    local url="$1" destination="$2"
    if [[ ! -s "$destination" ]]; then
        curl --fail --location --retry 3 "$url" --output "$destination.part"
        mv -- "$destination.part" "$destination"
    fi
}

download "$HF_ENDPOINT/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth" \
    weights/groundingdino_swint_ogc.pth
printf '%s\n' '3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799  weights/groundingdino_swint_ogc.pth' | sha256sum -c -
for name in config.json vocab.txt tokenizer_config.json tokenizer.json model.safetensors; do
    download "$HF_ENDPOINT/google-bert/bert-base-uncased/resolve/main/$name" \
        "weights/bert-base-uncased/$name"
done
