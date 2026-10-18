#!/usr/bin/env sh
# Downloads the SQuAD 2.0 development set (CC BY-SA 4.0) into data/. It isn't committed.
set -e
cd "$(dirname "$0")/.."
mkdir -p data
curl -fsSL -o data/dev-v2.0.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
echo "Saved data/dev-v2.0.json"
