#!/usr/bin/env bash
# Build one ML4PS tex file to PDF via the TeX Live docker image.
# Usage: ./build.sh observability   (compiles observability.tex -> observability.pdf)
set -euo pipefail
cd "$(dirname "$0")"
STEM="${1:?usage: build.sh <stem-without-.tex>}"
IMAGE="texlive/texlive:latest"
docker run --rm -v "$PWD":/doc -w /doc "$IMAGE" bash -lc \
  "pdflatex -interaction=nonstopmode -halt-on-error ${STEM}.tex >/dev/null && \
   bibtex ${STEM} >/dev/null && \
   pdflatex -interaction=nonstopmode -halt-on-error ${STEM}.tex >/dev/null && \
   pdflatex -interaction=nonstopmode -halt-on-error ${STEM}.tex"
echo ">> wrote $PWD/${STEM}.pdf"
