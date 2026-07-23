#!/bin/bash
# Build the arXiv submission tarball. arXiv compiles main.tex itself and
# uses the shipped main.bbl instead of running bibtex, so the tarball
# needs exactly: main.tex, main.bbl, and the referenced figure PDFs.
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex > /dev/null
bibtex main > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
mkdir -p dist
tar czf dist/arxiv.tar.gz main.tex main.bbl figures/artifact_arc.pdf figures/sensitivity_forest.pdf
echo "wrote dist/arxiv.tar.gz:"
tar tzf dist/arxiv.tar.gz
