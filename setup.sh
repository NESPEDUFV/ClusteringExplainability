#!/usr/bin/env bash
# Adiciona a raiz do projeto ao PYTHONPATH.
# Uso: source setup.sh
export PYTHONPATH="$(pwd):${PYTHONPATH}"
