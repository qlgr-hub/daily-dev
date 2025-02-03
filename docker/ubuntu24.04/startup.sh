#!/usr/bin/env bash

set -e

cd /workspace || exit
source ~/.venv/bin/activate
jupyter lab --allow-root --no-browser --ip=0.0.0.0 &

sudo /usr/sbin/sshd -D