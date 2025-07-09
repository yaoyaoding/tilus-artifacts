#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh # This sets up the 'conda' command for this new shell process
conda activate titus-artifacts           # This activates the environment for this new shell process

exec "$@"
