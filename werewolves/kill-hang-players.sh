#!/usr/bin/env bash

set -euo pipefail

mapfile -t pids < <(
    ps -eo user=,pid=,args= \
        | awk '$1 ~ /^player/ && $3 ~ /^python3$/ && $4 == "client.py" { print $2 }'
    )

echo "Killing PIDs: ${pids[*]}"

sudo kill "${pids[@]}"
