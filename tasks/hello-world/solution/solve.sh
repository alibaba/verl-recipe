#!/bin/sh
set -eu

task_root="${TASK_ROOT:-/app}"
mkdir -p "$task_root"
printf 'Hello, Harbor!\n' > "$task_root/answer.txt"
