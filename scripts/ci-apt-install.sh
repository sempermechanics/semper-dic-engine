#!/usr/bin/env bash
# Install apt packages on a CI runner: scripts/ci-apt-install.sh pkg...
#
# The runner's Ubuntu mirror now and then stalls for many minutes, and with
# -qq a stalled `apt-get update` prints nothing until it gives up; one ate 17
# of a 30-minute job. So every network step here is bounded and retried, and
# the .debs are kept in ~/apt-cache, which actions/cache restores and saves.
# Downloading is a separate, safely killable step: the install itself then
# runs from the cache, with nothing left to fetch.
set -euo pipefail
CACHE="$HOME/apt-cache"
mkdir -p "$CACHE/partial"
NET=(-o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)
APT=(-y --no-install-recommends -o Dir::Cache::archives="$CACHE")

retry() {   # retry <seconds per attempt> <label> cmd...
  local limit=$1 label=$2; shift 2
  for attempt in 1 2 3; do
    if timeout "$limit" "$@"; then return 0; fi
    echo "::warning::$label: attempt $attempt failed or took over ${limit}s"
  done
  echo "::error::$label failed 3 times"
  return 1
}

retry 120 "apt-get update" sudo apt-get update -q "${NET[@]}"
retry 240 "apt download" sudo apt-get install -qq --download-only "${APT[@]}" "${NET[@]}" "$@"
sudo apt-get install -qq "${APT[@]}" "$@"
sudo chown -R "$(id -u):$(id -g)" "$CACHE"
