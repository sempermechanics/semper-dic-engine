#!/usr/bin/env bash
# Install apt packages on a CI runner: scripts/ci-apt-install.sh pkg...
#
# The runner's Ubuntu mirror now and then stalls for many minutes, and with
# -qq a stalled `apt-get update` prints nothing until it gives up; one ate 17
# of a 30-minute job. So every network step here is bounded and retried, and
# the .debs are kept in ~/apt-cache, which actions/cache restores and saves.
# Downloading is a separate, safely killable step: the install itself then
# runs from the cache, with nothing left to fetch.
#
# A mirror that is slow rather than down stays slow on a retry, and apt only
# moves down the runner's mirror list (/etc/apt/apt-mirrors.txt, the Azure
# mirror first) on an error. So after a failed first attempt every later one
# goes to archive.ubuntu.com.
set -euo pipefail
CACHE="$HOME/apt-cache"
mkdir -p "$CACHE/partial"
NET=(-o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)
APT=(-y --no-install-recommends -o Dir::Cache::archives="$CACHE")
SWITCHED=0

use_main_archive() {
  [ "$SWITCHED" = 1 ] && return 0
  SWITCHED=1
  echo "::warning::apt: moving from the runner's mirror to archive.ubuntu.com"
  if [ -f /etc/apt/apt-mirrors.txt ]; then
    echo "http://archive.ubuntu.com/ubuntu/" | sudo tee /etc/apt/apt-mirrors.txt >/dev/null
  fi
  sudo find /etc/apt -name '*.list' -o -name '*.sources' |
    xargs -r sudo sed -i 's#azure\.archive\.ubuntu\.com#archive.ubuntu.com#g'
  timeout 120 sudo apt-get update -q "${NET[@]}" || true
}

retry() {   # retry <seconds per attempt> <label> cmd...
  local limit=$1 label=$2; shift 2
  for attempt in 1 2 3; do
    if timeout "$limit" "$@"; then return 0; fi
    echo "::warning::$label: attempt $attempt failed or took over ${limit}s"
    use_main_archive
  done
  echo "::error::$label failed 3 times"
  return 1
}

retry 120 "apt-get update" sudo apt-get update -q "${NET[@]}"
# -q, not -qq: one Get: line per package, so a stall shows which file.
retry 240 "apt download" sudo apt-get install -q --download-only "${APT[@]}" "${NET[@]}" "$@"
sudo apt-get install -qq "${APT[@]}" "$@"
sudo chown -R "$(id -u):$(id -g)" "$CACHE"
