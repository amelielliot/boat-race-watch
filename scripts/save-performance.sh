#!/usr/bin/env bash
set -euo pipefail

git add data/ledger.json PERFORMANCE.md data/validation LIVE_COMPARISON.md
if git diff --cached --quiet; then
  echo "Performance is unchanged; nothing to save."
  exit 0
fi
git config user.name "boat-race-watch[bot]"
git config user.email "boat-race-watch[bot]@users.noreply.github.com"
git commit -m "Update paper-trading performance and comparisons"

# Retry the same commit; never rerun predictions or force-push saved records.
for attempt in 1 2 3 4; do
  if output=$(git push origin HEAD:main 2>&1); then
    printf '%s\n' "$output"
    echo "Performance saved (attempt $attempt)."
    exit 0
  fi
  printf '%s\n' "$output"
  # Authentication, permissions and non-fast-forward errors need intervention.
  if ! printf '%s\n' "$output" | grep -Eqi 'Internal Server Error|HTTP (code = |error: )?5[0-9][0-9]|requested URL returned error: 5[0-9][0-9]|Bad Gateway|Service Unavailable|Gateway Timeout|timed out|Connection reset|Could not resolve host|remote end hung up'; then
    echo "::error::Non-transient save failure; records will be backed up as an artifact."
    exit 1
  fi
  if [ "$attempt" -lt 4 ]; then
    echo "::warning::Temporary GitHub save error; retrying ($attempt/4)."
    sleep "${SAVE_RETRY_DELAY_SECONDS:-3}"
  fi
done
echo "::error::GitHub save failed after four attempts; records will be backed up as an artifact."
exit 1
