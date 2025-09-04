#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Ensure gh available
if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI 'gh' is required" >&2
  exit 1
fi

branch="results/$(date +%Y%m%d_%H%M)"
git checkout -b "$branch"

# Stage results and logs summaries if present
git add results/** logs/** || true
git commit -m "results: nightly run outputs" || true
git push -u origin "$branch" || true

# Open PR to main (idempotent-ish)
gh pr create --base main --head "$branch" \
  --title "Nightly results $(date +%Y-%m-%d)" \
  --body "Automated nightly run results and logs." || true

echo "Published branch: $branch"

