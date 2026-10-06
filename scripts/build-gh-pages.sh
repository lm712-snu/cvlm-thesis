#!/usr/bin/env bash
# Build a static export of the app for GitHub Pages deployment.
#
# Usage:
#   ./scripts/build-gh-pages.sh [REPO_NAME]
#
# Examples:
#   ./scripts/build-gh-pages.sh                    # for user pages (username.github.io)
#   ./scripts/build-gh-pages.sh cvlm-thesis        # for project pages (username.github.io/cvlm-thesis)
#
# After running, the `out/` directory contains the static site.
# To deploy to GitHub Pages:
#   1. Push the contents of `out/` to a `gh-pages` branch, OR
#   2. Copy the contents of `out/` into a `docs/` folder and enable Pages in repo settings.

set -euo pipefail

REPO_NAME="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "[1/4] Patching next.config.ts for static export (repo: '${REPO_NAME:-<root>}')"
node scripts/patch-next-config.mjs "$REPO_NAME"

echo "[2/4] Building static export"
bun run build:static

echo "[3/4] Adding .nojekyll"
touch ./out/.nojekyll

echo "[4/4] Done."
echo ""
echo "Static site is in: $PROJECT_DIR/out"
echo ""
echo "Next steps:"
echo "  - Push the contents of ./out to a 'gh-pages' branch, OR"
echo "  - Copy ./out/* to ./docs/ and enable Pages in your repo settings, OR"
echo "  - Easiest: just push to GitHub and let .github/workflows/deploy.yml handle it."
