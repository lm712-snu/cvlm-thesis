#!/usr/bin/env bash
# Build a static export of the app for GitHub Pages deployment.
#
# Usage:
#   ./scripts/build-gh-pages.sh [REPO_NAME]
#
# Examples:
#   ./scripts/build-gh-pages.sh                    # for user pages (username.github.io)
#   ./scripts/build-gh-pages.sh cvlm-math-thesis   # for project pages (username.github.io/cvlm-math-thesis)
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

echo "[1/3] Configuring next.config.ts for static export (repo: '${REPO_NAME:-<root>}')"

# Backup the original config.
cp next.config.ts next.config.ts.bak

# Patch next.config.ts to use static export.
cat > next.config.ts <<EOF
import type { NextConfig } from "next";
const REPO_NAME = "${REPO_NAME}";
const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  basePath: REPO_NAME ? \`/\${REPO_NAME}\` : "",
  assetPrefix: REPO_NAME ? \`/\${REPO_NAME}/\` : "",
  reactStrictMode: false,
  typescript: { ignoreBuildErrors: true },
};
export default nextConfig;
EOF

echo "[2/3] Building static export"
bun run build:static || {
  echo "Build failed. Restoring original config."
  mv next.config.ts.bak next.config.ts
  exit 1
}

echo "[3/3] Restoring original dev config"
mv next.config.ts.bak next.config.ts

echo ""
echo "Done. Static site is in: $PROJECT_DIR/out"
echo ""
echo "Next steps:"
echo "  - Push the contents of ./out to a 'gh-pages' branch, OR"
echo "  - Copy ./out/* to ./docs/ and enable Pages in your repo settings."
