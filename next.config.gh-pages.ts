/**
 * Static-export configuration for GitHub Pages deployment.
 *
 * The dev environment uses `next.config.ts` with `output: "standalone"`.
 * To deploy on github.io, copy this file over `next.config.ts` (or rename it),
 * then run `npm run build:static`. The `out/` directory can be pushed to the
 * `gh-pages` branch (or to a `docs/` folder) and served by GitHub Pages.
 *
 * IMPORTANT: GitHub Pages serves project pages under `/<repo-name>/`, so set
 * `basePath` and `assetPrefix` to your repo name. For user/org pages
 * (`<username>.github.io`), leave them as `""`.
 */
import type { NextConfig } from "next";

// Change this to your GitHub repo name (e.g. "cvlm-math-thesis") if deploying
// to project pages. Leave as "" for user pages (username.github.io).
const REPO_NAME = "";

const nextConfig: NextConfig = {
  output: "export",
  images: {
    // GitHub Pages cannot run the Next.js image optimization server.
    unoptimized: true,
  },
  // basePath + assetPrefix must match your repo name when deploying to
  // https://<user>.github.io/<repo-name>/. For https://<user>.github.io/ leave empty.
  basePath: REPO_NAME ? `/${REPO_NAME}` : "",
  assetPrefix: REPO_NAME ? `/${REPO_NAME}/` : "",
  reactStrictMode: false,
  typescript: {
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
