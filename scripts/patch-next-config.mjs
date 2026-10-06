/**
 * Patches next.config.ts for static export to GitHub Pages.
 *
 * Usage:
 *   node scripts/patch-next-config.mjs <repo-name>
 *
 * If <repo-name> is empty or omitted, basePath / assetPrefix are left empty
 * (suitable for user/org pages at https://<user>.github.io/).
 *
 * Used by the GitHub Actions workflow in .github/workflows/deploy.yml.
 * You can also run it locally if you want to test the static build before pushing.
 */
import fs from "node:fs";
import path from "node:path";

const repo = process.argv[2] ?? "";
const projectDir = path.resolve(import.meta.dirname, "..");
const configPath = path.join(projectDir, "next.config.ts");

const content = `import type { NextConfig } from "next";
const REPO_NAME = "${repo}";
const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  basePath: REPO_NAME ? \`/\${REPO_NAME}\` : "",
  assetPrefix: REPO_NAME ? \`/\${REPO_NAME}/\` : "",
  reactStrictMode: false,
  typescript: { ignoreBuildErrors: true },
};
export default nextConfig;
`;

fs.writeFileSync(configPath, content);
console.log(`Wrote next.config.ts for repo: "${repo || "<root>"}"`);
