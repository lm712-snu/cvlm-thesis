# Work Log — Mathematics of Computer Vision Language Modules

## Task ID: tools-and-demo
**Agent:** z-ai-code (single-agent implementation)
**Scope:** Build modules 18, 19, 20, 21, 22 of the CVLM educational app.

### Deliverables
- `src/components/modules/Module18_Pipeline.tsx` — End-to-End Pipeline (14-stage vertical flowchart with "Run Pipeline" animation, expandable stage detail panels, synthetic 8×8×3 → 4 patches × 48 → embed d=4 → vocab V=8 → generated text; 3 viva questions).
- `src/components/modules/Module19_Sandbox.tsx` — Mathematical Sandbox (Textarea inputs for A/B matrices, Select for operation, supports matmul, transpose, add, ReLU, softmax rows, cosine sim, norm + vector variants; MatrixView + DimBadge for output; 5 example presets; clear dimension-mismatch error messages; 2 viva questions).
- `src/components/modules/Module20_DimensionTracker.tsx` — Dimension Tracker (sequential pipeline of ops: Linear, Conv2d, MultiheadAttention, Softmax, LayerNorm, ReLU, Reshape, Flatten; per-op input/output DimBadges, formula display, mismatch detection with red highlighting; Add Operation / Reset buttons; starting-shape presets; 2 viva questions).
- `src/components/modules/Module21_ThesisDemo.tsx` — Thesis Demo (uploads image via sharedImage store; 12-stage horizontal stepper with "Show Mathematics" expandable for each stage; real math computed from the uploaded image using randMatrix/seeded weights; final synthesized "A [object] [action] [location]" caption with honest disclaimer that it's a heuristic, not a real VLM; 3 viva questions).
- `src/components/modules/Module22_VivaMode.tsx` — Viva Mode (43-question bank covering all 22 modules; search Input + difficulty Tabs (All/Easy/Medium/Difficult); Accordion grouped by module using `MODULES` list; VivaCard for each question).

### Cross-cutting fixes
- `src/components/cvlm/MathBlock.tsx` — refactored the try/catch JSX pattern into a proper class-based `ErrorBoundary` (React Compiler rule `react-hooks/error-boundaries`).
- `src/components/modules/Module2_Convolution.tsx` — converted the auto-advance `useMemo` (which called `setState` synchronously) into a `useEffect` (rule `react-hooks/set-state-in-render`).
- `src/lib/math/attention.ts` — split the comma-operator one-liner in `argmax` into a normal block (rule `@typescript-eslint/no-unused-expressions`).

### Lint result
`bun run lint` → **0 errors, 0 warnings.**

### Design notes
- All math is computed client-side via the existing `@/lib/math` utilities (matmul, softmax, attention, sinusoidalPE, crossAttention, argmax, cosineSimilarity, norm, etc.). No external math libraries added.
- Every module follows the established style: `<ModuleHeader number=N title=... subtitle=...>`, `<MathBlock>` for LaTeX, `<ExpandableSection variant="math|why|how|intuition">`, `<VivaPanel>` (or `<VivaCard>`) for viva questions, `<DimBadge>` for tensor dimensions, shadcn/ui primitives (Card, Button, Select, Tabs, Accordion, Input, Textarea, Badge).
- Synthetic dimensions are kept tiny throughout (8×8×3 images, embed d=4–8, vocab V=8–10) so the matrices fit comfortably on screen and remain inspectable.
- TypeScript strict throughout, "use client" on every module, no emojis.
- Files are sized within budget (each under ~800 lines; thesis demo and viva mode slightly larger as allowed by the spec).

### What the user sees
The user can preview the app via the right-hand Preview Panel (or open in a new tab). Only `/` is exposed, but the modules compile cleanly and are wired into the `MODULES` catalog (`src/data/modules.ts`) ready to be selected by the sidebar.

---
Task ID: main-integration
Agent: main (orchestrator)
Task: Wire all modules together, configure static export for GitHub Pages, verify end-to-end with Agent Browser, write README + Mathematical Foundations.

Work Log:
- Created /src/app/page.tsx module router mapping all 22 module IDs to their components
- Added "Home" entry to sidebar; default activeModule = "home" so dashboard is the landing
- Updated layout.tsx with proper title + metadata for the thesis tool
- Created next.config.gh-pages.ts reference config for static export (output: "export", images.unoptimized, basePath/assetPrefix for project pages)
- Added scripts/build-gh-pages.sh: one-command static export with REPO_NAME arg
- Added build:static npm script (just runs next build with the patched config)
- Wrote README.md with full Mathematical Foundations documentation (linear algebra, image repr, convolution, activation functions, probability, optimization, attention, embeddings, VLM architecture, autoregressive generation, educational simplifications, references)
- Verified with Agent Browser:
  * Home dashboard renders all 22 module cards + 6 highlight cards
  * Module 1: image upload + sample image generation + 4-tab interface (Image/Pixel grid/Channels/Tensor) all functional
  * Module 8 (Self-Attention): all Q/K/V/scores/weights/output matrices render correctly; attention weights sum to 1 per row; viva questions present
  * Module 19 (Sandbox): matmul of [[1,2,3],[4,5,6]] × [[7,8],[9,10],[11,12]] = [[58,64],[139,154]] computed correctly
  * Module 22 (Viva Mode): 41 questions shown (18 Easy + 17 Medium + 6 Difficult); difficulty Tabs work
  * Module 21 (Thesis Demo): 12-stage stepper with all stage buttons visible; uploaded image persists from Module 1 via Zustand
  * Theme toggle: light/dark switch with no errors
  * ALL 22 modules clicked through → 0 runtime errors in browser console
- Lint: `bun run lint` → 0 errors, 0 warnings

Stage Summary:
- Complete working educational web app with 22 modules + dashboard
- All math implemented explicitly in /src/lib/math/{matrix,convolution,attention}.ts (no library black boxes in educational path)
- Ready for GitHub Pages deployment via `./scripts/build-gh-pages.sh <repo-name>`
- Screenshots saved to /home/z/my-project/download/thesis-tool-{home,attention}.png
