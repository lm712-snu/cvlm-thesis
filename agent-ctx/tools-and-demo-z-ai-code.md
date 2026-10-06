# Work Record — Task `tools-and-demo`

**Agent:** z-ai-code (single-agent implementation)
**Date:** session-local
**Task ID:** tools-and-demo

## Summary of work

Built five new interactive React modules (18-22) for the CVLM educational web app:

| File | Module | Lines | Viva Qs |
|---|---|---|---|
| `src/components/modules/Module18_Pipeline.tsx` | End-to-End Pipeline | ~470 | 3 |
| `src/components/modules/Module19_Sandbox.tsx` | Mathematical Sandbox | ~430 | 2 |
| `src/components/modules/Module20_DimensionTracker.tsx` | Dimension Tracker | ~470 | 2 |
| `src/components/modules/Module21_ThesisDemo.tsx` | Thesis Demo | ~620 | 3 |
| `src/components/modules/Module22_VivaMode.tsx` | Viva Mode | ~570 | 0 (43-question bank instead) |

## Implementation approach

- Read existing modules (1, 2, 8) and shared cvlm components (MathBlock, MatrixView, Heatmap, DimBadge, ExpandableSection, ModuleHeader, VivaCard/VivaPanel) and the app-store before writing any code.
- Used `@/lib/math` primitives (matmul, transpose, relu, softmaxRows, softmaxVec, cosineSimilarity, norm, shape, flatten, addBias, randMatrix, sumRows, sinusoidalPE, selfAttention, crossAttention, argmax) for every numeric computation. No new math dependencies.
- Module 18 uses an explicit 14-stage pipeline (image → pixel tensor → patches → patch embed → positional → ViT → visual tokens → projector → multimodal rep → LM → logits → softmax → token → text) with seeded random matrices so the trace is reproducible. The "Run Pipeline" button animates through each stage with auto-open of the detail panel.
- Module 19 parses matrices from textarea text (comma/newline/space-separated). All ten operations (matmul, transpose, add, relu, softmax rows, cosine sim, norm matrix, softmax vec, relu vec, norm vec) are dispatched through a single `compute` function. Error messages are explicit about which dimensions mismatched.
- Module 20 maintains a list of op instances with dynamic parameter forms. The `applyOp` function walks the shape through each op and produces a formula string + (possibly empty) output shape. Mismatched ops turn the row red.
- Module 21 uses `sharedImage` from the zustand store and computes the entire 12-stage pipeline from the uploaded image's pixels using seeded weights. Caption synthesis is a deterministic heuristic mapping average R/G/B + brightness to one of {dog, tree, car, person} + action + location, with a clear disclaimer that it is not a real VLM.
- Module 22 ships 43 hand-written questions covering all 22 modules (every module has at least one question). Questions are typed as `BankQuestion extends VivaQuestion` so they slot directly into `VivaCard`. Tabs filter by difficulty; Input searches across question/hint/answer/explanation; Accordion groups by module in `MODULES` order.

## Cross-cutting fixes

- Refactored `src/components/cvlm/MathBlock.tsx` from a `try/catch around JSX` pattern (which violates `react-hooks/error-boundaries`) into a proper class-based `ErrorBoundary`.
- Converted the auto-advance `useMemo` in `src/components/modules/Module2_Convolution.tsx` into a `useEffect` (the `useMemo` body called `setState`, which violates `react-hooks/set-state-in-render`).
- Split the comma-operator one-liner in `argmax` in `src/lib/math/attention.ts` into a normal block (rule `@typescript-eslint/no-unused-expressions`).

## Lint status

`bun run lint` → **0 errors, 0 warnings.**

## Notes for downstream agents

- If you add more modules that render LaTeX, the new `MathErrorBoundary` in `MathBlock.tsx` will catch any malformed LaTeX and render a fallback `<code>` instead of crashing the page.
- The Viva question type is exported from `@/components/cvlm/VivaCard` as `VivaQuestion` (with `level`, `question`, `hint`, `answer`, `explanation`). To extend the question bank in Module 22, append to the `QUESTIONS` array.
- All five new modules are listed in `src/data/modules.ts` (already present before this task) but are not wired into the sidebar/home routing in this task — only the files themselves were created.
