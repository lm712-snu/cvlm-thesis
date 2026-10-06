"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { RefreshCw, Dice5 } from "lucide-react";
import { MatrixView } from "@/components/cvlm/MatrixView";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import { matmul, addBias, randMatrix, shape, fmtVec } from "@/lib/math";

export function Module14_Projector() {
  const [dLM, setDLM] = useState(2);
  const [seed, setSeed] = useState(7);

  const N_TOKENS = 3; // number of visual tokens
  const DV = 4; // d_v: vision feature dim

  // Deterministic synthetic data: Z_visual (3x4), W_P (4 x d_LM), b (d_LM)
  const { Z_visual, W_P, b, Z_projected } = useMemo(() => {
    const Z = randMatrix(N_TOKENS, DV, -1, 1, seed);
    const W = randMatrix(DV, dLM, -1, 1, seed + 1);
    const bias = randMatrix(1, dLM, -0.5, 0.5, seed + 2)[0];
    const projected = addBias(matmul(Z, W), bias);
    return { Z_visual: Z, W_P: W, b: bias, Z_projected: projected };
  }, [dLM, seed]);

  const [zRows, zCols] = shape(Z_visual);
  const [wRows, wCols] = shape(W_P);
  const [pRows, pCols] = shape(Z_projected);

  // Click-to-explain a single output cell
  const [cell, setCell] = useState<[number, number] | null>(null);
  const explain = (i: number, j: number) => {
    const zRow = Z_visual[i];
    const wCol = W_P.map((r) => r[j]);
    const dot = zRow.reduce((s, z, k) => s + z * wCol[k], 0);
    return { zRow, wCol, dot, bias: b[j], out: Z_projected[i][j] };
  };

  return (
    <div>
      <ModuleHeader
        number={14}
        title="Vision-to-Language Projector"
        subtitle="A linear layer that lifts visual features from vision space (d_v) into the LM's embedding space (d_LM). The smallest bridge in a VLM."
      >
        <MathBlock block>
          {`Z_{\\text{projected}} = Z_{\\text{visual}} \\, W_P + b, \\quad W_P \\in \\mathbb{R}^{d_v \\times d_{LM}},\\ b \\in \\mathbb{R}^{d_{LM}}`}
        </MathBlock>
      </ModuleHeader>

      <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
        {/* Controls */}
        <Card className="h-fit">
          <CardHeader>
            <CardTitle className="text-base">Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            <div>
              <div className="flex justify-between">
                <Label className="text-xs">LM embedding dim d_LM</Label>
                <span className="font-mono text-xs">{dLM}</span>
              </div>
              <Slider
                value={[dLM]}
                min={2}
                max={8}
                step={1}
                onValueChange={([v]) => setDLM(v ?? 2)}
              />
              <p className="mt-1 text-[10px] text-muted-foreground">
                Vocabulary of the language model. Higher = richer LM space.
              </p>
            </div>

            <div className="rounded border bg-muted/30 p-2 text-xs space-y-1">
              <div className="flex justify-between">
                <span>Vision dim d_v</span>
                <DimBadge dims={String(DV)} variant="input" />
              </div>
              <div className="flex justify-between">
                <span>Num visual tokens N</span>
                <DimBadge dims={String(N_TOKENS)} variant="intermediate" />
              </div>
              <div className="flex justify-between">
                <span>W_P shape</span>
                <DimBadge dims={`${DV} × ${dLM}`} variant="weight" />
              </div>
              <div className="flex justify-between">
                <span>b shape</span>
                <DimBadge dims={String(dLM)} variant="weight" />
              </div>
              <div className="flex justify-between">
                <span>Z_visual</span>
                <DimBadge dims={`${N_TOKENS} × ${DV}`} variant="input" />
              </div>
              <div className="flex justify-between">
                <span>Z_projected</span>
                <DimBadge dims={`${N_TOKENS} × ${dLM}`} variant="output" />
              </div>
            </div>

            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={() => setSeed((s) => s + 1)}>
                <Dice5 className="h-3.5 w-3.5" /> New data
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setSeed(7)}>
                <RefreshCw className="h-3.5 w-3.5" /> Reset
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Main work area */}
        <div className="space-y-4">
          {/* Dimension change banner */}
          <Card className="border-primary/30">
            <CardContent className="pt-4">
              <div className="flex flex-wrap items-center justify-center gap-3 text-sm">
                <div className="flex flex-col items-center">
                  <span className="text-[10px] uppercase text-muted-foreground">Vision space</span>
                  <DimBadge dims={`ℝ^${DV}`} variant="input" />
                </div>
                <MathBlock>{`\\xrightarrow{\\;W_P \\in \\mathbb{R}^{${DV} \\times ${dLM}}\\;}`}</MathBlock>
                <div className="flex flex-col items-center">
                  <span className="text-[10px] uppercase text-muted-foreground">LM space</span>
                  <DimBadge dims={`ℝ^${dLM}`} variant="output" />
                </div>
                <span className="text-xs text-muted-foreground">
                  (each visual token is independently re-embedded)
                </span>
              </div>
            </CardContent>
          </Card>

          {/* Step 1: matrix multiply */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Step 1 — Matrix multiply: Z_visual · W_P</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="grid gap-3 sm:grid-cols-[auto_auto_auto_auto_auto_auto] sm:items-center">
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">
                    Z_visual ({zRows}×{zCols})
                  </p>
                  <MatrixView matrix={Z_visual} digits={2} heatmap diverging cellSize="sm" />
                </div>
                <div className="text-2xl text-muted-foreground">·</div>
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">
                    W_P ({wRows}×{wCols})
                  </p>
                  <MatrixView matrix={W_P} digits={2} heatmap diverging cellSize="sm" />
                </div>
                <div className="text-2xl text-muted-foreground">+</div>
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">b ({b.length})</p>
                  <MatrixView matrix={[b]} digits={2} heatmap diverging cellSize="sm" />
                </div>
                <div className="text-2xl text-muted-foreground">=</div>
              </div>
              <p className="text-xs text-muted-foreground">
                Each row of <code>Z_visual</code> is one visual token's d_v-dim feature vector. Each{" "}
                column of <code>W_P</code> is a direction in the LM space. Their dot product is a
                coordinate in LM space. Bias shifts the origin per LM dimension.
              </p>
            </CardContent>
          </Card>

          {/* Step 2: result */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">
                Step 2 — Projected visual tokens Z_projected ({pRows}×{pCols})
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <MatrixView
                matrix={Z_projected}
                digits={2}
                heatmap
                diverging
                cellSize="md"
                rowLabels={Z_projected.map((_, i) => `t${i}`)}
                colLabels={Z_projected[0].map((_, j) => `lm${j}`)}
                highlight={cell ?? undefined}
                onCellClick={(i, j) => setCell([i, j])}
              />
              <p className="text-xs text-muted-foreground">
                Click any output cell to see how it was computed from{" "}
                <code>Z_visual[i]</code>, <code>W_P[:,j]</code>, and <code>b[j]</code>.
              </p>

              {cell && (
                <div className="rounded border-2 border-primary/40 bg-primary/5 p-3 text-xs">
                  {(() => {
                    const [i, j] = cell;
                    const e = explain(i, j);
                    return (
                      <div className="space-y-1 font-mono">
                        <p className="font-sans font-medium">
                          Output[{i},{j}] = (Z_visual[{i}] · W_P[:,{j}]) + b[{j}]
                        </p>
                        <p>Z_visual[{i}] = {fmtVec(e.zRow, 3)}</p>
                        <p>W_P[:,{j}] = {fmtVec(e.wCol, 3)}</p>
                        <p>dot product = {e.dot.toFixed(4)}</p>
                        <p>b[{j}] = {e.bias.toFixed(4)}</p>
                        <p className="pt-1 text-primary">
                          Z_projected[{i},{j}] = {e.dot.toFixed(4)} + {e.bias.toFixed(4)} ={" "}
                          {e.out.toFixed(4)}
                        </p>
                        <p className="pt-1 text-muted-foreground font-sans">
                          Interpretation: visual token {i} is now embedded in the LM's d_LM={" "}
                          {dLM}-dim space; coordinate {j} measures its alignment with the{" "}
                          {j}-th learned direction.
                        </p>
                      </div>
                    );
                  })()}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Comparison expandable sections */}
          <div className="grid gap-3 md:grid-cols-2">
            <ExpandableSection title="Why is this 'just' a linear layer?" variant="math" defaultOpen>
              <p>
                A linear layer is{" "}
                <MathBlock>{`y = Wx + b`}</MathBlock> — a single affine transformation.
                No nonlinearity, no attention, no recurrence. It is the simplest possible way to
                convert a d_v-dim feature into a d_LM-dim embedding.
              </p>
              <p className="mt-2">
                The reason it works at all is that the vision encoder (CLIP-ViT, etc.) has already
                done the heavy lifting: its features are already semantically meaningful, so a single
                projection is often enough to align them with the LM's token-embedding space.
              </p>
              <p className="mt-2">
                The projector's parameters are the <em>only</em> thing trained from scratch in
                LLaVA stage 1 — the encoder and LLM are frozen.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Beyond linear: MLP, Q-Former, Resampler" variant="how" defaultOpen>
              <p>
                Real VLMs use richer projectors when a single linear layer is insufficient:
              </p>
              <ul className="mt-2 list-disc space-y-2 pl-5">
                <li>
                  <strong>MLP (LLaVA).</strong>{" "}
                  <MathBlock>{`Z_p = \\text{GELU}(Z W_1 + b_1) W_2 + b_2`}</MathBlock>
                  Two linear layers with a GELU nonlinearity — small capacity but enough to adapt
                  the embedding distribution.
                </li>
                <li>
                  <strong>Q-Former (BLIP-2).</strong> A small transformer with M learnable queries
                  that <em>cross-attend</em> to the N image features, producing a fixed-size M-token
                  summary (Module 13). Acts both as projector and compressor.
                </li>
                <li>
                  <strong>Perceiver Resampler (Flamingo).</strong> Similar to Q-Former — a small
                  cross-attention module that resamples a variable number of visual features into a
                  fixed number of latents.
                </li>
              </ul>
              <p className="mt-2">
                All three share the same goal: produce visual tokens in the LM's embedding space,
                optionally with token-count compression.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Why does d_LM matter?" variant="why">
              <p>
                The LM was pretrained with token embeddings of dimension d_LM (e.g. 4096 for
                LLaMA-7B, 2048 for GPT-2-small). If the projector outputs a different dim, the LM's
                first layer cannot consume it — the inner dimensions of any matmul must match.
              </p>
              <p className="mt-2">
                Try changing d_LM in the slider: the projector's W_P grows or shrinks accordingly,
                but the visual tokens (rows of Z_visual) stay the same shape. The projector is the{" "}
                <em>shape adapter</em>.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Intuition: 'translating' between two languages" variant="intuition">
              <p>
                The vision encoder speaks a 'vision language' (a d_v-dim vector space where nearby
                points look similar visually). The LM speaks a 'language language' (a d_LM-dim
                embedding space where nearby tokens are semantically related).
              </p>
              <p className="mt-2">
                The projector is a translator: it takes each visual sentence (token) and rewrites
                it in the LM's vocabulary. The translation is <em>linear</em> because we want
                similar visual features to map to similar LM embeddings — preserving the structure of
                the source space.
              </p>
              <p className="mt-2">
                A nonlinear MLP can additionally <em>warp</em> the space (e.g. thresholding,
                sparsification) which sometimes helps when the two spaces are not naturally aligned.
              </p>
            </ExpandableSection>
          </div>

          <VivaPanel
            questions={[
              {
                level: "Easy",
                question: "What shape are W_P and b in a linear projector from d_v to d_LM?",
                hint: "Inner dimensions must match for matrix multiplication.",
                answer:
                  "W_P ∈ ℝ^(d_v × d_LM) and b ∈ ℝ^(d_LM). Z_visual (N × d_v) multiplied by W_P gives (N × d_LM), then b is broadcast across rows.",
                explanation:
                  "If d_v ≠ d_LM, the projector is the only place that fixes the dimension mismatch — both the vision encoder and the LM have fixed input/output dims from pretraining.",
              },
              {
                level: "Medium",
                question:
                  "Why does LLaVA use a 2-layer MLP instead of a single linear layer for the projector?",
                hint: "Think about the role of nonlinearity.",
                answer:
                  "A linear layer can only do an affine re-embedding (rotation + translation + scaling). An MLP with a GELU nonlinearity can additionally warp the space — collapsing unimportant directions, thresholding, sparsifying — which improves alignment between CLIP-ViT features and the LM's token embedding manifold.",
                explanation:
                  "Ablation studies in the LLaVA paper show MLP outperforms a single linear layer by a few points on VQA benchmarks. The capacity is still tiny compared to the LM, so training is cheap.",
              },
              {
                level: "Difficult",
                question:
                  "A linear projector preserves the rank and the principal angles of Z_visual. Why is this desirable?",
                hint: "Consider what information a linear map can and cannot destroy.",
                answer:
                  "A linear map y = Wx + b cannot collapse distinct inputs onto the same output if W has full column rank — it preserves linear independence and the cosine geometry of the source space. This means visually distinct features remain distinct after projection, so the LM gets the full information the encoder produced. A nonlinear projector could in principle destroy information (e.g. ReLU collapses all negatives to zero); a linear one cannot.",
                explanation:
                  "In practice a pure linear projector is too weak because the LM's embedding space is non-Euclidean (curved manifold). The 2-layer MLP adds exactly enough nonlinearity to adapt to that curvature while preserving most of the encoder's geometric structure. This is the 'minimum viable projector' design principle behind LLaVA.",
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
}
