"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { MatrixView } from "@/components/cvlm/MatrixView";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import { layerNorm, matmul, addBias, randMatrix, shape } from "@/lib/math/matrix";
import { Boxes, ArrowRight, ArrowDown, Plus, RefreshCw } from "lucide-react";

export function Module7_ViT() {
  // Small numerical example dims.
  const [n, setN] = useState(4);
  const [d, setD] = useState(4);
  const [seed, setSeed] = useState(3);

  const { X, ln1, attnOut, X1, ln2, mlpOut, X2 } = useMemo(() => {
    const X = randMatrix(n, d, -1, 1, seed);
    const ln1 = layerNorm(X);
    // Simplified "attention" output: a small random linear projection of ln1
    // back into the same shape, so the residual add is shape-preserving.
    // Real attention is detailed in Modules 8 and 9.
    const Wa = randMatrix(d, d, -0.5, 0.5, seed + 1);
    const ba = Array.from({ length: d }, (_, i) => Number((((seed + i) * 7) % 100) / 200 - 0.25));
    const attnOut = addBias(matmul(ln1, Wa), ba);
    const X1 = X.map((row, i) => row.map((v, j) => v + attnOut[i][j]));
    const ln2 = layerNorm(X1);
    // MLP: linear(4 → 2x d) → linear(2x d → d). For demo, single linear.
    const Wm = randMatrix(d, d, -0.5, 0.5, seed + 2);
    const bm = Array.from({ length: d }, (_, i) => Number((((seed + i) * 11) % 100) / 200 - 0.25));
    const mlpOut = addBias(matmul(ln2, Wm), bm);
    const X2 = X1.map((row, i) => row.map((v, j) => v + mlpOut[i][j]));
    return { X, ln1, attnOut, X1, ln2, mlpOut, X2 };
  }, [n, d, seed]);

  const [nR, dR] = shape(X);

  // Pipeline stages for the top flow diagram.
  const pipeStages = [
    { label: "Image", dim: "H × W × C", variant: "input" as const },
    { label: "Patches", dim: "N × P²C", variant: "intermediate" as const },
    { label: "Patch Embed", dim: "N × d", variant: "intermediate" as const },
    { label: "+ Positional", dim: "N × d", variant: "intermediate" as const },
    { label: `Encoder × L`, dim: "N × d", variant: "weight" as const },
    { label: "Visual Features", dim: "N × d", variant: "output" as const },
  ];

  return (
    <div>
      <ModuleHeader
        number={7}
        title="Vision Transformer (ViT)"
        subtitle="Image → patches → embeddings + PE → stack of transformer encoder blocks → visual features. The encoder block is the repeatable unit: LayerNorm + Attention + residual, then LayerNorm + MLP + residual."
      >
        <MathBlock block>
          {`X' = X + \\text{Attn}(\\text{LN}(X)), \\quad X'' = X' + \\text{MLP}(\\text{LN}(X'))`}
        </MathBlock>
      </ModuleHeader>

      <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
        <Card className="h-fit">
          <CardHeader><CardTitle className="text-base">Configuration</CardTitle></CardHeader>
          <CardContent className="space-y-4 text-sm">
            <div>
              <div className="flex justify-between"><Label className="text-xs">Sequence N</Label><span className="font-mono text-xs">{n}</span></div>
              <Slider value={[n]} min={2} max={8} onValueChange={([v]) => setN(v ?? 4)} />
            </div>
            <div>
              <div className="flex justify-between"><Label className="text-xs">Embed dim d</Label><span className="font-mono text-xs">{d}</span></div>
              <Slider value={[d]} min={2} max={8} onValueChange={([v]) => setD(v ?? 4)} />
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setSeed((s) => s + 1)}
                className="inline-flex items-center gap-1 rounded border px-2 py-1 text-xs hover:bg-accent/40"
              >
                <RefreshCw className="h-3 w-3" /> New input
              </button>
            </div>
            <div className="rounded border bg-muted/30 p-2 text-xs space-y-1">
              <div className="flex justify-between"><span>Input X</span><DimBadge dims={`${nR} × ${dR}`} variant="input" /></div>
              <div className="flex justify-between"><span>After LN₁ + Attn + residual</span><DimBadge dims={`${nR} × ${dR}`} variant="intermediate" /></div>
              <div className="flex justify-between"><span>After LN₂ + MLP + residual</span><DimBadge dims={`${nR} × ${dR}`} variant="output" /></div>
              <div className="mt-1 text-muted-foreground">
                Note: shape is preserved by every residual — this is the key property of residual connections.
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-4">
          {/* Full pipeline diagram */}
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-sm flex items-center gap-2"><Boxes className="h-4 w-4" /> End-to-end ViT pipeline</CardTitle></CardHeader>
            <CardContent>
              <div className="flex flex-wrap items-center gap-2">
                {pipeStages.map((s, i) => (
                  <div key={s.label} className="flex items-center gap-2">
                    <div className="rounded-lg border-2 bg-card px-3 py-2 text-center min-w-[90px]">
                      <div className="text-xs font-semibold">{s.label}</div>
                      <div className="mt-1"><DimBadge dims={s.dim} variant={s.variant} /></div>
                    </div>
                    {i < pipeStages.length - 1 && (
                      <ArrowRight className="h-4 w-4 text-muted-foreground" />
                    )}
                  </div>
                ))}
              </div>
              <p className="mt-3 text-[11px] text-muted-foreground">
                Everything outside the "Encoder × L" box is fixed (deterministic from the image).
                The encoder block is repeated L times (12 for ViT-Base, 24 for ViT-Large, 32 for ViT-Huge).
              </p>
            </CardContent>
          </Card>

          {/* Single encoder block diagram */}
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-sm">One transformer encoder block (residual + norm)</CardTitle></CardHeader>
            <CardContent>
              <div className="rounded-lg border bg-muted/20 p-4">
                <div className="flex flex-col items-center gap-3">
                  {/* Input */}
                  <div className="rounded-lg border-2 border-emerald-500/40 bg-emerald-500/5 px-4 py-2 text-center">
                    <div className="text-xs font-semibold">Input X</div>
                    <DimBadge dims="N × d" variant="input" />
                  </div>
                  <ArrowDown className="h-4 w-4 text-muted-foreground" />
                  {/* Sub-block 1: LN → Attn → +Residual */}
                  <div className="rounded-lg border-2 border-sky-500/40 bg-sky-500/5 p-3 w-full max-w-md">
                    <div className="text-[10px] font-bold uppercase tracking-wider text-sky-700 dark:text-sky-300 mb-2">Sub-block 1: Attention</div>
                    <div className="flex flex-col items-center gap-2">
                      <div className="rounded border bg-card px-3 py-1 text-xs">LayerNorm</div>
                      <ArrowDown className="h-3 w-3 text-muted-foreground" />
                      <div className="rounded border bg-card px-3 py-1 text-xs">Multi-Head Self-Attn (Module 9)</div>
                      <ArrowDown className="h-3 w-3 text-muted-foreground" />
                      <div className="flex items-center gap-1">
                        <div className="rounded border border-amber-500/50 bg-amber-500/10 px-2 py-1 text-[10px] font-mono">+ X (residual)</div>
                      </div>
                    </div>
                  </div>
                  <ArrowDown className="h-4 w-4 text-muted-foreground" />
                  {/* Output of sub-block 1 */}
                  <div className="rounded-lg border-2 border-violet-500/40 bg-violet-500/5 px-4 py-2 text-center">
                    <div className="text-xs font-semibold">X' = X + Attn(LN(X))</div>
                    <DimBadge dims="N × d" variant="intermediate" />
                  </div>
                  <ArrowDown className="h-4 w-4 text-muted-foreground" />
                  {/* Sub-block 2: LN → MLP → +Residual */}
                  <div className="rounded-lg border-2 border-fuchsia-500/40 bg-fuchsia-500/5 p-3 w-full max-w-md">
                    <div className="text-[10px] font-bold uppercase tracking-wider text-fuchsia-700 dark:text-fuchsia-300 mb-2">Sub-block 2: MLP</div>
                    <div className="flex flex-col items-center gap-2">
                      <div className="rounded border bg-card px-3 py-1 text-xs">LayerNorm</div>
                      <ArrowDown className="h-3 w-3 text-muted-foreground" />
                      <div className="rounded border bg-card px-3 py-1 text-xs">MLP (Linear → GELU → Linear)</div>
                      <ArrowDown className="h-3 w-3 text-muted-foreground" />
                      <div className="flex items-center gap-1">
                        <div className="rounded border border-amber-500/50 bg-amber-500/10 px-2 py-1 text-[10px] font-mono">+ X' (residual)</div>
                      </div>
                    </div>
                  </div>
                  <ArrowDown className="h-4 w-4 text-muted-foreground" />
                  <div className="rounded-lg border-2 border-emerald-500/40 bg-emerald-500/5 px-4 py-2 text-center">
                    <div className="text-xs font-semibold">X'' = X' + MLP(LN(X'))</div>
                    <DimBadge dims="N × d" variant="output" />
                  </div>
                </div>
              </div>
              <p className="mt-3 text-[11px] text-muted-foreground">
                Pre-norm variant (LN before sub-layer). The residual additions (<Plus className="inline h-3 w-3" />) preserve shape —
                without them, training deep stacks would collapse due to vanishing gradients.
              </p>
            </CardContent>
          </Card>

          {/* Numerical example of the residual connection */}
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-sm">Numerical example: trace one encoder block</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <p className="text-xs text-muted-foreground">
                Using N={n}, d={d}. To keep the math fully visible, "Attn" here is a small linear projection
                of <MathBlock>{`\\text{LN}(X)`}</MathBlock>; the real scaled-dot-product attention is shown in
                Module 8, and multi-head in Module 9.
              </p>

              {/* Sub-block 1 trace */}
              <div className="rounded border bg-muted/20 p-3 space-y-2">
                <p className="text-xs font-semibold text-sky-700 dark:text-sky-300">Sub-block 1: X → LN → Attn → +X → X'</p>
                <div className="grid gap-2 md:grid-cols-4 md:items-center">
                  <div>
                    <p className="text-[10px] uppercase text-muted-foreground">X</p>
                    <MatrixView matrix={X} digits={2} heatmap diverging cellSize="xs" />
                  </div>
                  <div>
                    <p className="text-[10px] uppercase text-muted-foreground">LN(X)</p>
                    <MatrixView matrix={ln1} digits={2} heatmap diverging cellSize="xs" />
                  </div>
                  <div>
                    <p className="text-[10px] uppercase text-muted-foreground">Attn(LN(X))</p>
                    <MatrixView matrix={attnOut} digits={2} heatmap diverging cellSize="xs" />
                  </div>
                  <div>
                    <p className="text-[10px] uppercase text-muted-foreground">X' = X + Attn(LN(X))</p>
                    <MatrixView matrix={X1} digits={2} heatmap diverging cellSize="xs" />
                  </div>
                </div>
                <p className="text-[10px] text-muted-foreground">
                  Note that X and X' have the same shape ({nR} × {dR}). The residual is a small perturbation
                  on top of the identity — when gradients flow back, they have a direct path through the +X term.
                </p>
              </div>

              {/* Sub-block 2 trace */}
              <div className="rounded border bg-muted/20 p-3 space-y-2">
                <p className="text-xs font-semibold text-fuchsia-700 dark:text-fuchsia-300">Sub-block 2: X' → LN → MLP → +X' → X''</p>
                <div className="grid gap-2 md:grid-cols-4 md:items-center">
                  <div>
                    <p className="text-[10px] uppercase text-muted-foreground">X'</p>
                    <MatrixView matrix={X1} digits={2} heatmap diverging cellSize="xs" />
                  </div>
                  <div>
                    <p className="text-[10px] uppercase text-muted-foreground">LN(X')</p>
                    <MatrixView matrix={ln2} digits={2} heatmap diverging cellSize="xs" />
                  </div>
                  <div>
                    <p className="text-[10px] uppercase text-muted-foreground">MLP(LN(X'))</p>
                    <MatrixView matrix={mlpOut} digits={2} heatmap diverging cellSize="xs" />
                  </div>
                  <div>
                    <p className="text-[10px] uppercase text-muted-foreground">X'' = X' + MLP(LN(X'))</p>
                    <MatrixView matrix={X2} digits={2} heatmap diverging cellSize="xs" />
                  </div>
                </div>
                <p className="text-[10px] text-muted-foreground">
                  Same shape again. After L stacked blocks, the output is a {nR} × {dR} matrix of visual features
                  ready for downstream heads (classification, captioning, cross-attention with text, …).
                </p>
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <ExpandableSection title="Mathematics of the encoder block" variant="math" defaultOpen>
              <p>
                A ViT encoder block consists of two sub-blocks, each with a residual connection and a
                LayerNorm:
              </p>
              <MathBlock block>{`X' = X + \\text{MSA}(\\text{LN}(X))`}</MathBlock>
              <MathBlock block>{`X'' = X' + \\text{MLP}(\\text{LN}(X'))`}</MathBlock>
              <p className="mt-2">
                where <MathBlock>{`\\text{MSA}`}</MathBlock> is multi-head self-attention (Module 9) and{" "}
                <MathBlock>{`\\text{MLP}`}</MathBlock> is a position-wise feed-forward network{" "}
                <MathBlock>{`\\text{Linear}(d, 4d) \\to \\text{GELU} \\to \\text{Linear}(4d, d)`}</MathBlock>.
              </p>
              <p className="mt-2">
                The full encoder stacks this block L times: <MathBlock>{`Z_L = \\text{Encoder}_L(\\ldots\\text{Encoder}_1(Z_0))`}</MathBlock>.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Why residual connections?" variant="why" defaultOpen>
              <p>
                Without the <MathBlock>{`+ X`}</MathBlock> term, the block would compute{" "}
                <MathBlock>{`X' = \\text{Attn}(\\text{LN}(X))`}</MathBlock> — the network must learn to
                reconstruct the input every layer. With the residual, the block only needs to learn the
                <em> delta</em>: <MathBlock>{`\\Delta X = \\text{Attn}(\\text{LN}(X))`}</MathBlock>.
              </p>
              <p className="mt-2">
                Two practical effects:
              </p>
              <ol className="list-decimal pl-5 space-y-1 mt-1">
                <li><strong>Gradient flow</strong> — back-prop can skip layers via the residual path, avoiding vanishing gradients in deep stacks.</li>
                <li><strong>Identity init</strong> — a near-zero attention output preserves the input, making deep stacks behave like shallower ones until they learn useful deltas.</li>
              </ol>
            </ExpandableSection>

            <ExpandableSection title="LayerNorm: pre-norm vs post-norm" variant="how">
              <p>
                The original Transformer (Vaswani 2017) used <em>post-norm</em>:{" "}
                <MathBlock>{`X' = \\text{LN}(X + \\text{Attn}(X))`}</MathBlock>. Modern ViT uses
                <em> pre-norm</em>: <MathBlock>{`X' = X + \\text{Attn}(\\text{LN}(X))`}</MathBlock>.
              </p>
              <p className="mt-2">
                Pre-norm keeps the residual path "clean" (no LN between residuals), making deep stacks
                easier to train — same motivation as in ResNet's identity shortcuts. Without pre-norm,
                ViTs deeper than ~12 layers become unstable.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Why an MLP after attention?" variant="intuition">
              <p>
                Attention <em>mixes information across tokens</em> — each output is a weighted average of
                all input tokens. But attention alone cannot transform a single token's features
                independently (it is linear in V). The MLP does exactly that: it applies the same
                nonlinear transformation to each token separately.
              </p>
              <p className="mt-2 text-muted-foreground">
                The 4× hidden expansion gives the MLP enough capacity to perform complex per-token
                reasoning — similar to the "key-value memory" interpretation of feed-forward layers in
                transformers.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Numerical example: tracing shape preservation" variant="numerical">
              <p>
                Start with X of shape <DimBadge dims={`${nR} × ${dR}`} variant="input" />. After one full
                encoder block, X'' is <DimBadge dims={`${nR} × ${dR}`} variant="output" /> — identical
                shape. This is what makes "Encoder × L" possible: the block can be stacked any number of
                times because shape never changes.
              </p>
              <p className="mt-2 text-[11px] text-muted-foreground">
                Channel dim d is preserved throughout: <MathBlock>{`d_{\\text{model}} = d`}</MathBlock> is
                the working dimension of the entire ViT (768 for ViT-Base, 1024 for ViT-Large, 1280 for ViT-Huge).
                Only the MLP's hidden dim differs (4·d).
              </p>
            </ExpandableSection>

            <ExpandableSection title="Connecting to later modules" variant="how">
              <p>
                The actual attention computation in this block is shown in detail in:
              </p>
              <ul className="list-disc pl-5 space-y-1 mt-1 text-sm">
                <li><strong>Module 8 — Self-Attention</strong>: the scaled-dot-product formula <MathBlock>{`\\text{softmax}(QK^\\top/\\sqrt{d_k}) V`}</MathBlock> with explicit matrices.</li>
                <li><strong>Module 9 — Multi-Head Attention</strong>: splitting d_model into h heads and concatenating.</li>
              </ul>
              <p className="mt-2 text-muted-foreground">
                Downstream, the encoder output Z_L is fed to either a CLS-head (classification) or a
                cross-attention layer (Module 15) for vision-language tasks.
              </p>
            </ExpandableSection>
          </div>

          <VivaPanel
            questions={[
              {
                level: "Easy",
                question: "What are the two sub-blocks of a ViT encoder block?",
                hint: "Each has a LayerNorm and a residual.",
                answer:
                  "Sub-block 1: LayerNorm → Multi-Head Self-Attention → residual add. Sub-block 2: LayerNorm → MLP → residual add. The output shape equals the input shape.",
                explanation:
                  "Both sub-blocks follow the same pattern: normalize, transform, add residual. This structure repeats L times — for ViT-Base L = 12.",
              },
              {
                level: "Medium",
                question:
                  "Why does the residual connection preserve the shape of X through the encoder block? Why is this important for stacking L blocks?",
                hint: "Look at the dimension of the attention output vs the input.",
                answer:
                  "Self-attention maps an (N × d) input to an (N × d) output (each query produces one d-dim output, and there are N queries). MLP is per-token, so it also preserves shape. Therefore X + sublayer(X) is shape-preserving. This is what lets the block be stacked L times — every block accepts and produces (N × d).",
                explanation:
                  "Without shape preservation, you'd need to reshape between layers, and the residual path (which is essential for gradient flow in deep stacks) would not be possible.",
              },
              {
                level: "Difficult",
                question:
                  "Why does the ViT use pre-norm (LN before attention) rather than post-norm (LN after the residual add)? What goes wrong with post-norm in deep ViTs?",
                hint: "Consider what the residual path looks like in each case.",
                answer:
                  "With post-norm, X' = LN(X + Attn(X)). The LayerNorm sits on the residual path, so back-prop gradients through the +X term must pass through LN's normalization, which can amplify or suppress gradients depending on the variance. In deep stacks (L > 12) this causes training instability. Pre-norm puts LN inside the sub-block: X' = X + Attn(LN(X)), leaving the residual path a pure identity, so gradients flow cleanly through deep stacks.",
                explanation:
                  "Empirically, post-norm Transformers require careful learning-rate warmup and often fail to train at depth; pre-norm enables stable training of 24- and 32-layer ViTs without such tricks. This is now the de-facto default in modern transformer implementations.",
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
}
