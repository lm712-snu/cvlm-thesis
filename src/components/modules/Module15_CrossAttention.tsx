"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { RefreshCw, Dice5 } from "lucide-react";
import { MatrixView } from "@/components/cvlm/MatrixView";
import { Heatmap } from "@/components/cvlm/Heatmap";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import { crossAttention } from "@/lib/math/attention";
import { randMatrix, fmtVec } from "@/lib/math/matrix";

const TEXT_TOKENS = ["a", "dog", "runs"];
const IMAGE_PATCHES = ["p0", "p1", "p2", "p3"];
const D_TEXT = 3;
const D_IMG = 3;
const D_K = 2;

export function Module15_CrossAttention() {
  const [seed, setSeed] = useState(11);
  const [hovered, setHovered] = useState<[number, number] | null>([1, 1]);

  const { X_text, X_image, W_Q, W_K, W_V, attn } = useMemo(() => {
    const X_text = randMatrix(TEXT_TOKENS.length, D_TEXT, -2, 2, seed);
    const X_image = randMatrix(IMAGE_PATCHES.length, D_IMG, -2, 2, seed + 1);
    const W_Q = randMatrix(D_TEXT, D_K, -1, 1, seed + 2);
    const W_K = randMatrix(D_IMG, D_K, -1, 1, seed + 3);
    const W_V = randMatrix(D_IMG, D_K, -1, 1, seed + 4);
    const attn = crossAttention(X_text, X_image, W_Q, W_K, W_V);
    return { X_text, X_image, W_Q, W_K, W_V, attn };
  }, [seed]);

  // Explain a clicked cell of the attention weights matrix
  const explain = (i: number, j: number) => {
    const qi = attn.Q[i];
    const kj = attn.K[j];
    const dot = qi.reduce((s, q, k) => s + q * kj[k], 0);
    const scale = Math.sqrt(attn.dK);
    const score = dot / scale;
    return { qi, kj, dot, score, weight: attn.weights[i][j] };
  };

  const e = hovered ? explain(hovered[0], hovered[1]) : null;

  return (
    <div>
      <ModuleHeader
        number={15}
        title="Cross-Modal Attention"
        subtitle="Queries come from text; keys and values come from the image. Each text token learns which image patches to read from."
      >
        <MathBlock block>
          {`\\text{Attention}(\\text{text}, \\text{image}) = \\text{softmax}\\!\\left(\\frac{Q_{\\text{text}} K_{\\text{image}}^\\top}{\\sqrt{d_k}}\\right) V_{\\text{image}}`}
        </MathBlock>
      </ModuleHeader>

      <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
        {/* Side controls */}
        <Card className="h-fit">
          <CardHeader>
            <CardTitle className="text-base">Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={() => setSeed((s) => s + 1)}>
                <Dice5 className="h-3.5 w-3.5" /> New data
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setSeed(11)}>
                <RefreshCw className="h-3.5 w-3.5" /> Reset
              </Button>
            </div>

            <div className="rounded border bg-muted/30 p-2 text-xs space-y-1">
              <div className="flex justify-between">
                <span>X_text (text tokens)</span>
                <DimBadge dims={`${TEXT_TOKENS.length} × ${D_TEXT}`} variant="input" />
              </div>
              <div className="flex justify-between">
                <span>X_image (image patches)</span>
                <DimBadge dims={`${IMAGE_PATCHES.length} × ${D_IMG}`} variant="input" />
              </div>
              <div className="flex justify-between">
                <span>W_Q (text)</span>
                <DimBadge dims={`${D_TEXT} × ${D_K}`} variant="weight" />
              </div>
              <div className="flex justify-between">
                <span>W_K, W_V (image)</span>
                <DimBadge dims={`${D_IMG} × ${D_K}`} variant="weight" />
              </div>
              <div className="flex justify-between">
                <span>Q (text)</span>
                <DimBadge dims={`${TEXT_TOKENS.length} × ${D_K}`} variant="intermediate" />
              </div>
              <div className="flex justify-between">
                <span>K, V (image)</span>
                <DimBadge dims={`${IMAGE_PATCHES.length} × ${D_K}`} variant="intermediate" />
              </div>
              <div className="flex justify-between">
                <span>Attention weights</span>
                <DimBadge dims={`${TEXT_TOKENS.length} × ${IMAGE_PATCHES.length}`} variant="intermediate" />
              </div>
              <div className="flex justify-between">
                <span>Output</span>
                <DimBadge dims={`${TEXT_TOKENS.length} × ${D_K}`} variant="output" />
              </div>
            </div>

            <div className="rounded border bg-muted/20 p-2 text-[11px]">
              <p className="font-semibold uppercase tracking-wide text-muted-foreground">
                Text tokens
              </p>
              <ul className="mt-1 space-y-0.5 font-mono">
                {TEXT_TOKENS.map((t, i) => (
                  <li key={i}>t{i} = &quot;{t}&quot;</li>
                ))}
              </ul>
              <p className="mt-2 font-semibold uppercase tracking-wide text-muted-foreground">
                Image patches
              </p>
              <ul className="mt-1 space-y-0.5 font-mono">
                {IMAGE_PATCHES.map((p, i) => (
                  <li key={i}>{p}</li>
                ))}
              </ul>
            </div>
          </CardContent>
        </Card>

        {/* Main work area */}
        <div className="space-y-4">
          {/* Step 1: projections */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">
                Step 1 — Project text → Q, image → K and V
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded border bg-muted/20 p-3">
                  <p className="mb-1 text-xs uppercase text-muted-foreground">
                    Q = X_text · W_Q  <DimBadge dims={`${TEXT_TOKENS.length} × ${D_K}`} variant="intermediate" />
                  </p>
                  <div className="grid grid-cols-[auto_auto_auto] items-center gap-1">
                    <MatrixView matrix={X_text} digits={2} heatmap diverging cellSize="xs"
                      rowLabels={TEXT_TOKENS.map((_, i) => `t${i}`)}
                    />
                    <span className="px-1 text-lg text-muted-foreground">·</span>
                    <MatrixView matrix={W_Q} digits={2} heatmap diverging cellSize="xs" />
                  </div>
                  <p className="mt-1 text-[10px] text-muted-foreground">→</p>
                  <MatrixView matrix={attn.Q} digits={2} heatmap diverging cellSize="sm"
                    rowLabels={TEXT_TOKENS.map((_, i) => `t${i}`)}
                    colLabels={attn.Q[0].map((_, j) => `q${j}`)}
                  />
                </div>

                <div className="rounded border bg-muted/20 p-3">
                  <p className="mb-1 text-xs uppercase text-muted-foreground">
                    K = X_image · W_K  <DimBadge dims={`${IMAGE_PATCHES.length} × ${D_K}`} variant="intermediate" />
                  </p>
                  <div className="grid grid-cols-[auto_auto_auto] items-center gap-1">
                    <MatrixView matrix={X_image} digits={2} heatmap diverging cellSize="xs"
                      rowLabels={IMAGE_PATCHES}
                    />
                    <span className="px-1 text-lg text-muted-foreground">·</span>
                    <MatrixView matrix={W_K} digits={2} heatmap diverging cellSize="xs" />
                  </div>
                  <p className="mt-1 text-[10px] text-muted-foreground">→</p>
                  <MatrixView matrix={attn.K} digits={2} heatmap diverging cellSize="sm"
                    rowLabels={IMAGE_PATCHES}
                    colLabels={attn.K[0].map((_, j) => `k${j}`)}
                  />
                </div>

                <div className="rounded border bg-muted/20 p-3 sm:col-span-2">
                  <p className="mb-1 text-xs uppercase text-muted-foreground">
                    V = X_image · W_V  <DimBadge dims={`${IMAGE_PATCHES.length} × ${D_K}`} variant="intermediate" />
                  </p>
                  <div className="grid grid-cols-[auto_auto_auto] items-center gap-1">
                    <MatrixView matrix={X_image} digits={2} heatmap diverging cellSize="xs"
                      rowLabels={IMAGE_PATCHES}
                    />
                    <span className="px-1 text-lg text-muted-foreground">·</span>
                    <MatrixView matrix={W_V} digits={2} heatmap diverging cellSize="xs" />
                  </div>
                  <p className="mt-1 text-[10px] text-muted-foreground">→</p>
                  <MatrixView matrix={attn.V} digits={2} heatmap diverging cellSize="sm"
                    rowLabels={IMAGE_PATCHES}
                    colLabels={attn.V[0].map((_, j) => `v${j}`)}
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Step 2: attention weights */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">
                Step 2 — Attention scores and weights (heatmap, click any cell)
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <p className="text-xs text-muted-foreground">
                Each <strong>row</strong> is a text token's distribution over image patches.
                Brighter = stronger attention. Row i answers: <em>&quot;which patches does text
                token i read from?&quot;</em>
              </p>

              <div className="grid gap-3 md:grid-cols-2">
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">
                    Scores S = Q · Kᵀ / √d_k  <DimBadge dims={`${TEXT_TOKENS.length} × ${IMAGE_PATCHES.length}`} variant="intermediate" />
                  </p>
                  <MatrixView
                    matrix={attn.scores}
                    digits={2}
                    heatmap
                    diverging
                    cellSize="sm"
                    rowLabels={TEXT_TOKENS.map((_, i) => `t${i} "${TEXT_TOKENS[i]}"`)}
                    colLabels={IMAGE_PATCHES}
                    highlight={hovered ?? undefined}
                    onCellClick={(i, j) => setHovered([i, j])}
                  />
                </div>
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">
                    Weights A = softmax(S)  (rows sum to 1)
                  </p>
                  <Heatmap
                    matrix={attn.weights}
                    min={0}
                    max={1}
                    cellSize={48}
                    rowLabels={TEXT_TOKENS.map((_, i) => `t${i} "${TEXT_TOKENS[i]}"`)}
                    colLabels={IMAGE_PATCHES}
                    highlight={hovered ?? undefined}
                    onCellClick={(i, j) => setHovered([i, j])}
                    format={(v) => v.toFixed(2)}
                  />
                </div>
              </div>

              {e && hovered && (
                <div className="rounded border-2 border-primary/40 bg-primary/5 p-3 text-xs">
                  <p className="mb-2 font-medium font-sans">
                    Cell A[{hovered[0]}, {hovered[1]}] — text token &quot;{TEXT_TOKENS[hovered[0]]}&quot;
                    attending to image patch {IMAGE_PATCHES[hovered[1]]}
                  </p>
                  <div className="space-y-1 font-mono">
                    <p>q_{hovered[0]} (text) = {fmtVec(e.qi, 3)}</p>
                    <p>k_{hovered[1]} (image) = {fmtVec(e.kj, 3)}</p>
                    <p>q · k = {e.dot.toFixed(4)}</p>
                    <p>S[{hovered[0]},{hovered[1]}] = (q·k) / √{attn.dK} = {e.score.toFixed(4)}</p>
                    <p className="text-primary">
                      A[{hovered[0]},{hovered[1]}] = softmax(S[{hovered[0]}, :]) = {e.weight.toFixed(4)}
                    </p>
                    <p className="pt-1 text-muted-foreground font-sans">
                      Interpretation: text token &quot;{TEXT_TOKENS[hovered[0]]}&quot; reads{" "}
                      {(e.weight * 100).toFixed(1)}% of its information from patch {IMAGE_PATCHES[hovered[1]]}.
                      The remaining {((1 - e.weight) * 100).toFixed(1)}% is split across the other patches
                      (see row {hovered[0]} of the weights matrix).
                    </p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Step 3: output */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">
                Step 3 — Output: each text token becomes an image-aware embedding
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="grid gap-3 sm:grid-cols-[auto_auto_auto] sm:items-center">
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">A (weights)</p>
                  <Heatmap matrix={attn.weights} min={0} max={1} cellSize={24}
                    rowLabels={TEXT_TOKENS.map((_, i) => `t${i}`)}
                    colLabels={IMAGE_PATCHES}
                    highlight={hovered ?? undefined}
                    onCellClick={(i, j) => setHovered([i, j])}
                    format={(v) => v.toFixed(2)}
                  />
                </div>
                <div className="text-2xl text-muted-foreground">·</div>
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">
                    V (image content) <DimBadge dims={`${IMAGE_PATCHES.length} × ${D_K}`} variant="intermediate" />
                  </p>
                  <MatrixView matrix={attn.V} digits={2} heatmap diverging cellSize="sm"
                    rowLabels={IMAGE_PATCHES}
                  />
                </div>
              </div>
              <p className="text-[10px] text-muted-foreground">↓ Output = A · V</p>
              <div>
                <p className="mb-1 text-xs uppercase text-muted-foreground">
                  Output <DimBadge dims={`${TEXT_TOKENS.length} × ${D_K}`} variant="output" />
                </p>
                <MatrixView
                  matrix={attn.output}
                  digits={2}
                  heatmap
                  diverging
                  cellSize="md"
                  rowLabels={TEXT_TOKENS.map((t, i) => `t${i} "${t}"`)}
                  colLabels={attn.output[0].map((_, j) => `o${j}`)}
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Each output row is a convex combination of V's rows, weighted by the corresponding
                row of A. Text tokens that strongly attend to patch p will receive mostly V[p].
              </p>
            </CardContent>
          </Card>

          {/* Expanded explanations */}
          <div className="grid gap-3 md:grid-cols-2">
            <ExpandableSection title="Cross-attention vs self-attention" variant="math" defaultOpen>
              <p>
                Self-attention (Module 8): Q, K, V all come from the same sequence X —
                <MathBlock>{`Q = K = V = X`}</MathBlock> (after projection). It lets tokens in one
                sequence talk to each other.
              </p>
              <p className="mt-2">
                Cross-attention: Q comes from one sequence (text), K and V from another (image). It
                lets the text sequence <em>read information out of</em> the image sequence. The text
                token decides <em>what to ask for</em> (its query); the image patches provide{" "}
                <em>what they offer</em> (their keys) and <em>what they actually are</em> (their
                values).
              </p>
              <p className="mt-2">
                The output dimension matches the query sequence length (n_text rows) — every text
                token gets exactly one image-aware embedding back.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Why is the output shape (n_text, d_v), not (n_image, d_v)?" variant="why">
              <p>
                The shape of <MathBlock>{`AV`}</MathBlock> is determined by the{" "}
                <em>query</em> side, not the key side. Since A is{" "}
                <MathBlock>{`n_{\\text{text}} \\times n_{\\text{image}}`}</MathBlock> and V is{" "}
                <MathBlock>{`n_{\\text{image}} \\times d_v`}</MathBlock>, the product is{" "}
                <MathBlock>{`n_{\\text{text}} \\times d_v`}</MathBlock>.
              </p>
              <p className="mt-2">
                Intuitively, every text token asks a question and receives one answer; the image
                patches just supply the lookup table. The text side stays fixed; the image side is
                consumed and discarded.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Where does cross-attention live in a VLM?" variant="how">
              <p>
                Three places, depending on the architecture:
              </p>
              <ol className="mt-2 list-decimal space-y-1 pl-5">
                <li>
                  Inside the Q-Former (BLIP-2): a stack of cross-attention layers, each with
                  learnable query tokens attending to the ViT features.
                </li>
                <li>
                  Inside an encoder-decoder LM (PaLI, T5-based): the decoder's cross-attention
                  layers attend to the projected visual tokens at every generation step.
                </li>
                <li>
                  Implicitly, in decoder-only LMs (LLaVA): the visual tokens are prepended to the
                  text sequence and the LM's <em>self-attention</em> mixes text and image —
                  equivalent to cross-attention in effect, even if not in name.
                </li>
              </ol>
            </ExpandableSection>

            <ExpandableSection title="Intuition: 'reading' from the image" variant="intuition">
              <p>
                When you read the sentence &quot;a dog runs&quot; while looking at a picture, your
                eye flicks to the part of the image relevant to each word: maybe patch p1 when you
                read &quot;dog&quot;, and patches p2/p3 when you read &quot;runs&quot;.
              </p>
              <p className="mt-2">
                Cross-attention does the same: each text token computes a softmax distribution over
                image patches — its &quot;gaze&quot; — and reads back the weighted average of those
                patches' values. The result is an image-aware text embedding.
              </p>
              <p className="mt-2">
                Click different cells in the heatmap above to see how each text token's gaze is
                distributed across the four image patches.
              </p>
            </ExpandableSection>
          </div>

          <VivaPanel
            questions={[
              {
                level: "Easy",
                question: "In cross-modal attention, where do Q, K, and V come from respectively?",
                hint: "One side asks, the other answers.",
                answer:
                  "Q comes from the text (queries), K and V come from the image (keys and values). The text decides where to look; the image supplies both the 'addresses' (K) and the 'contents' (V).",
                explanation:
                  "This is what makes it 'cross' — the K and V sources are different from the Q source. In self-attention all three come from the same sequence.",
              },
              {
                level: "Medium",
                question:
                  "If the text sequence has 3 tokens and the image has 4 patches with d_k = 2, what are the shapes of the attention scores and the output?",
                hint: "Apply the matmul shapes carefully.",
                answer:
                  "Scores S = Q · Kᵀ is (3 × 2) · (2 × 4) = 3 × 4. Weights A = softmax(S) is also 3 × 4 (rows sum to 1). Output = A · V is (3 × 4) · (4 × 2) = 3 × 2.",
                explanation:
                  "The output has the same number of rows as the query sequence (n_text) and the same number of columns as d_v. The image's sequence length (n_image) only affects the inner dimension — it is consumed.",
              },
              {
                level: "Difficult",
                question:
                  "Why is cross-attention well-suited for tasks like image captioning and VQA, but suboptimal for retrieval where you must score (image, text) pairs in bulk?",
                hint: "Consider directionality and what is being computed.",
                answer:
                  "Cross-attention is directional: it produces image-aware text embeddings, useful when the goal is to generate text conditioned on the image. For retrieval (CLIP-style), you need symmetric scores between independently encoded images and texts — a contrastive dot product in a shared space is O(1) per pair after pre-encoding, whereas cross-attention must be recomputed for every (image, text) query pair, which is O(N·M) and asymmetric.",
                explanation:
                  "This is why CLIP uses dual encoders + cosine similarity for retrieval (cheap, symmetric, scalable) while BLIP-2 / PaLI use cross-attention for generation (rich, directional, expensive). The two paradigms are complementary and many modern systems combine them: CLIP for retrieval, a cross-attention VLM for generation.",
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
}
