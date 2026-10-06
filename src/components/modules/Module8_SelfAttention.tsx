"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { RefreshCw, Dice5 } from "lucide-react";
import { MatrixView } from "@/components/cvlm/MatrixView";
import { Heatmap } from "@/components/cvlm/Heatmap";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import { selfAttention } from "@/lib/math/attention";
import { randMatrix, shape, fmtVec } from "@/lib/math/matrix";

export function Module8_SelfAttention() {
  const [n, setN] = useState(4);
  const [dModel, setDModel] = useState(3);
  const [dK, setDK] = useState(2);
  const [seed, setSeed] = useState(42);
  const [hovered, setHovered] = useState<[number, number] | null>(null);

  // Generate X, W_Q, W_K, W_V deterministically from seed.
  const { X, W_Q, W_K, W_V, attn } = useMemo(() => {
    const X = randMatrix(n, dModel, -2, 2, seed);
    const W_Q = randMatrix(dModel, dK, -1, 1, seed + 1);
    const W_K = randMatrix(dModel, dK, -1, 1, seed + 2);
    const W_V = randMatrix(dModel, dK, -1, 1, seed + 3); // d_v = d_k for simplicity
    const attn = selfAttention(X, W_Q, W_K, W_V);
    return { X, W_Q, W_K, W_V, attn };
  }, [n, dModel, dK, seed]);

  // Compute how a single attention weight was derived.
  const explainCell = (i: number, j: number) => {
    const qi = attn.Q[i];
    const kj = attn.K[j];
    const dot = qi.reduce((s, q, k) => s + q * kj[k], 0);
    const scale = Math.sqrt(attn.dK);
    const score = dot / scale;
    return { qi, kj, dot, score, weight: attn.weights[i][j] };
  };

  return (
    <div>
      <ModuleHeader
        number={8}
        title="Self-Attention"
        subtitle="The heart of every transformer: Q, K, V → softmax(QKᵀ/√d_k)V. Explicit matrices, no library calls."
      >
        <MathBlock block>
          {`\\text{Attention}(X) = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V, \\quad Q = XW_Q,\\ K = XW_K,\\ V = XW_V`}
        </MathBlock>
      </ModuleHeader>

      <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
        <Card className="h-fit">
          <CardHeader><CardTitle className="text-base">Configuration</CardTitle></CardHeader>
          <CardContent className="space-y-4 text-sm">
            <div>
              <div className="flex justify-between"><Label className="text-xs">Sequence length n</Label><span className="font-mono text-xs">{n}</span></div>
              <Slider value={[n]} min={2} max={8} onValueChange={([v]) => setN(v ?? 4)} />
            </div>
            <div>
              <div className="flex justify-between"><Label className="text-xs">d_model</Label><span className="font-mono text-xs">{dModel}</span></div>
              <Slider value={[dModel]} min={2} max={6} onValueChange={([v]) => setDModel(v ?? 3)} />
            </div>
            <div>
              <div className="flex justify-between"><Label className="text-xs">d_k = d_v</Label><span className="font-mono text-xs">{dK}</span></div>
              <Slider value={[dK]} min={2} max={4} onValueChange={([v]) => setDK(v ?? 2)} />
            </div>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={() => setSeed((s) => s + 1)}>
                <Dice5 className="h-3.5 w-3.5" /> New weights
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setSeed(42)}>
                <RefreshCw className="h-3.5 w-3.5" /> Reset
              </Button>
            </div>
            <div className="rounded border bg-muted/30 p-2 text-xs space-y-1">
              <div className="flex justify-between"><span>X</span><DimBadge dims={`${n} × ${dModel}`} variant="input" /></div>
              <div className="flex justify-between"><span>W_Q, W_K, W_V</span><DimBadge dims={`${dModel} × ${dK}`} variant="weight" /></div>
              <div className="flex justify-between"><span>Q, K, V</span><DimBadge dims={`${n} × ${dK}`} variant="intermediate" /></div>
              <div className="flex justify-between"><span>Scores</span><DimBadge dims={`${n} × ${n}`} variant="intermediate" /></div>
              <div className="flex justify-between"><span>Output</span><DimBadge dims={`${n} × ${dK}`} variant="output" /></div>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Step-by-step computation</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              {/* Q = X W_Q */}
              <div className="grid gap-2 sm:grid-cols-[auto_auto_auto_auto] sm:items-center">
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">Q = X · W_Q</p>
                  <MatrixView matrix={X} digits={2} heatmap diverging cellSize="xs" />
                </div>
                <div className="text-2xl text-muted-foreground">·</div>
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">W_Q</p>
                  <MatrixView matrix={W_Q} digits={2} heatmap diverging cellSize="xs" />
                </div>
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">= Q</p>
                  <MatrixView matrix={attn.Q} digits={2} heatmap diverging cellSize="xs" />
                </div>
              </div>
              <div className="grid gap-2 sm:grid-cols-[auto_auto_auto_auto] sm:items-center">
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">K = X · W_K</p>
                  <MatrixView matrix={X} digits={2} heatmap diverging cellSize="xs" />
                </div>
                <div className="text-2xl text-muted-foreground">·</div>
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">W_K</p>
                  <MatrixView matrix={W_K} digits={2} heatmap diverging cellSize="xs" />
                </div>
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">= K</p>
                  <MatrixView matrix={attn.K} digits={2} heatmap diverging cellSize="xs" />
                </div>
              </div>
              <div className="grid gap-2 sm:grid-cols-[auto_auto_auto_auto] sm:items-center">
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">V = X · W_V</p>
                  <MatrixView matrix={X} digits={2} heatmap diverging cellSize="xs" />
                </div>
                <div className="text-2xl text-muted-foreground">·</div>
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">W_V</p>
                  <MatrixView matrix={W_V} digits={2} heatmap diverging cellSize="xs" />
                </div>
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">= V</p>
                  <MatrixView matrix={attn.V} digits={2} heatmap diverging cellSize="xs" />
                </div>
              </div>

              {/* Scores */}
              <div className="rounded border bg-muted/20 p-3">
                <p className="mb-2 text-xs uppercase text-muted-foreground">
                  S = Q · Kᵀ / √d_k = Q · Kᵀ / √{attn.dK} = Q · Kᵀ / {Math.sqrt(attn.dK).toFixed(3)}
                </p>
                <div className="grid gap-3 sm:grid-cols-3 sm:items-center">
                  <MatrixView matrix={attn.scores} digits={2} heatmap diverging cellSize="sm"
                    highlight={hovered ?? undefined}
                    onCellClick={(i, j) => setHovered([i, j])}
                  />
                  <div className="text-2xl text-muted-foreground">→ softmax</div>
                  <div>
                    <p className="mb-1 text-xs uppercase text-muted-foreground">Weights A (rows sum to 1)</p>
                    <Heatmap
                      matrix={attn.weights}
                      diverging={false}
                      min={0}
                      max={1}
                      cellSize={28}
                      highlight={hovered ?? undefined}
                      onCellClick={(i, j) => setHovered([i, j])}
                      format={(v) => v.toFixed(2)}
                    />
                  </div>
                </div>
                <p className="mt-2 text-[10px] text-muted-foreground">
                  Click a cell to see how it was computed. Each row of A is a probability distribution (sums to 1).
                </p>
              </div>

              {/* Output */}
              <div className="rounded border bg-muted/20 p-3">
                <p className="mb-2 text-xs uppercase text-muted-foreground">Output = A · V</p>
                <div className="grid gap-3 sm:grid-cols-3 sm:items-center">
                  <Heatmap matrix={attn.weights} min={0} max={1} cellSize={24} format={(v) => v.toFixed(2)} highlight={hovered ?? undefined} onCellClick={(i, j) => setHovered([i, j])} />
                  <div className="text-2xl text-muted-foreground">·</div>
                  <MatrixView matrix={attn.V} digits={2} heatmap diverging cellSize="sm" />
                </div>
                <div className="mt-3">
                  <p className="mb-1 text-xs uppercase text-muted-foreground">Output (n × d_v)</p>
                  <MatrixView matrix={attn.output} digits={2} heatmap diverging cellSize="sm" />
                </div>
              </div>

              {/* Cell explanation */}
              {hovered && (
                <div className="rounded border-2 border-primary/40 bg-primary/5 p-3 text-xs">
                  <p className="mb-2 font-medium">
                    Attention weight A[{hovered[0]}, {hovered[1]}]:
                  </p>
                  {(() => {
                    const [i, j] = hovered;
                    const e = explainCell(i, j);
                    return (
                      <div className="space-y-1 font-mono">
                        <p>q_{i} = {fmtVec(e.qi, 3)}</p>
                        <p>k_{j} = {fmtVec(e.kj, 3)}</p>
                        <p>q · k = {e.dot.toFixed(4)}</p>
                        <p>S[{i},{j}] = (q · k) / √{attn.dK} = {e.score.toFixed(4)}</p>
                        <p>A[{i},{j}] = softmax(S[{i},:]) = {e.weight.toFixed(4)}</p>
                        <p className="mt-2 text-muted-foreground">
                          Interpretation: token {i} attends to token {j} with weight {(e.weight * 100).toFixed(1)}%.
                        </p>
                      </div>
                    );
                  })()}
                </div>
              )}
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <ExpandableSection title="What is QKᵀ, intuitively?" variant="why" defaultOpen>
              <p>
                <MathBlock>{`Q K^\\top`}</MathBlock> is a similarity matrix between every query and every key.
                A row of <MathBlock>{`Q`}</MathBlock> is one token's "question"; a column of{" "}
                <MathBlock>{`K^\\top`}</MathBlock> is another token's "answer-key". Their dot product is how
                well-matched they are.
              </p>
              <p className="mt-2">
                Softmax along the row turns these similarities into a probability distribution — how much each token
                should listen to every other token.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Why divide by √d_k?" variant="math" defaultOpen>
              <p>
                If <MathBlock>{`q, k \\in \\mathbb{R}^{d_k}`}</MathBlock> have independent entries with mean 0 and
                variance 1, then{" "}
                <MathBlock>{`q \\cdot k = \\sum_{i=1}^{d_k} q_i k_i`}</MathBlock> has variance <MathBlock>{`d_k`}</MathBlock>,
                so standard deviation <MathBlock>{`\\sqrt{d_k}`}</MathBlock>.
              </p>
              <p className="mt-2">
                For large <MathBlock>{`d_k`}</MathBlock> (e.g. 64), raw dot products grow large, pushing softmax into
                saturated regions where gradients vanish. Dividing by{" "}
                <MathBlock>{`\\sqrt{d_k}`}</MathBlock> keeps the variance ≈ 1, stabilizing training. This is the
                <em> scaled </em> dot-product attention.
              </p>
            </ExpandableSection>

            <ExpandableSection title="What does V do?" variant="intuition">
              <p>
                Q and K decide <em>who attends to whom</em>. V is the actual <em>content</em> that gets passed along.
                The output is a weighted average of V rows, weighted by the attention pattern.
              </p>
              <p className="mt-2">
                Metaphor: Q is "what I'm looking for", K is "what I have to offer", V is "what I actually am".
                A token gathers information from neighbors whose offerings (K) match its needs (Q), weighted by
                similarity, and aggregates their true content (V).
              </p>
            </ExpandableSection>

            <ExpandableSection title="Why softmax along the last axis?" variant="math">
              <p>
                Each row of <MathBlock>{`QK^\\top`}</MathBlock> describes how strongly token <MathBlock>{`i`}</MathBlock>{" "}
                should attend to every token <MathBlock>{`j`}</MathBlock>. We normalize across{" "}
                <MathBlock>{`j`}</MathBlock> so the weights form a convex combination (sum to 1).
              </p>
              <p className="mt-2">
                If we did not softmax, the output <MathBlock>{`AV`}</MathBlock> would scale with the number of tokens
                and lose its interpretation as a weighted average. Softmax gives a proper probability distribution.
              </p>
            </ExpandableSection>
          </div>

          <VivaPanel
            questions={[
              {
                level: "Easy",
                question: "What do Q, K, and V stand for, and what does each one do?",
                hint: "Three roles in information retrieval.",
                answer:
                  "Q = queries (what a token is looking for), K = keys (what a token offers to be matched against), V = values (the actual content carried forward).",
                explanation:
                  "Attention computes for each query a weighted sum of values, where weights are softmax-normalized dot products of the query with all keys.",
              },
              {
                level: "Medium",
                question: "Why is the dot product scaled by 1/√d_k?",
                hint: "Variance of a sum of independent random variables.",
                answer:
                  "Raw q·k has variance d_k. Without scaling, large d_k makes softmax saturate (gradients → 0). Dividing by √d_k keeps variance ≈ 1, stabilizing training.",
                explanation:
                  "This is the 'scaled' in 'scaled dot-product attention'. It is mathematically equivalent to initializing W_Q and W_K with a 1/√d_k factor, but cleaner.",
              },
              {
                level: "Difficult",
                question:
                  "Without softmax, the operation QKᵀV would still be a linear combination of V rows. Why is softmax essential?",
                hint: "Consider normalization and the role of probabilities.",
                answer:
                  "Without softmax, output magnitudes would scale with the number of tokens and the sum of dot products. Softmax (i) normalizes attention weights to a probability distribution (sum 1 per row), (ii) sharpens the distribution so the largest similarities dominate, (iii) provides a nonlinearity that makes the layer expressive.",
                explanation:
                  "QKᵀV without softmax is just X(W_QW_KᵀW_V) — a linear projection of X — and a stack of such layers would collapse to a single linear map. Softmax introduces the nonlinearity that makes transformers powerful.",
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
}
