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
import { multiHeadAttention } from "@/lib/math/attention";
import { randMatrix, shape } from "@/lib/math/matrix";

const N = 4; // sequence length
const D_MODEL = 4; // model dimension
const DK = 2; // per-head key/value dimension

export function Module9_MultiHead() {
  const [h, setH] = useState(2);
  const [seed, setSeed] = useState(7);

  // Deterministic configuration: X is shared across all heads.
  // Each head has its own W_Q_i, W_K_i, W_V_i. W_O projects h*d_v back to d_model.
  const { X, W_Qs, W_Ks, W_Vs, W_O, result } = useMemo(() => {
    const X = randMatrix(N, D_MODEL, -2, 2, seed * 13);
    const W_Qs: number[][][] = [];
    const W_Ks: number[][][] = [];
    const W_Vs: number[][][] = [];
    for (let i = 0; i < h; i++) {
      W_Qs.push(randMatrix(D_MODEL, DK, -1, 1, seed * 100 + i * 7 + 1));
      W_Ks.push(randMatrix(D_MODEL, DK, -1, 1, seed * 100 + i * 7 + 2));
      W_Vs.push(randMatrix(D_MODEL, DK, -1, 1, seed * 100 + i * 7 + 3));
    }
    // W_O is (h*d_v x d_model) = (h*DK x D_MODEL)
    const W_O = randMatrix(h * DK, D_MODEL, -1, 1, seed * 1000 + 99);
    const result = multiHeadAttention(X, W_Qs, W_Ks, W_Vs, W_O);
    return { X, W_Qs, W_Ks, W_Vs, W_O, result };
  }, [h, seed]);

  const [concatShape, outputCols] = useMemo(() => {
    const [rows, cols] = shape(result.concat);
    const [, outCols] = shape(result.output);
    return [`${rows} x ${cols}`, outCols] as const;
  }, [result]);

  const tokens = ["t0", "t1", "t2", "t3"];

  return (
    <div>
      <ModuleHeader
        number={9}
        title="Multi-Head Attention"
        subtitle="Several attention heads run in parallel, each learning a different relationship. Their outputs are concatenated and projected back to d_model."
      >
        <MathBlock block>
          {`\\text{MultiHead}(X) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h)\\, W_O,\\quad \\text{head}_i = \\text{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)})`}
        </MathBlock>
      </ModuleHeader>

      <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
        {/* Configuration */}
        <Card className="h-fit">
          <CardHeader><CardTitle className="text-base">Configuration</CardTitle></CardHeader>
          <CardContent className="space-y-4 text-sm">
            <div>
              <div className="flex justify-between">
                <Label className="text-xs">Number of heads h</Label>
                <span className="font-mono text-xs">{h}</span>
              </div>
              <Slider value={[h]} min={1} max={4} step={1} onValueChange={([v]) => setH(v ?? 2)} />
              <p className="mt-1 text-[10px] text-muted-foreground">
                d_model = {D_MODEL} is split evenly, so d_k = d_v = {DK} per head.
              </p>
            </div>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={() => setSeed((s) => s + 1)}>
                <Dice5 className="h-3.5 w-3.5" /> New weights
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setSeed(7)}>
                <RefreshCw className="h-3.5 w-3.5" /> Reset
              </Button>
            </div>
            <div className="rounded border bg-muted/30 p-2 text-xs space-y-1">
              <div className="flex justify-between"><span>Input X</span><DimBadge dims={`${N} x ${D_MODEL}`} variant="input" /></div>
              <div className="flex justify-between"><span>Per-head W_Q/K/V</span><DimBadge dims={`${D_MODEL} x ${DK}`} variant="weight" /></div>
              <div className="flex justify-between"><span>Per-head output</span><DimBadge dims={`${N} x ${DK}`} variant="intermediate" /></div>
              <div className="flex justify-between"><span>Concat</span><DimBadge dims={concatShape} variant="intermediate" /></div>
              <div className="flex justify-between"><span>W_O</span><DimBadge dims={`${h * DK} x ${D_MODEL}`} variant="weight" /></div>
              <div className="flex justify-between"><span>Final output</span><DimBadge dims={`${N} x ${outputCols}`} variant="output" /></div>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-4">
          {/* Shared input */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Shared input sequence X</CardTitle>
            </CardHeader>
            <CardContent>
              <MatrixView matrix={X} digits={2} heatmap diverging cellSize="sm" rowLabels={tokens} colLabels={["d0", "d1", "d2", "d3"]} />
              <p className="mt-1 text-[10px] text-muted-foreground">
                The same X feeds every head; each head projects it through its own learned W_Q, W_K, W_V.
              </p>
            </CardContent>
          </Card>

          {/* Per-head attention */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Per-head attention matrices</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className={`grid gap-3 ${h <= 2 ? "md:grid-cols-2" : "md:grid-cols-2"}`}>
                {result.heads.map((hd, i) => (
                  <div key={i} className="rounded border bg-muted/20 p-3">
                    <div className="mb-2 flex items-center justify-between">
                      <p className="text-xs font-medium uppercase text-muted-foreground">Head {i + 1}</p>
                      <DimBadge dims={`${N} x ${N}`} variant="intermediate" />
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <MatrixView matrix={hd.scores} digits={2} heatmap diverging cellSize="xs" rowLabels={tokens} colLabels={tokens} />
                      <div className="text-xl text-muted-foreground">→</div>
                      <Heatmap
                        matrix={hd.weights}
                        min={0}
                        max={1}
                        cellSize={28}
                        rowLabels={tokens}
                        colLabels={tokens}
                        format={(v) => v.toFixed(2)}
                      />
                    </div>
                    <div className="mt-2">
                      <p className="mb-1 text-[10px] uppercase text-muted-foreground">head output = A · V</p>
                      <MatrixView matrix={hd.output} digits={2} heatmap diverging cellSize="xs" rowLabels={tokens} colLabels={["v0", "v1"]} />
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                Each head's softmax matrix is a probability distribution along its rows (sums to 1).
                Different heads highlight different token-to-token relationships.
              </p>
            </CardContent>
          </Card>

          {/* Concatenation + projection */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Concat + projection back to d_model</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <div className="rounded border bg-muted/20 p-3">
                <p className="mb-2 text-xs uppercase text-muted-foreground">
                  Step 1: Concat head outputs along feature axis → shape {concatShape}
                </p>
                <MatrixView
                  matrix={result.concat}
                  digits={2}
                  heatmap
                  diverging
                  cellSize="xs"
                  rowLabels={tokens}
                  colLabels={Array.from({ length: h * DK }, (_, k) => k < DK ? `h1.v${k}` : `h${Math.floor(k / DK) + 1}.v${k % DK}`)}
                />
                <p className="mt-1 text-[10px] text-muted-foreground">
                  Columns are grouped by head: first {DK} columns come from head 1, next {DK} from head 2, etc.
                </p>
              </div>

              <div className="rounded border bg-muted/20 p-3">
                <p className="mb-2 text-xs uppercase text-muted-foreground">
                  Step 2: Multiply by W_O ({h * DK} x {D_MODEL}) → final output ({N} x {outputCols})
                </p>
                <div className="flex flex-wrap items-center gap-3">
                  <div>
                    <p className="mb-1 text-[10px] uppercase text-muted-foreground">Concat</p>
                    <MatrixView matrix={result.concat} digits={2} heatmap diverging cellSize="xs" />
                  </div>
                  <div className="text-2xl text-muted-foreground">·</div>
                  <div>
                    <p className="mb-1 text-[10px] uppercase text-muted-foreground">W_O</p>
                    <MatrixView matrix={W_O} digits={2} heatmap diverging cellSize="xs" rowLabels={Array.from({ length: h * DK }, (_, k) => `r${k}`)} colLabels={["o0", "o1", "o2", "o3"]} />
                  </div>
                  <div className="text-2xl text-muted-foreground">=</div>
                  <div>
                    <p className="mb-1 text-[10px] uppercase text-muted-foreground">Output</p>
                    <MatrixView matrix={result.output} digits={2} heatmap diverging cellSize="sm" rowLabels={tokens} colLabels={["o0", "o1", "o2", "o3"]} />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <ExpandableSection title="Why multiple heads?" variant="why" defaultOpen>
              <p>
                A single attention head learns <em>one</em> pattern of relationships among tokens. Different heads can
                specialize: one may attend to syntactic dependencies (subject ↔ verb), another to semantic
                relatedness, another to local window context.
              </p>
              <p className="mt-2">
                Mathematically, splitting d_model into h subspaces gives each head a lower-dimensional view
                (d_k = d_model / h) but with its own projection. This is a trade-off: more heads means finer
                specialization but smaller per-head capacity.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Why project back with W_O?" variant="math" defaultOpen>
              <p>
                The concatenated matrix has shape <MathBlock>{`(n \\times h\\,d_v)`}</MathBlock>, which does not match
                <MathBlock>{`d_{\\text{model}}`}</MathBlock> unless <MathBlock>{`h\\,d_v = d_{\\text{model}}`}</MathBlock>.
                Even when shapes match, the linear projection <MathBlock>{`W_O`}</MathBlock> mixes information across heads,
                letting the model combine each head's distinct view into a unified representation.
              </p>
              <p className="mt-2">
                Without <MathBlock>{`W_O`}</MathBlock>, the residual-stream connection (used in transformer blocks) would
                be ill-defined: <MathBlock>{`X + \\text{MultiHead}(X)`}</MathBlock> only makes sense when both have the
                same dimension.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Does changing h change the parameter count?" variant="math">
              <p>
                Surprisingly, no — at least in the standard design where <MathBlock>{`d_k = d_v = d_{\\text{model}}/h`}</MathBlock>.
                The Q, K, V projections together have parameters <MathBlock>{`3 \\cdot d_{\\text{model}} \\cdot d_k \\cdot h = 3\\,d_{\\text{model}}^2`}</MathBlock>,
                independent of <MathBlock>{`h`}</MathBlock>. Plus <MathBlock>{`W_O`}</MathBlock> has{" "}
                <MathBlock>{`h\\,d_v \\cdot d_{\\text{model}} = d_{\\text{model}}^2`}</MathBlock> parameters. So the total is
                <MathBlock>{`4\\,d_{\\text{model}}^2`}</MathBlock> regardless of <MathBlock>{`h`}</MathBlock>.
              </p>
              <p className="mt-2">
                This means <MathBlock>{`h`}</MathBlock> is a pure expressivity trade-off, not a capacity knob. In this toy demo,
                however, d_k is fixed at {DK} per head, so increasing <MathBlock>{`h`}</MathBlock> does grow the concat width.
              </p>
            </ExpandableSection>

            <ExpandableSection title="What changes when I crank h up?" variant="intuition">
              <p>
                With h = 1, you have vanilla single-head attention. With h = 4 (and a fixed d_model), each head sees
                only a 1-dimensional subspace — it can still attend to a single pattern per head, but each row of Q, K
                is a scalar. Increasing h beyond d_model is meaningless (you'd get d_k = 0).
              </p>
              <p className="mt-2">
                Typical transformer configs use h = 8 or 12 with d_model = 512 or 768, keeping d_k around 64.
              </p>
            </ExpandableSection>
          </div>

          <VivaPanel
            questions={[
              {
                level: "Easy",
                question: "What does each attention head compute, and how are their outputs combined?",
                hint: "Two keywords: Concat and projection.",
                answer:
                  "Each head computes an independent scaled dot-product attention with its own W_Q, W_K, W_V. The head outputs are concatenated along the feature axis and multiplied by a learned projection matrix W_O to give the final output.",
                explanation:
                  "Concat(head_1, ..., head_h) has shape (n, h*d_v). W_O is (h*d_v, d_model), so the product restores d_model and mixes information across heads.",
              },
              {
                level: "Medium",
                question: "If d_model is fixed, why is the total parameter count of multi-head attention roughly independent of h?",
                hint: "Consider what happens to d_k when h grows.",
                answer:
                  "When d_k = d_v = d_model / h, the four projection matrices together have 4 * d_model^2 parameters regardless of h. Increasing h subdivides d_model into smaller per-head subspaces rather than adding capacity.",
                explanation:
                  "W_Q, W_K, W_V combined: 3 * d_model * (d_model/h) * h = 3 d_model^2. W_O: (h*d_v) * d_model = d_model^2. Total = 4 d_model^2. (In this demo d_k is fixed, so the concat width grows with h — that is non-standard.)",
              },
              {
                level: "Difficult",
                question:
                  "Multi-head attention with h heads is NOT the same as h independent single-head attentions applied separately and then averaged. Why?",
                hint: "Think about the dimensionality of each head and the role of W_O.",
                answer:
                  "Each head operates on a lower-dimensional projection (d_k = d_model/h) of the same input, learning a specialized subspace view. Concatenating and applying W_O mixes information across heads, producing interactions that simply averaging h full-dimensional attentions cannot. W_O is what makes the heads a coupled system rather than an ensemble.",
                explanation:
                  "If you ran h full-dimensional attentions and averaged, you'd lose the subspace specialization and the cross-head mixing. The architecture explicitly trades per-head capacity for diversity, then recombines via W_O. This is also why pruning individual heads can damage performance: the remaining heads were not trained to cover the pruned heads' subspace.",
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
}
