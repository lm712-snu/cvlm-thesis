"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from "recharts";
import { RefreshCw, RotateCcw, Flame } from "lucide-react";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import { softmaxVec, argmax, fmtVec } from "@/lib/math";

const VOCAB = ["dog", "cat", "runs", "jumps", "the", "a", "fast", "slow"] as const;
const V = VOCAB.length;

const DEFAULT_LOGITS = [2.5, 1.0, 0.3, -0.8, -1.2, -2.0, 0.8, -0.4];

const PRESETS: { name: string; logits: number[]; tau: number }[] = [
  { name: "Sharp (dog wins)", logits: [3.5, 0.5, 0.2, -1, -1.5, -2, 0.5, -0.5], tau: 0.5 },
  { name: "Mild (dog ~ fast)", logits: [1.8, 0.4, 0.2, -0.5, -1, -1.5, 1.6, -0.3], tau: 1.0 },
  { name: "Flat (high τ)", logits: [1.0, 0.8, 0.5, 0.2, -0.3, -0.6, 0.9, 0.1], tau: 2.5 },
];

export function Module16_NextToken() {
  const [logits, setLogits] = useState<number[]>(DEFAULT_LOGITS);
  const [tau, setTau] = useState(1.0);

  const setLogit = (i: number, v: number) =>
    setLogits((prev) => prev.map((x, j) => (j === i ? v : x)));

  // Compute step-by-step
  const { scaled, exps, sumExp, probs, amax } = useMemo(() => {
    const scaled = logits.map((z) => z / tau);
    const mx = Math.max(...scaled);
    const exps = scaled.map((x) => Math.exp(x - mx));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((e) => e / sumExp);
    const amax = argmax(probs);
    return { scaled, exps, sumExp, probs, amax };
  }, [logits, tau]);

  const chartData = useMemo(
    () =>
      VOCAB.map((tok, i) => ({
        token: tok,
        prob: probs[i],
        isMax: i === amax,
      })),
    [probs, amax]
  );

  return (
    <div>
      <ModuleHeader
        number={16}
        title="Next-Token Prediction"
        subtitle="The LM head: hidden → logits → softmax → probability → argmax. The math behind every generated word."
      >
        <MathBlock block>
          {`P(\\text{token}_i \\mid \\text{context}) = \\frac{\\exp(z_i / \\tau)}{\\sum_{j=1}^{V} \\exp(z_j / \\tau)}, \\quad \\hat{y} = \\arg\\max_i P_i`}
        </MathBlock>
      </ModuleHeader>

      <div className="grid gap-4 lg:grid-cols-[300px_1fr]">
        {/* Left: controls */}
        <Card className="h-fit">
          <CardHeader>
            <CardTitle className="text-base">Controls</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            {/* Temperature */}
            <div>
              <div className="flex justify-between">
                <Label className="text-xs flex items-center gap-1">
                  <Flame className="h-3 w-3" /> Temperature τ
                </Label>
                <span className="font-mono text-xs">{tau.toFixed(2)}</span>
              </div>
              <Slider value={[tau]} min={0.1} max={3.0} step={0.05}
                onValueChange={([v]) => setTau(v ?? 1.0)} />
              <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
                <span>0.1 (sharp)</span>
                <span>1.0 (orig)</span>
                <span>3.0 (flat)</span>
              </div>
            </div>

            {/* Logit sliders */}
            <div>
              <p className="mb-2 text-xs font-semibold uppercase text-muted-foreground">
                Logits z ∈ ℝ^{V}
              </p>
              <div className="space-y-2">
                {VOCAB.map((tok, i) => (
                  <div key={tok} className="grid grid-cols-[60px_1fr_40px] items-center gap-2">
                    <span className="font-mono text-xs">{tok}</span>
                    <Slider
                      value={[logits[i]]}
                      min={-5}
                      max={5}
                      step={0.1}
                      onValueChange={([v]) => setLogit(i, v ?? 0)}
                    />
                    <span className="text-right font-mono text-[11px] tabular-nums">
                      {logits[i].toFixed(1)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Presets */}
            <div>
              <p className="mb-2 text-xs font-semibold uppercase text-muted-foreground">Presets</p>
              <div className="flex flex-wrap gap-1">
                {PRESETS.map((p) => (
                  <Button
                    key={p.name}
                    size="sm"
                    variant="outline"
                    className="h-auto py-1 text-[10px]"
                    onClick={() => {
                      setLogits(p.logits);
                      setTau(p.tau);
                    }}
                  >
                    {p.name}
                  </Button>
                ))}
              </div>
            </div>

            <div className="flex gap-2">
              <Button size="sm" variant="ghost" onClick={() => { setLogits(DEFAULT_LOGITS); setTau(1.0); }}>
                <RotateCcw className="h-3.5 w-3.5" /> Reset
              </Button>
              <Button size="sm" variant="outline" onClick={() => setLogits(Array.from({ length: V }, () => Math.random() * 8 - 4))}>
                <RefreshCw className="h-3.5 w-3.5" /> Random
              </Button>
            </div>

            <div className="rounded border bg-muted/30 p-2 text-xs space-y-1">
              <div className="flex justify-between">
                <span>Vocab size V</span>
                <DimBadge dims={String(V)} variant="label" />
              </div>
              <div className="flex justify-between">
                <span>Argmax token</span>
                <Badge variant="default" className="text-[10px]">{VOCAB[amax]}</Badge>
              </div>
              <div className="flex justify-between">
                <span>Argmax prob</span>
                <span className="font-mono">{(probs[amax] * 100).toFixed(2)}%</span>
              </div>
              <div className="flex justify-between">
                <span>Sum of probs</span>
                <span className="font-mono">{probs.reduce((a, b) => a + b, 0).toFixed(4)}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Right: visualization + step-by-step */}
        <div className="space-y-4">
          {/* Bar chart */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">
                Probability distribution over the {V}-word vocabulary
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <XAxis dataKey="token" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10 }} />
                    <Tooltip
                      formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
                      labelFormatter={(l) => `token: ${l}`}
                      contentStyle={{ fontSize: 11 }}
                    />
                    <ReferenceLine y={1 / V} stroke="#888" strokeDasharray="3 3"
                      label={{ value: `uniform=${(100 / V).toFixed(1)}%`, fontSize: 9, position: "right" }} />
                    <Bar dataKey="prob" radius={[3, 3, 0, 0]}>
                      {chartData.map((d, i) => (
                        <Cell
                          key={i}
                          fill={d.isMax ? "hsl(var(--primary))" : "hsl(var(--primary) / 0.35)"}
                          stroke={d.isMax ? "hsl(var(--primary))" : "transparent"}
                          strokeWidth={d.isMax ? 2 : 0}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="mt-2 text-xs text-muted-foreground">
                The dashed line is the uniform distribution. The highlighted bar is the argmax
                token <strong>&quot;{VOCAB[amax]}&quot;</strong> with probability{" "}
                <strong>{(probs[amax] * 100).toFixed(2)}%</strong>.
              </p>
            </CardContent>
          </Card>

          {/* Step-by-step computation */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Step-by-step computation</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="rounded border bg-muted/20 p-3">
                <p className="mb-2 text-xs uppercase text-muted-foreground">
                  Step 0 — Logits z (LM head output)
                </p>
                <p className="font-mono text-xs">z = {fmtVec(logits, 2)}</p>
                <p className="mt-1 text-[10px] text-muted-foreground">
                  Computed as <MathBlock>{`z = h W_E^\\top + b_E`}</MathBlock> where h is the LM's
                  final hidden state and W_E is the embedding matrix (Module 13).
                </p>
              </div>

              <div className="rounded border bg-muted/20 p-3">
                <p className="mb-2 text-xs uppercase text-muted-foreground">
                  Step 1 — Scale by temperature: z / τ
                </p>
                <p className="font-mono text-xs">z/τ = {fmtVec(scaled, 3)}</p>
                <p className="mt-1 text-[10px] text-muted-foreground">
                  Low τ → values spread apart (sharp). High τ → values compress toward 0 (flat).
                </p>
              </div>

              <div className="rounded border bg-muted/20 p-3">
                <p className="mb-2 text-xs uppercase text-muted-foreground">
                  Step 2 — Exponentiate: exp(z_i / τ), using the numerically stable trick exp(x − max)
                </p>
                <p className="font-mono text-xs">exp = {fmtVec(exps, 3)}</p>
                <p className="mt-1 text-[10px] text-muted-foreground">
                  max(z/τ) = {Math.max(...scaled).toFixed(3)} is subtracted before exp to prevent
                  overflow; the result is identical because softmax is shift-invariant.
                </p>
              </div>

              <div className="rounded border bg-muted/20 p-3">
                <p className="mb-2 text-xs uppercase text-muted-foreground">
                  Step 3 — Sum: Σ exp(z_j / τ)
                </p>
                <p className="font-mono text-xs">sum = {sumExp.toFixed(4)}</p>
              </div>

              <div className="rounded border-2 border-primary/40 bg-primary/5 p-3">
                <p className="mb-2 text-xs uppercase text-muted-foreground">
                  Step 4 — Normalize: P_i = exp(z_i / τ) / sum
                </p>
                <p className="font-mono text-xs">P = {fmtVec(probs, 4)}</p>
                <p className="mt-1 text-xs">
                  Argmax token: <strong>&quot;{VOCAB[amax]}&quot;</strong> (P ={" "}
                  {(probs[amax] * 100).toFixed(2)}%)
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Expandable sections */}
          <div className="grid gap-3 md:grid-cols-2">
            <ExpandableSection title="Why softmax and not just normalized logits?" variant="why" defaultOpen>
              <p>
                Two issues with naive normalization <MathBlock>{`P_i = z_i / \\sum z_j`}</MathBlock>:
              </p>
              <ol className="mt-2 list-decimal space-y-1 pl-5">
                <li>
                  <strong>Negative logits.</strong> If some z_i &lt; 0, the sum can be ≤ 0, giving
                  negative or undefined probabilities. exp() guarantees positivity.
                </li>
                <li>
                  <strong>No amplification.</strong> Linear normalization keeps the relative gaps
                  the same. exp() amplifies differences — a slightly larger logit becomes a much
                  larger probability.
                </li>
              </ol>
              <p className="mt-2">
                Softmax is also the gradient of cross-entropy loss, so using it makes the gradient
                of training loss clean and well-behaved.
              </p>
            </ExpandableSection>

            <ExpandableSection title="What does temperature actually do?" variant="math" defaultOpen>
              <p>
                Dividing logits by τ before softmax rescales the distribution:
              </p>
              <MathBlock block>{`P_i(\\tau) = \\frac{\\exp(z_i / \\tau)}{\\sum_j \\exp(z_j / \\tau)}`}</MathBlock>
              <ul className="mt-2 list-disc space-y-1 pl-5">
                <li>
                  <strong>τ → 0+</strong>: distribution collapses to a one-hot at argmax. The model
                  becomes deterministic — always picks the top token.
                </li>
                <li>
                  <strong>τ = 1</strong>: original softmax, no rescaling.
                </li>
                <li>
                  <strong>τ → ∞</strong>: distribution tends to uniform — every token is equally
                  likely, output is random.
                </li>
              </ul>
              <p className="mt-2">
                In practice: <strong>τ ≈ 0.7</strong> for factual QA, <strong>τ ≈ 1.0</strong> for
                code, <strong>τ ≈ 0.9–1.2</strong> for creative writing.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Numerical stability: the max-subtraction trick" variant="how">
              <p>
                Direct computation of <MathBlock>{`\\exp(z_i)`}</MathBlock> overflows for large z_i
                (e.g. z = 1000 gives exp = ∞). The trick:
              </p>
              <MathBlock block>{`\\text{softmax}(z)_i = \\frac{\\exp(z_i - m)}{\\sum_j \\exp(z_j - m)}, \\quad m = \\max_j z_j`}</MathBlock>
              <p className="mt-2">
                Because softmax is shift-invariant (adding a constant to all z_i doesn't change the
                output), subtracting m gives an identical result while keeping all exponents ≤ 0,
                hence all exponentials ≤ 1 — no overflow.
              </p>
            </ExpandableSection>

            <ExpandableSection title="From argmax to sampling" variant="intuition">
              <p>
                Argmax (greedy) always picks the single most likely token. This is deterministic but
                boring — the same input always produces the same output, and the model can get stuck
                in loops.
              </p>
              <p className="mt-2">
                Real generation samples from the distribution: draw a token with probability P_i.
                This is why temperature matters: low τ → confident, repetitive output; high τ →
                diverse, sometimes nonsensical output.
              </p>
              <p className="mt-2">
                Module 17 takes this further with top-k and top-p sampling.
              </p>
            </ExpandableSection>
          </div>

          <VivaPanel
            questions={[
              {
                level: "Easy",
                question: "Write the softmax formula and explain what each symbol means.",
                hint: "It's a normalised exponential.",
                answer:
                  "P(token_i | context) = exp(z_i / τ) / Σ_j exp(z_j / τ). z_i is the logit (score) for token i; τ is the temperature; V is the vocabulary size.",
                explanation:
                  "Softmax converts an arbitrary real vector into a probability distribution (non-negative, sums to 1). The temperature τ rescales how sharp the distribution is.",
              },
              {
                level: "Medium",
                question:
                  "If two logits differ by Δ, by how much do their probabilities differ at temperature τ? What does this imply about τ → 0?",
                hint: "Take the ratio of softmax probabilities.",
                answer:
                  "P_i / P_j = exp((z_i − z_j) / τ) = exp(Δ / τ). As τ → 0, this ratio grows exponentially, so the larger-z token dominates and the distribution collapses to one-hot. As τ → ∞, the ratio → 1 and the distribution becomes uniform.",
                explanation:
                  "This is why temperature controls 'sharpness': it scales the effective logit gap. A 0.5 difference at τ=1 is barely visible; at τ=0.1 it becomes a 150:1 ratio.",
              },
              {
                level: "Difficult",
                question:
                  "Argmax always returns the same token for fixed logits. Why might sampling from the softmax distribution produce better captions than always using argmax?",
                hint: "Think about the joint probability of a sequence.",
                answer:
                  "Greedy decoding maximises P(y_t | y_<t, X) at each step, but the globally most likely sequence is not the concatenation of locally most likely tokens — the product can be dominated by a different path. Sampling explores alternative high-probability paths and (i) avoids deterministic repetition loops, (ii) yields natural variation, (iii) sometimes discovers globally better sequences. Combined with top-k/top-p (Module 17), sampling is the standard production method.",
                explanation:
                  "Formally, argmax at each step solves max_t P(y_t | y_<t, X), but the real objective is max_{y_1:T} Π_t P(y_t | y_<t, X). These are different problems — the latter is intractable exactly (exponential in T), which is why heuristic search (beam search) and stochastic sampling exist.",
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
}
