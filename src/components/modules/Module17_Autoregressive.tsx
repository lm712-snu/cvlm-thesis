"use client";

import { useState, useMemo, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from "recharts";
import { Play, StepForward, RotateCcw, Square } from "lucide-react";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import { softmaxVec, argmax, topKMask, topPMask, fmtVec } from "@/lib/math";

/* Hardcoded "fake model" producing logits for each autoregressive step.
 * Vocabulary order matters: index 0 = "a", 1 = "dog", ..., 7 = "<end>".
 * Each row's argmax drives the demo caption "a dog runs fast". */

const VOCAB = ["a", "dog", "cat", "runs", "jumps", "fast", "slow", "<end>"] as const;
const V = VOCAB.length;

const START_TOKEN = "<start>"; // not part of the vocab; represented visually as a placeholder
const IMAGE_TOKEN = "<image>";

const STEP_LOGITS: number[][] = [
  // Step 1: prompt = [image, <start>]  → "a"
  [3.5, 1.0, 0.9, 0.2, 0.2, 0.0, -0.4, 2.6],
  // Step 2: prompt = [image, <start>, "a"]  → "dog"
  [0.4, 3.6, 2.7, 0.3, 0.3, 0.0, -0.3, 0.9],
  // Step 3: prompt = [image, <start>, "a", "dog"]  → "runs"
  [0.3, 0.9, 0.4, 3.5, 2.6, 0.2, -0.2, 0.6],
  // Step 4: prompt = [image, ..., "runs"]  → "fast"
  [0.0, 0.3, 0.2, 0.4, 0.5, 3.4, 2.8, 0.7],
  // Step 5: prompt = [image, ..., "fast"]  → "<end>"
  [0.0, 0.2, 0.1, 0.3, 0.3, 0.5, 0.4, 3.7],
];

const N_STEPS = STEP_LOGITS.length;

type Strategy = "greedy" | "topk" | "topp";

const STRATEGY_LABEL: Record<Strategy, string> = {
  greedy: "Greedy (argmax)",
  topk: "Top-k = 2",
  topp: "Top-p = 0.9",
};

interface StepResult {
  logits: number[];
  probs: number[];          // original softmax probs
  maskedProbs: number[];    // after masking + renormalisation
  inSupport: boolean[];     // whether each token survived masking
  selected: number;         // argmax of maskedProbs
  selectedProb: number;    // maskedProbs[selected]
}

function computeStep(logits: number[], strategy: Strategy): StepResult {
  const probs = softmaxVec(logits, 1.0);
  let maskedProbs: number[];
  let inSupport: boolean[];

  if (strategy === "greedy") {
    maskedProbs = probs.slice();
    inSupport = probs.map(() => true);
  } else if (strategy === "topk") {
    const masked = topKMask(logits, 2);
    inSupport = masked.map((v) => v !== -Infinity);
    const exps = masked.map((v) => (v === -Infinity ? 0 : Math.exp(v - Math.max(...masked.filter((x) => x !== -Infinity)))));
    const s = exps.reduce((a, b) => a + b, 0);
    maskedProbs = exps.map((e) => e / s);
  } else {
    // top-p: re-normalise the kept subset
    inSupport = probs.map(() => false);
    // Determine support via cumulative sorted probs (mirror topPMask)
    const idx = probs.map((p, i) => [p, i] as [number, number]).sort((a, b) => b[0] - a[0]);
    let cum = 0;
    for (const [, i] of idx) {
      inSupport[i] = true;
      cum += probs[i];
      if (cum >= 0.9) break;
    }
    const s = probs.reduce((acc, p, i) => acc + (inSupport[i] ? p : 0), 0);
    maskedProbs = probs.map((p, i) => (inSupport[i] ? p / s : 0));
  }

  const selected = argmax(maskedProbs);
  return { logits, probs, maskedProbs, inSupport, selected, selectedProb: maskedProbs[selected] };
}

export function Module17_Autoregressive() {
  const [strategy, setStrategy] = useState<Strategy>("greedy");
  const [stepIdx, setStepIdx] = useState(0); // 0..N_STEPS inclusive; N_STEPS means "done"
  const [playing, setPlaying] = useState(false);

  // Pre-compute all step results for the chosen strategy
  const steps = useMemo<StepResult[]>(
    () => STEP_LOGITS.map((z) => computeStep(z, strategy)),
    [strategy]
  );

  // Generated tokens: steps[0..stepIdx-1]
  const generated = steps.slice(0, Math.min(stepIdx, N_STEPS)).map((s) => VOCAB[s.selected]);
  const cumProb = steps
    .slice(0, Math.min(stepIdx, N_STEPS))
    .reduce((acc, s) => acc * s.selectedProb, 1);

  // The current visible step's data (or the final summary if done)
  const current = stepIdx < N_STEPS ? steps[stepIdx] : steps[N_STEPS - 1];

  const chartData = useMemo(
    () =>
      VOCAB.map((tok, i) => ({
        token: tok,
        prob: current.maskedProbs[i],
        originalProb: current.probs[i],
        inSupport: current.inSupport[i],
        isSelected: i === current.selected,
      })),
    [current]
  );

  // Advance one step
  const stepNext = () => {
    setStepIdx((s) => {
      if (s >= N_STEPS) return s;
      return s + 1;
    });
  };

  // Auto-play loop using useEffect with cleanup (avoids duplicate timers).
  useEffect(() => {
    if (!playing) return;
    if (stepIdx >= N_STEPS) return;
    const id = setTimeout(() => {
      setStepIdx((s) => Math.min(s + 1, N_STEPS));
    }, 800);
    return () => clearTimeout(id);
  }, [playing, stepIdx]);

  // When stepIdx reaches the end during playback, stop. We do this inside the
  // step-advance callback above to avoid a synchronous setState in render.
  useEffect(() => {
    if (playing && stepIdx >= N_STEPS) {
      const id = setTimeout(() => setPlaying(false), 0);
      return () => clearTimeout(id);
    }
  }, [playing, stepIdx]);

  const reset = () => {
    setStepIdx(0);
    setPlaying(false);
  };

  // Final caption
  const finalTokens = steps.slice(0, N_STEPS).map((s) => VOCAB[s.selected]);
  const finalCaption = finalTokens.filter((t) => t !== "<end>").join(" ");
  const finalCumProb = steps.reduce((acc, s) => acc * s.selectedProb, 1);

  return (
    <div>
      <ModuleHeader
        number={17}
        title="Autoregressive Generation"
        subtitle="Token-by-token generation under the chain-rule factorisation. Watch a fake 'model' emit one word at a time and see how the decoding strategy shapes the support of the distribution."
      >
        <MathBlock block>
          {`P(y_1, \\ldots, y_T \\mid X) = \\prod_{t=1}^{T} P(y_t \\mid y_{<t}, X)`}
        </MathBlock>
      </ModuleHeader>

      <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
        {/* Controls */}
        <Card className="h-fit">
          <CardHeader>
            <CardTitle className="text-base">Decoding</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            <div>
              <Label className="text-xs">Strategy</Label>
              <Select value={strategy} onValueChange={(v) => { setStrategy(v as Strategy); reset(); }}>
                <SelectTrigger className="mt-1 h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="greedy">Greedy (argmax)</SelectItem>
                  <SelectItem value="topk">Top-k = 2</SelectItem>
                  <SelectItem value="topp">Top-p = 0.9</SelectItem>
                </SelectContent>
              </Select>
              <p className="mt-1 text-[10px] text-muted-foreground">
                {STRATEGY_LABEL[strategy]} — all strategies use argmax as the selection rule; the
                mask changes which tokens are eligible.
              </p>
            </div>

            <div className="flex flex-wrap gap-1">
              <Button size="sm" variant="default" onClick={() => { reset(); setPlaying(true); }}>
                <Play className="h-3.5 w-3.5" /> Play
              </Button>
              <Button size="sm" variant="outline" onClick={stepNext} disabled={stepIdx >= N_STEPS}>
                <StepForward className="h-3.5 w-3.5" /> Step
              </Button>
              {playing ? (
                <Button size="sm" variant="ghost" onClick={() => setPlaying(false)}>
                  <Square className="h-3.5 w-3.5" /> Pause
                </Button>
              ) : null}
              <Button size="sm" variant="ghost" onClick={reset}>
                <RotateCcw className="h-3.5 w-3.5" /> Reset
              </Button>
            </div>

            <div className="rounded border bg-muted/30 p-2 text-xs space-y-1">
              <div className="flex justify-between">
                <span>Step</span>
                <span className="font-mono">
                  {Math.min(stepIdx + 1, N_STEPS)} / {N_STEPS}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Vocab V</span>
                <DimBadge dims={String(V)} variant="label" />
              </div>
              <div className="flex justify-between">
                <span>Selected (current)</span>
                <Badge variant="default" className="text-[10px]">{VOCAB[current.selected]}</Badge>
              </div>
              <div className="flex justify-between">
                <span>P(selected | strategy mask)</span>
                <span className="font-mono">{current.selectedProb.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span>Cumulative P</span>
                <span className="font-mono">{stepIdx > 0 ? cumProb.toFixed(6) : "—"}</span>
              </div>
            </div>

            <p className="text-[10px] text-muted-foreground">
              Vocab: {VOCAB.map((t) => `"${t}"`).join(", ")}.
            </p>
          </CardContent>
        </Card>

        {/* Main */}
        <div className="space-y-4">
          {/* Caption preview */}
          <Card className="border-primary/30">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Generated caption (in progress)</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex flex-wrap items-center gap-1">
                <span className="rounded border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-[11px] font-mono">
                  {IMAGE_TOKEN}
                </span>
                <span className="rounded border border-sky-500/40 bg-sky-500/10 px-2 py-0.5 text-[11px] font-mono">
                  {START_TOKEN}
                </span>
                {generated.map((t, i) => (
                  <span
                    key={i}
                    className={`rounded border px-2 py-0.5 text-[11px] font-mono ${
                      t === "<end>"
                        ? "border-rose-500/40 bg-rose-500/10"
                        : "border-violet-500/40 bg-violet-500/10"
                    }`}
                  >
                    {t}
                  </span>
                ))}
                {stepIdx < N_STEPS && (
                  <span className="ml-2 inline-block h-4 w-2 animate-pulse bg-primary align-middle" />
                )}
              </div>
              {stepIdx > 0 && (
                <p className="text-xs text-muted-foreground">
                  Cumulative probability ={" "}
                  <MathBlock>{`\\prod_{t=1}^{${stepIdx}} P(y_t \\mid y_{<t}, X) = ${cumProb.toFixed(6)}`}</MathBlock>
                </p>
              )}
              {stepIdx >= N_STEPS && (
                <div className="mt-2 rounded border-2 border-emerald-500/40 bg-emerald-500/5 p-3">
                  <p className="text-[10px] uppercase text-muted-foreground">Final caption</p>
                  <p className="text-lg font-semibold">&quot;{finalCaption}&quot;</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Total sequence probability under {STRATEGY_LABEL[strategy]}: {finalCumProb.toFixed(6)}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Current step visualization */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">
                Step {Math.min(stepIdx + 1, N_STEPS)} — distribution over next token
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <XAxis dataKey="token" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10 }} />
                    <Tooltip
                      formatter={(v: number, _n: string, props) => {
                        const orig = (props?.payload as { originalProb?: number })?.originalProb ?? 0;
                        return [`${(v * 100).toFixed(2)}% (orig: ${(orig * 100).toFixed(2)}%)`, "prob"];
                      }}
                      labelFormatter={(l) => `token: ${l}`}
                      contentStyle={{ fontSize: 11 }}
                    />
                    <ReferenceLine y={1 / V} stroke="#888" strokeDasharray="3 3"
                      label={{ value: `uniform=${(100 / V).toFixed(1)}%`, fontSize: 9, position: "right" }} />
                    <Bar dataKey="prob" radius={[3, 3, 0, 0]}>
                      {chartData.map((d, i) => (
                        <Cell
                          key={i}
                          fill={
                            d.isSelected
                              ? "hsl(var(--primary))"
                              : d.inSupport
                                ? "hsl(var(--primary) / 0.45)"
                                : "hsl(var(--muted))"
                          }
                          stroke={d.isSelected ? "hsl(var(--primary))" : "transparent"}
                          strokeWidth={d.isSelected ? 2 : 0}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="text-xs text-muted-foreground">
                {strategy === "greedy" && "Greedy: no mask — every token is eligible; argmax is selected."}
                {strategy === "topk" && "Top-k=2: only the two highest-logit tokens survive (others grey); probs renormalised."}
                {strategy === "topp" && "Top-p=0.9: the smallest token set whose cumulative probability ≥ 0.9 (the nucleus); others grey."}
                {" "}The darkest bar is the argmax-selected token; lighter primary bars are eligible alternatives.
              </p>

              {/* Step-by-step numeric panel */}
              <div className="rounded border bg-muted/20 p-3 space-y-1 text-xs font-mono">
                <p className="font-sans font-medium text-foreground">
                  Step {Math.min(stepIdx + 1, N_STEPS)} details
                </p>
                <p>logits   z = {fmtVec(current.logits, 2)}</p>
                <p>softmax P = {fmtVec(current.probs, 4)}</p>
                <p>
                  masked   = {fmtVec(current.maskedProbs, 4)}{" "}
                  <span className="text-muted-foreground">
                    (support: [{current.inSupport.map((b) => (b ? "1" : "0")).join("")}])
                  </span>
                </p>
                <p className="pt-1 text-primary">
                  selected y_{stepIdx + 1} = &quot;{VOCAB[current.selected]}&quot; (P = {current.selectedProb.toFixed(4)})
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Decoding comparison */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Decoding strategies compared</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-xs">
              <table className="w-full">
                <thead>
                  <tr className="border-b text-left text-[10px] uppercase text-muted-foreground">
                    <th className="py-1 pr-2">Strategy</th>
                    <th className="py-1 pr-2">Mask</th>
                    <th className="py-1 pr-2">Selection</th>
                    <th className="py-1">Effect on caption</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  <tr className="border-b">
                    <td className="py-1 pr-2">Greedy</td>
                    <td className="py-1 pr-2">none</td>
                    <td className="py-1 pr-2">argmax(logits)</td>
                    <td className="py-1 font-sans">deterministic; same caption every run</td>
                  </tr>
                  <tr className="border-b">
                    <td className="py-1 pr-2">Top-k = 2</td>
                    <td className="py-1 pr-2">keep top 2</td>
                    <td className="py-1 pr-2">argmax(masked)</td>
                    <td className="py-1 font-sans">same argmax here; in sampling, picks one of the 2</td>
                  </tr>
                  <tr>
                    <td className="py-1 pr-2">Top-p = 0.9</td>
                    <td className="py-1 pr-2">nucleus</td>
                    <td className="py-1 pr-2">argmax(masked)</td>
                    <td className="py-1 font-sans">support varies per step; same argmax here</td>
                  </tr>
                </tbody>
              </table>
              <p className="text-muted-foreground pt-1">
                Note: in this demo all three strategies select the same token at each step (because
                argmax is always in the top-k and in the nucleus). They differ in the support they
                expose to a downstream sampler — see Module 16 for sampling vs argmax.
              </p>
            </CardContent>
          </Card>

          {/* Expandable sections */}
          <div className="grid gap-3 md:grid-cols-2">
            <ExpandableSection title="The chain rule: why generation is sequential" variant="math" defaultOpen>
              <p>
                A joint distribution over a sequence factorises by the chain rule:
              </p>
              <MathBlock block>{`P(y_1, \\ldots, y_T \\mid X) = \\prod_{t=1}^{T} P(y_t \\mid y_{<t}, X)`}</MathBlock>
              <p className="mt-2">
                Each factor P(y_t | y_&lt;t, X) is computed by running the LM forward on the current
                context (image X + all tokens generated so far y_&lt;t) and reading off the softmax
                at the last position. This is why generation is inherently sequential — you cannot
                compute step t+1 without finishing step t.
              </p>
              <p className="mt-2">
                The cumulative probability above is exactly the right-hand side: the product of
                each step's selected-token probability. It is the model's confidence in the entire
                sequence.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Greedy, top-k, top-p — when to use which?" variant="how" defaultOpen>
              <ul className="space-y-2">
                <li>
                  <strong>Greedy.</strong> Cheapest and deterministic. Picked for production code
                  generation, classification, factual QA. Risks: loops, generic outputs.
                </li>
                <li>
                  <strong>Top-k.</strong> Caps the candidate set to a fixed number k. Simple and
                  effective. Risk: the right number of candidates varies by step (some steps are
                  obvious, others genuinely ambiguous) and k cannot adapt.
                </li>
                <li>
                  <strong>Top-p (nucleus).</strong> Keeps the smallest set whose cumulative prob
                  ≥ p. Adapts: when one token dominates, the nucleus is tiny; when many are
                  plausible, it grows. Generally preferred over top-k.
                </li>
              </ul>
              <p className="mt-2">
                All three are typically combined with temperature (Module 16) and a final sampling
                step. Pure argmax over the masked distribution (as in this demo) is rarely used in
                production — the mask is only useful when paired with sampling.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Why does the cumulative probability shrink so fast?" variant="why">
              <p>
                Each P(y_t | y_&lt;t, X) is at most 1, so the product can only decrease. Even with
                per-step probabilities of 0.7, five steps give 0.7⁵ ≈ 0.17.
              </p>
              <p className="mt-2">
                This is why absolute sequence probabilities are not meaningful — most 'good'
                captions have tiny probabilities. We compare <em>relative</em> probabilities
                (perplexity, log-likelihood) instead. The log of the product is the sum of
                log-probs, which is the standard training objective (cross-entropy).
              </p>
            </ExpandableSection>

            <ExpandableSection title="The image stays fixed throughout" variant="intuition">
              <p>
                Note that the image X appears in <em>every</em> factor P(y_t | y_&lt;t, X). The
                visual tokens are part of the context at every generation step — they are not
                'consumed' after step 1.
              </p>
              <p className="mt-2">
                In LLaVA-style architectures the visual tokens are simply prepended to the text
                sequence at every step, and the LM's self-attention re-reads them. In encoder-decoder
                architectures (PaLI, BLIP-2 with T5) the decoder cross-attends to the visual tokens
                at every step (Module 15).
              </p>
              <p className="mt-2">
                Either way: the image is a constant conditioning signal; only the text grows.
              </p>
            </ExpandableSection>
          </div>

          <VivaPanel
            questions={[
              {
                level: "Easy",
                question: "State the chain-rule factorisation of P(y_1, ..., y_T | X) and explain each term.",
                hint: "Joint = product of conditionals.",
                answer:
                  "P(y_1,...,y_T | X) = Π_{t=1}^{T} P(y_t | y_{<t}, X), where y_{<t} = (y_1, ..., y_{t-1}) is the prefix so far and X is the image. Each factor is a softmax over the vocabulary given the current context.",
                explanation:
                  "This is just the chain rule of probability applied to a sequence. It is why generation is inherently autoregressive — step t+1's distribution cannot be computed until step t is finished.",
              },
              {
                level: "Medium",
                question:
                  "Why does top-p (nucleus) sampling adapt its candidate set size per step while top-k does not?",
                hint: "Consider the shape of the probability distribution at different steps.",
                answer:
                  "Top-k keeps a fixed number k of candidates regardless of their probabilities — when one token dominates (e.g. step 5 with P(<end>)=0.95), top-k still keeps k tokens even though most are negligible; when many tokens are plausible (e.g. step 2 with several nouns close in probability), top-k may truncate too aggressively. Top-p keeps the smallest set whose cumulative probability ≥ p, so the set shrinks when the distribution is sharp and grows when it is flat.",
                explanation:
                  "Empirically, top-p produces more natural text than fixed top-k for the same computational budget. Modern VLMs (LLaVA, GPT-4V) expose both parameters, often defaulting to top-p=0.9 or 0.95 with temperature ~0.7.",
              },
              {
                level: "Difficult",
                question:
                  "The cumulative probability of the generated caption is often very small (e.g. 1e-3). Why is this not a problem, and what metric should you use instead to compare models?",
                hint: "Think about the metric used during training.",
                answer:
                  "Sequence probabilities are products of many numbers < 1, so they decay exponentially with T. Absolute values are uninformative. Instead, use the per-token log-probability (i.e. average cross-entropy) or perplexity = exp(average cross-entropy), which normalises out sequence length. For comparing two candidate captions of the same length, the ratio of probabilities (or difference of log-probs) is meaningful even when both are tiny.",
                explanation:
                  "Training minimises the sum of -log P(y_t | y_<t, X) over the ground-truth sequence — i.e. cross-entropy. This is exactly the negative log of the cumulative probability. So minimising cross-entropy is maximising the cumulative probability, and perplexity converts that back to a per-token 'branching factor' that humans can interpret (e.g. perplexity 5 ≈ 'on average, the model was choosing among ~5 equally-likely tokens').",
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
}
