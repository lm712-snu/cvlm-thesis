"use client";

import { useState, useMemo, useEffect, useRef } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Play, Square, RotateCcw, ChevronDown } from "lucide-react";
import { MatrixView } from "@/components/cvlm/MatrixView";
import { Heatmap } from "@/components/cvlm/Heatmap";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import {
  randMatrix,
  matmul,
  softmaxRows,
  relu,
  addBias,
} from "@/lib/math/matrix";
import { selfAttention, sinusoidalPE, argmax } from "@/lib/math/attention";

const VOCAB = ["a", "dog", "sits", "on", "the", "grass", "in", "park"];

interface PipelineData {
  imageR: number[][];
  imageG: number[][];
  imageB: number[][];
  patches: number[][];
  patchEmbed: number[][];
  positional: number[][];
  withPos: number[][];
  vitWeights: number[][];
  vitScores: number[][];
  vitOutput: number[][];
  visualTokens: number[][];
  projectorOut: number[][];
  multimodal: number[][];
  lmHidden: number[][];
  logits: number[][];
  probs: number[][];
  tokens: number[];
  text: string;
}

function buildPipeline(seed = 7): PipelineData {
  const round = (m: number[][]) => m.map((r) => r.map((v) => Math.round(v)));
  const imageR = round(randMatrix(8, 8, 30, 220, seed));
  const imageG = round(randMatrix(8, 8, 30, 220, seed + 1));
  const imageB = round(randMatrix(8, 8, 30, 220, seed + 2));

  // Patches: 2x2 grid of 4x4x3 patches → 4 rows × 48 cols
  const P = 4;
  const patches: number[][] = [];
  for (let py = 0; py < 2; py++) {
    for (let px = 0; px < 2; px++) {
      const row: number[] = [];
      for (let i = 0; i < P; i++) {
        for (let j = 0; j < P; j++) {
          row.push(imageR[py * P + i][px * P + j]);
          row.push(imageG[py * P + i][px * P + j]);
          row.push(imageB[py * P + i][px * P + j]);
        }
      }
      patches.push(row);
    }
  }

  // Patch embed: Linear 48 → 4
  const W_E = randMatrix(48, 4, -0.05, 0.05, seed + 10);
  const b_E = [0.10, 0.05, -0.08, 0.03];
  const patchEmbed = addBias(matmul(patches, W_E), b_E);

  // Positional: sinusoidal PE (4 x 4)
  const positional = sinusoidalPE(4, 4);
  const withPos = patchEmbed.map((row, i) => row.map((v, j) => v + positional[i][j]));

  // ViT self-attention. d_model = d_k = 4 (single head for clarity).
  const W_Q = randMatrix(4, 4, -0.30, 0.30, seed + 20);
  const W_K = randMatrix(4, 4, -0.30, 0.30, seed + 21);
  const W_V = randMatrix(4, 4, -0.30, 0.30, seed + 22);
  const attn = selfAttention(withPos, W_Q, W_K, W_V);

  const visualTokens = attn.output;

  // Projector: Linear 4 → 4
  const W_P = randMatrix(4, 4, -0.40, 0.40, seed + 30);
  const b_P = [0.05, -0.05, 0.10, 0.0];
  const projectorOut = addBias(matmul(visualTokens, W_P), b_P);

  // Multimodal representation (after activation, simulating fusion)
  const multimodal = relu(projectorOut);

  // LM hidden state: another self-attention pass on multimodal rep
  const lmAttn = selfAttention(multimodal, W_Q, W_K, W_V);
  const lmHidden = lmAttn.output;

  // Logits: Linear 4 → 8 (vocab)
  const W_LM = randMatrix(4, 8, -0.50, 0.50, seed + 40);
  const b_LM = [0.2, -0.1, 0.0, 0.1, -0.05, 0.05, -0.2, 0.1];
  const logits = addBias(matmul(lmHidden, W_LM), b_LM);

  const probs = softmaxRows(logits);
  const tokens = logits.map((row) => argmax(row));
  const text = tokens.map((t) => VOCAB[t]).join(" ");

  return {
    imageR,
    imageG,
    imageB,
    patches,
    patchEmbed,
    positional,
    withPos,
    vitWeights: attn.weights,
    vitScores: attn.scores,
    vitOutput: attn.output,
    visualTokens,
    projectorOut,
    multimodal,
    lmHidden,
    logits,
    probs,
    tokens,
    text,
  };
}

interface StageDef {
  id: number;
  name: string;
  tag: string;
  inputDim: string;
  outputDim: string;
  math: string;
  desc: string;
  modules: string; // prior modules referenced
}

const STAGE_DEFS: StageDef[] = [
  { id: 0, name: "Image", tag: "Input", inputDim: "8 × 8 × 3", outputDim: "8 × 8 × 3", math: "X \\in \\mathbb{R}^{H \\times W \\times 3}", desc: "A raw RGB image is a 3-tensor of shape H × W × 3.", modules: "Module 1" },
  { id: 1, name: "Pixel Tensor", tag: "Tensor", inputDim: "8 × 8 × 3", outputDim: "8 × 8 × 3", math: "X_{i,j,c} \\in [0,255]", desc: "Each pixel is a 3-vector (R, G, B). The tensor is stored channel-wise.", modules: "Module 1" },
  { id: 2, name: "Patches", tag: "Tokenize", inputDim: "8 × 8 × 3", outputDim: "4 × 48", math: "N = \\frac{H}{P}\\frac{W}{P},\\quad x_p \\in \\mathbb{R}^{P^2 C}", desc: "Split image into 2×2 grid of 4×4×3 patches, flatten each to length 48.", modules: "Module 4" },
  { id: 3, name: "Patch Embed", tag: "Linear", inputDim: "4 × 48", outputDim: "4 × 4", math: "Z = X_p W_E + b_E,\\ W_E \\in \\mathbb{R}^{48 \\times 4}", desc: "Linear projection of each flattened patch into d=4 embedding dim.", modules: "Module 5" },
  { id: 4, name: "Positional", tag: "Add PE", inputDim: "4 × 4", outputDim: "4 × 4", math: "\\tilde Z = Z + PE,\\ PE(p,2i)=\\sin(p/10000^{2i/d})", desc: "Add sinusoidal positional encoding so tokens carry position info.", modules: "Module 6" },
  { id: 5, name: "ViT Encoder", tag: "Attention", inputDim: "4 × 4", outputDim: "4 × 4", math: "\\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V", desc: "One self-attention layer lets every patch attend to every other patch.", modules: "Modules 7-9" },
  { id: 6, name: "Visual Tokens", tag: "Output", inputDim: "4 × 4", outputDim: "4 × 4", math: "Z_{vis} = \\text{ViT}(\\tilde Z)", desc: "The encoder produces a sequence of visual tokens, one per patch.", modules: "Module 7" },
  { id: 7, name: "Projector", tag: "Linear", inputDim: "4 × 4", outputDim: "4 × 4", math: "H = Z_{vis} W_P + b_P", desc: "Linear projection from vision dim to LM dim (here kept equal for clarity).", modules: "Module 14" },
  { id: 8, name: "Multimodal Rep", tag: "Fuse", inputDim: "4 × 4", outputDim: "4 × 4", math: "H_{mm} = \\text{ReLU}(H)", desc: "Activation introduces non-linearity; result is the multimodal context.", modules: "Modules 13-15" },
  { id: 9, name: "LM Layers", tag: "Transformer", inputDim: "4 × 4", outputDim: "4 × 4", math: "H_{LM} = \\text{Transformer}(H_{mm})", desc: "Language model transformer layers integrate the multimodal context.", modules: "Modules 10, 15" },
  { id: 10, name: "Logits", tag: "Head", inputDim: "4 × 4", outputDim: "4 × 8", math: "\\text{logits} = H_{LM} W_{LM}^\\top + b_{LM}", desc: "Linear head projects hidden state to vocabulary size (here 8).", modules: "Module 16" },
  { id: 11, name: "Softmax", tag: "Normalize", inputDim: "4 × 8", outputDim: "4 × 8", math: "P(t_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}}", desc: "Softmax converts logits to a probability distribution per position.", modules: "Module 16" },
  { id: 12, name: "Token", tag: "Argmax", inputDim: "4 × 8", outputDim: "4", math: "y_t = \\arg\\max_i P(t_i)", desc: "Greedy decoding selects the highest-probability token id at each step.", modules: "Module 17" },
  { id: 13, name: "Generated Text", tag: "Decode", inputDim: "4 ids", outputDim: "string", math: "y = \\text{Decode}(y_1, \\dots, y_T)", desc: "Token ids are mapped back to words via the vocabulary.", modules: "Module 17" },
];

export function Module18_Pipeline() {
  const data = useMemo(() => buildPipeline(7), []);
  const [openStage, setOpenStage] = useState<number | null>(null);
  const [activeStage, setActiveStage] = useState<number>(-1);
  const [running, setRunning] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!running) return;
    if (activeStage >= STAGE_DEFS.length - 1) {
      // Deferred setState to avoid synchronous state update inside the effect body.
      const stopId = setTimeout(() => setRunning(false), 0);
      return () => clearTimeout(stopId);
    }
    timerRef.current = setTimeout(() => {
      setActiveStage((s) => s + 1);
      setOpenStage((s) => (s === null ? 0 : s + 1));
    }, 700);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [running, activeStage]);

  const runPipeline = () => {
    setActiveStage(0);
    setOpenStage(0);
    setRunning(true);
  };

  const stop = () => {
    setRunning(false);
    if (timerRef.current) clearTimeout(timerRef.current);
  };

  const reset = () => {
    stop();
    setActiveStage(-1);
    setOpenStage(null);
  };

  return (
    <div>
      <ModuleHeader
        number={18}
        title="End-to-End Pipeline"
        subtitle="A single unified demo: from an RGB image all the way to generated text. Click any stage to inspect the math and intermediate tensor."
      >
        <MathBlock block>
          {`P(y_{1:T} \\mid X) = \\prod_{t=1}^{T} P\\!\\left(y_t \\mid y_{<t},\\, \\text{LM}\\!\\left(\\text{Proj}\\!\\left(\\text{ViT}\\!\\left(\\text{Patch}(X) + PE\\right)\\right)\\right)\\right)`}
        </MathBlock>
      </ModuleHeader>

      <Card className="mb-4">
        <CardContent className="flex flex-wrap items-center justify-between gap-3 py-4">
          <div className="text-sm text-muted-foreground">
            Synthetic dimensions: <DimBadge dims="8×8×3 image" variant="input" />{" "}
            → <DimBadge dims="4 patches × 48" variant="intermediate" />{" "}
            → <DimBadge dims="embed d=4" variant="weight" />{" "}
            → <DimBadge dims="vocab V=8" variant="output" />.
          </div>
          <div className="flex gap-2">
            {!running ? (
              <Button size="sm" onClick={runPipeline}>
                <Play className="h-4 w-4" /> Run Pipeline
              </Button>
            ) : (
              <Button size="sm" variant="destructive" onClick={stop}>
                <Square className="h-4 w-4" /> Pause
              </Button>
            )}
            <Button size="sm" variant="outline" onClick={reset}>
              <RotateCcw className="h-4 w-4" /> Reset
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="space-y-2">
        {STAGE_DEFS.map((s, i) => {
          const isActive = activeStage === i;
          const isDone = activeStage > i && activeStage !== -1;
          const isOpen = openStage === i;
          return (
            <div key={s.id}>
              <button
                onClick={() => setOpenStage(isOpen ? null : i)}
                className={`w-full text-left rounded-lg border px-4 py-3 transition-all ${
                  isActive
                    ? "border-primary ring-2 ring-primary/30 bg-primary/5"
                    : isDone
                    ? "border-emerald-500/40 bg-emerald-500/5"
                    : "border-border hover:border-primary/40 hover:bg-accent/30"
                }`}
              >
                <div className="flex items-center gap-3">
                  <span
                    className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full border-2 text-xs font-mono ${
                      isActive
                        ? "border-primary bg-primary text-primary-foreground"
                        : isDone
                        ? "border-emerald-500 bg-emerald-500 text-white"
                        : "border-muted-foreground/30 text-muted-foreground"
                    }`}
                  >
                    {i + 1}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-medium text-sm">{s.name}</span>
                      <Badge variant="secondary" className="text-[10px]">{s.tag}</Badge>
                      {isActive && (
                        <span className="text-[10px] text-primary animate-pulse">running...</span>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5 truncate">{s.desc}</p>
                  </div>
                  <div className="hidden sm:flex items-center gap-1 text-[10px]">
                    <DimBadge dims={s.inputDim} variant="input" />
                    <span className="text-muted-foreground">→</span>
                    <DimBadge dims={s.outputDim} variant="output" />
                  </div>
                  <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${isOpen ? "rotate-180" : ""}`} />
                </div>
              </button>
              {isOpen && (
                <div className="rounded-b-lg border border-t-0 border-border bg-card/40 p-4 space-y-3">
                  <StageDetail index={i} data={data} />
                </div>
              )}
              {i < STAGE_DEFS.length - 1 && (
                <div className="flex justify-center py-0.5">
                  <div className={`h-4 w-0.5 ${isDone ? "bg-emerald-500/60" : "bg-border"}`} />
                </div>
              )}
            </div>
          );
        })}
      </div>

      <Card className="mt-6 border-primary/30">
        <CardContent className="py-4 text-center">
          <p className="text-xs uppercase tracking-wider text-muted-foreground">Final Output</p>
          <p className="mt-1 text-lg font-mono">&ldquo;{data.text}&rdquo;</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Token ids: <code>{data.tokens.join(", ")}</code> from vocab: <code>{VOCAB.join(", ")}</code>
          </p>
        </CardContent>
      </Card>

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <ExpandableSection title="How the modules fit together" variant="how" defaultOpen>
          <p>
            Each stage in this pipeline corresponds to a dedicated module earlier in the course:
            Modules 1-3 cover image tensors and convolution; Modules 4-7 cover patches, embeddings,
            positional encoding, and the ViT encoder; Modules 8-9 cover self- and multi-head attention;
            Modules 11-15 cover alignment and the projector; Modules 16-17 cover logits, softmax, and
            autoregressive generation.
          </p>
          <p className="mt-2">
            This pipeline is the <em>summary</em>: every matrix you have seen separately is here, in
            sequence, with explicit dimensions. Real VLMs (LLaVA, BLIP, Flamingo) follow the same
            skeleton, scaled up by ~1000× in dimension.
          </p>
        </ExpandableSection>

        <ExpandableSection title="Why this pipeline works" variant="why">
          <p>
            The vision encoder turns pixels into a sequence of token vectors that the language model
            can read. The projector is the bridge: it aligns the vision feature space with the LM
            token space. Without it, the LM would receive vectors in a totally different basis and
            produce garbage.
          </p>
          <p className="mt-2">
            Autoregressive decoding factorizes the joint distribution as a product of conditional
            next-token distributions, which is mathematically simple but empirically powerful.
          </p>
        </ExpandableSection>
      </div>

      <div className="mt-6">
        <VivaPanel
          questions={[
            {
              level: "Easy",
              question: "List the 14 stages of the VLM pipeline in order.",
              hint: "Image in, text out — what does each stage do?",
              answer:
                "Image → Pixel Tensor → Patches → Patch Embed → Positional → ViT → Visual Tokens → Projector → Multimodal Rep → LM Layers → Logits → Softmax → Token → Generated Text.",
              explanation:
                "Each stage performs one mathematical transformation. Grouping them this way makes the role of each prior module explicit and helps you debug dimension mismatches.",
            },
            {
              level: "Medium",
              question: "Where exactly does the dimension change from 'vision' to 'language', and why?",
              hint: "Look for the layer that has d_in = vision dim, d_out = LM dim.",
              answer:
                "The Projector. It is a linear map W_P ∈ ℝ^(d_vision × d_lm) that maps each visual token from d_vision to d_lm so the language model can consume it as if it were a token embedding.",
              explanation:
                "Without the projector, the LM's first attention layer would multiply visual tokens by its own weight matrices calibrated for word embeddings, producing out-of-distribution activations.",
            },
            {
              level: "Difficult",
              question:
                "Suppose you replaced greedy argmax with temperature sampling at the Softmax stage. What changes mathematically, and what would you expect to observe in the generated text?",
              hint: "Greedy = argmax; sampling = draw from the categorical distribution.",
              answer:
                "Mathematically, instead of y_t = argmax P(y_t) we draw y_t ~ Categorical(softmax(logits/τ)). Lower τ sharpens the distribution (more deterministic), higher τ flattens it (more diverse, noisier).",
              explanation:
                "In our toy vocab, greedy always returns the same sentence. Sampling would yield different orderings of the same handful of words. In real VLMs, temperature controls the creativity-vs-accuracy trade-off.",
            },
          ]}
        />
      </div>
    </div>
  );
}

function StageDetail({ index, data }: { index: number; data: PipelineData }) {
  const s = STAGE_DEFS[index];
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <DimBadge dims={`in: ${s.inputDim}`} variant="input" />
        <DimBadge dims={`out: ${s.outputDim}`} variant="output" />
        <Badge variant="outline" className="text-[10px]">Refs: {s.modules}</Badge>
      </div>
      <div className="rounded-md border bg-muted/30 p-3 text-sm">
        <MathBlock block>{s.math}</MathBlock>
      </div>
      <StageVisual index={index} data={data} />
    </div>
  );
}

function StageVisual({ index, data }: { index: number; data: PipelineData }) {
  switch (index) {
    case 0:
    case 1: {
      // Show synthesized image as a colored grid + 3 channel heatmaps
      return (
        <div className="grid gap-3 md:grid-cols-[auto_1fr]">
          <div>
            <p className="mb-1 text-xs uppercase text-muted-foreground">Synthesized image (8×8×3)</p>
            <div
              className="grid"
              style={{ gridTemplateColumns: `repeat(8, 28px)` }}
            >
              {Array.from({ length: 8 }).map((_, i) =>
                Array.from({ length: 8 }).map((_, j) => (
                  <div
                    key={`${i}-${j}`}
                    className="border border-background/40"
                    style={{
                      width: 28,
                      height: 28,
                      backgroundColor: `rgb(${data.imageR[i][j]}, ${data.imageG[i][j]}, ${data.imageB[i][j]})`,
                    }}
                    title={`(${i},${j}) RGB(${data.imageR[i][j]}, ${data.imageG[i][j]}, ${data.imageB[i][j]})`}
                  />
                ))
              )}
            </div>
          </div>
          <div className="grid gap-2 grid-cols-3">
            {[
              { name: "R", data: data.imageR, color: "#ef4444" },
              { name: "G", data: data.imageG, color: "#22c55e" },
              { name: "B", data: data.imageB, color: "#0ea5e9" },
            ].map((c) => (
              <div key={c.name}>
                <p className="mb-1 text-[10px] uppercase" style={{ color: c.color }}>{c.name} channel</p>
                <Heatmap matrix={c.data} min={0} max={255} cellSize={16} format={() => ""} />
              </div>
            ))}
          </div>
        </div>
      );
    }
    case 2: {
      // Patches: show 4 patches as colored blocks + flattened matrix
      const patchesForView = data.patches.map((row) => row.slice(0, 12)); // truncate for display
      return (
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground">
            4 patches (2×2 grid), each flattened to length 48 (4·4·3). Showing first 12 of 48 columns.
          </p>
          <MatrixView
            matrix={patchesForView}
            heatmap
            digits={0}
            cellSize="xs"
            rowLabels={["P0", "P1", "P2", "P3"]}
          />
          <p className="text-[10px] text-muted-foreground">shape = [4, 48]</p>
        </div>
      );
    }
    case 3:
      return (
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">Patch embeddings after Linear(48 → 4):</p>
          <MatrixView matrix={data.patchEmbed} heatmap diverging digits={3} cellSize="sm" rowLabels={["P0", "P1", "P2", "P3"]} />
        </div>
      );
    case 4: {
      // Show positional + withPos
      return (
        <div className="grid gap-3 md:grid-cols-2">
          <div>
            <p className="mb-1 text-xs text-muted-foreground">Positional encoding PE (4×4)</p>
            <MatrixView matrix={data.positional} heatmap diverging digits={3} cellSize="sm" />
          </div>
          <div>
            <p className="mb-1 text-xs text-muted-foreground">Z + PE (4×4)</p>
            <MatrixView matrix={data.withPos} heatmap diverging digits={3} cellSize="sm" />
          </div>
        </div>
      );
    }
    case 5: {
      return (
        <div className="space-y-2">
          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <p className="mb-1 text-xs text-muted-foreground">Attention weights (4×4, rows sum to 1)</p>
              <Heatmap matrix={data.vitWeights} min={0} max={1} cellSize={28} format={(v) => v.toFixed(2)} />
            </div>
            <div>
              <p className="mb-1 text-xs text-muted-foreground">Encoder output (4×4)</p>
              <MatrixView matrix={data.vitOutput} heatmap diverging digits={3} cellSize="sm" />
            </div>
          </div>
        </div>
      );
    }
    case 6:
      return <MatrixView matrix={data.visualTokens} heatmap diverging digits={3} cellSize="sm" rowLabels={["t0", "t1", "t2", "t3"]} />;
    case 7:
      return <MatrixView matrix={data.projectorOut} heatmap diverging digits={3} cellSize="sm" />;
    case 8:
      return <MatrixView matrix={data.multimodal} heatmap diverging digits={3} cellSize="sm" />;
    case 9:
      return <MatrixView matrix={data.lmHidden} heatmap diverging digits={3} cellSize="sm" />;
    case 10:
      return (
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">Logits over vocab (4×8):</p>
          <MatrixView
            matrix={data.logits}
            heatmap
            diverging
            digits={2}
            cellSize="sm"
            rowLabels={["t0", "t1", "t2", "t3"]}
            colLabels={VOCAB}
          />
        </div>
      );
    case 11:
      return (
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">Softmax probabilities (4×8, rows sum to 1):</p>
          <Heatmap
            matrix={data.probs}
            min={0}
            max={1}
            cellSize={32}
            format={(v) => v.toFixed(2)}
            rowLabels={["t0", "t1", "t2", "t3"]}
            colLabels={VOCAB}
          />
        </div>
      );
    case 12: {
      return (
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2">
            {data.tokens.map((t, i) => (
              <div key={i} className="rounded-md border bg-card px-3 py-2 text-center">
                <p className="text-[10px] text-muted-foreground">step {i}</p>
                <p className="font-mono text-lg">{t}</p>
                <p className="text-[10px] text-muted-foreground">→ {VOCAB[t]}</p>
              </div>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">Token ids chosen by argmax per row.</p>
        </div>
      );
    }
    case 13: {
      return (
        <div className="space-y-2">
          <div className="rounded-md border-2 border-primary/40 bg-primary/5 p-4 text-center">
            <p className="text-xs uppercase tracking-wider text-muted-foreground">Generated caption</p>
            <p className="mt-1 text-xl font-mono">&ldquo;{data.text}&rdquo;</p>
          </div>
          <p className="text-xs text-muted-foreground">
            Decoded by mapping each token id back to its word in the vocabulary.
          </p>
        </div>
      );
    }
    default:
      return null;
  }
}
