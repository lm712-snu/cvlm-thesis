"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ChevronLeft, ChevronRight, Eye, EyeOff, Sparkles, Info } from "lucide-react";
import { ImageUploader, type UploadedImage } from "@/components/cvlm/ImageUploader";
import { MatrixView } from "@/components/cvlm/MatrixView";
import { Heatmap } from "@/components/cvlm/Heatmap";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import { useAppStore } from "@/store/app-store";
import {
  randMatrix,
  matmul,
  addBias,
  relu,
  softmaxRows,
  sumRows,
  shape,
  cosineSimilarity,
  type Matrix,
} from "@/lib/math/matrix";
import { selfAttention, crossAttention, sinusoidalPE, argmax } from "@/lib/math/attention";

const CANDIDATE_CAPTIONS = [
  "a cat on the grass",
  "a dog running in the park",
  "a car parked on the street",
  "a tree in the forest",
  "a person walking a path",
];

const VOCAB = ["a", "cat", "dog", "car", "tree", "person", "sits", "runs", "park", "street"];

interface ThesisData {
  H: number;
  W: number;
  P: number;
  N: number;
  patchDim: number;
  d: number;
  V: number;
  patches: Matrix;
  patchEmbed: Matrix;
  pe: Matrix;
  withPos: Matrix;
  vitWeights: Matrix;
  visualTokens: Matrix;
  captionEmbeds: Matrix;
  sims: number[];
  crossWeights: Matrix;
  crossOutput: Matrix;
  lmRep: Matrix;
  logits: Matrix;
  probs: Matrix;
  tokenIds: number[];
  caption: string;
  captionParts: { object: string; action: string; location: string };
  confidence: number;
}

function meanMatrix(m: number[][]): number {
  let s = 0;
  let n = 0;
  for (const r of m) for (const v of r) {
    s += v;
    n++;
  }
  return s / Math.max(1, n);
}

function synthesizeCaption(img: UploadedImage): { object: string; action: string; location: string; caption: string; confidence: number } {
  const avgR = meanMatrix(img.channels[0]);
  const avgG = meanMatrix(img.channels[1]);
  const avgB = meanMatrix(img.channels[2]);
  const brightness = (avgR + avgG + avgB) / 3;

  let object: string;
  if (avgR > avgG && avgR > avgB) object = "dog";
  else if (avgG > avgR && avgG > avgB) object = "tree";
  else if (avgB > avgR && avgB > avgG) object = "car";
  else object = "person";

  const actions = ["sitting on", "running through", "standing by", "lying near"];
  const locations = ["the grass", "a path", "the street", "a fence"];

  const actIdx = Math.floor((brightness / 256) * actions.length) % actions.length;
  const locIdx = Math.floor(((avgR + avgG) / 512) * locations.length) % locations.length;
  const action = actions[Math.max(0, actIdx)];
  const location = locations[Math.max(0, locIdx)];
  const caption = `A ${object} ${action} ${location}.`;
  const confidence = 0.55 + 0.35 * (1 - brightness / 256);
  return { object, action, location, caption, confidence };
}

function buildThesisData(img: UploadedImage, seed = 7): ThesisData {
  const H = img.height;
  const W = img.width;
  const P = Math.min(4, Math.min(H, W));
  const nH = Math.floor(H / P);
  const nW = Math.floor(W / P);
  const N = nH * nW;
  const patchDim = P * P * 3;
  const d = 8;
  const V = VOCAB.length;

  const patches: Matrix = [];
  for (let py = 0; py < nH; py++) {
    for (let px = 0; px < nW; px++) {
      const row: number[] = [];
      for (let i = 0; i < P; i++) {
        for (let j = 0; j < P; j++) {
          row.push(img.channels[0][py * P + i][px * P + j]);
          row.push(img.channels[1][py * P + i][px * P + j]);
          row.push(img.channels[2][py * P + i][px * P + j]);
        }
      }
      patches.push(row);
    }
  }

  const W_E = randMatrix(patchDim, d, -0.04, 0.04, seed + 10);
  const b_E = Array.from({ length: d }, (_, i) => 0.03 * (i - d / 2));
  const patchEmbed = addBias(matmul(patches, W_E), b_E);

  const pe = sinusoidalPE(N, d);
  const withPos = patchEmbed.map((row, i) => row.map((v, j) => v + pe[i][j]));

  const W_Q = randMatrix(d, d, -0.25, 0.25, seed + 20);
  const W_K = randMatrix(d, d, -0.25, 0.25, seed + 21);
  const W_V = randMatrix(d, d, -0.25, 0.25, seed + 22);
  const vit = selfAttention(withPos, W_Q, W_K, W_V);
  const visualTokens = vit.output;

  const captionEmbeds = randMatrix(CANDIDATE_CAPTIONS.length, d, -1, 1, seed + 30);
  const visualMean = sumRows(visualTokens).map((v) => v / N);
  const sims = captionEmbeds.map((c) => cosineSimilarity(c, visualMean));

  const textTokens = randMatrix(2, d, -1, 1, seed + 40);
  const Wc_Q = randMatrix(d, d, -0.25, 0.25, seed + 41);
  const Wc_K = randMatrix(d, d, -0.25, 0.25, seed + 42);
  const Wc_V = randMatrix(d, d, -0.25, 0.25, seed + 43);
  const cross = crossAttention(textTokens, visualTokens, Wc_Q, Wc_K, Wc_V);

  const lmRep = relu(cross.output);

  const W_LM = randMatrix(d, V, -0.4, 0.4, seed + 50);
  const b_LM = Array.from({ length: V }, (_, i) => 0.05 * i);
  const logits = addBias(matmul(lmRep, W_LM), b_LM);
  const probs = softmaxRows(logits);
  const tokenIds = logits.map((row) => argmax(row));

  const cap = synthesizeCaption(img);

  return {
    H, W, P, N, patchDim, d, V,
    patches, patchEmbed, pe, withPos,
    vitWeights: vit.weights, visualTokens,
    captionEmbeds, sims,
    crossWeights: cross.weights, crossOutput: cross.output,
    lmRep, logits, probs, tokenIds,
    caption: cap.caption, captionParts: cap, confidence: cap.confidence,
  };
}

const STAGES = [
  { title: "Image Tensor", desc: "An RGB image as a 3-tensor." },
  { title: "Patch Extraction", desc: "Split into N flattened patches." },
  { title: "Patch Embeddings", desc: "Linear projection to dim d." },
  { title: "Positional Embeddings", desc: "Add sinusoidal PE per token." },
  { title: "Vision Transformer", desc: "Self-attention over patches." },
  { title: "Visual Tokens", desc: "Encoded patch representations." },
  { title: "Image-Text Alignment", desc: "Cosine similarity with captions." },
  { title: "Cross-Modal Interaction", desc: "Text queries attend to image." },
  { title: "Language Model Rep", desc: "Multimodal context vector." },
  { title: "Logits over Vocab", desc: "Linear head produces scores." },
  { title: "Softmax Probabilities", desc: "Normalize to a distribution." },
  { title: "Generated Text", desc: "Decode tokens to a caption." },
] as const;

export function Module21_ThesisDemo() {
  const sharedImage = useAppStore((s) => s.sharedImage);
  const setSharedImage = useAppStore((s) => s.setSharedImage);
  const [stage, setStage] = useState(0);
  const [showMath, setShowMath] = useState(true);

  const data = useMemo(() => (sharedImage ? buildThesisData(sharedImage) : null), [sharedImage]);

  return (
    <div>
      <ModuleHeader
        number={21}
        title="Thesis Demo"
        subtitle="The complete story: how a chain of twelve mathematical operations transforms an image into a sentence of language."
      >
        <MathBlock block>
          {`P(y \\mid X) = \\text{Decode}\\!\\left(\\text{LM}\\!\\left(\\text{CrossAttn}\\!\\left(\\text{txt},\\, \\text{ViT}(\\text{Patch}(X) + PE)\\right)\\right)\\right)`}
        </MathBlock>
      </ModuleHeader>

      <Card className="mb-4">
        <CardContent className="py-4 space-y-3">
          <ImageUploader image={sharedImage} onImage={setSharedImage} maxSize={16} />
          <div className="rounded-md border bg-muted/30 p-2 text-xs flex items-start gap-2">
            <Info className="h-4 w-4 mt-0.5 text-muted-foreground shrink-0" />
            <p className="text-muted-foreground">
              Upload any image (or use the sample). The pipeline below processes it through twelve
              mathematical stages and produces a <em>synthesized</em> caption. The caption generator is a
              deterministic heuristic based on image statistics — it is <strong>not</strong> a real VLM. The
              point is to trace every tensor operation, not to produce state-of-the-art captions.
            </p>
          </div>
        </CardContent>
      </Card>

      {!sharedImage || !data ? (
        <Card>
          <CardContent className="py-12 text-center text-sm text-muted-foreground">
            Upload an image (or use the sample) to run the twelve-stage pipeline.
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Stepper */}
          <Card className="mb-4">
            <CardContent className="py-4">
              <div className="flex flex-wrap items-center gap-2">
                {STAGES.map((s, i) => (
                  <button
                    key={i}
                    onClick={() => setStage(i)}
                    className={`flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs transition-all ${
                      stage === i
                        ? "border-primary bg-primary text-primary-foreground"
                        : i < stage
                        ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                        : "border-border hover:border-primary/40 hover:bg-accent/30"
                    }`}
                  >
                    <span className={`flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-mono ${
                      stage === i ? "bg-primary-foreground text-primary" : ""
                    }`}>
                      {i + 1}
                    </span>
                    <span className="hidden sm:inline">{s.title}</span>
                  </button>
                ))}
              </div>
              <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <Button size="sm" variant="outline" onClick={() => setStage((s) => Math.max(0, s - 1))} disabled={stage === 0}>
                    <ChevronLeft className="h-4 w-4" /> Prev
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => setStage((s) => Math.min(STAGES.length - 1, s + 1))} disabled={stage === STAGES.length - 1}>
                    Next <ChevronRight className="h-4 w-4" />
                  </Button>
                </div>
                <Button size="sm" variant="ghost" onClick={() => setShowMath((s) => !s)}>
                  {showMath ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  {showMath ? "Hide mathematics" : "Show mathematics"}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Stage detail */}
          <StagePanel
            stage={stage}
            data={data}
            image={sharedImage}
            showMath={showMath}
          />

          {/* Final output */}
          <Card className="mt-6 border-2 border-primary/40">
            <CardContent className="py-6 text-center space-y-2">
              <p className="text-xs uppercase tracking-wider text-muted-foreground flex items-center justify-center gap-1">
                <Sparkles className="h-3.5 w-3.5" /> Generated caption
              </p>
              <p className="text-2xl font-mono">&ldquo;{data.caption}&rdquo;</p>
              <p className="text-xs text-muted-foreground">
                Heuristic confidence: <code className="font-mono">{(data.confidence * 100).toFixed(1)}%</code>
                {" — "}decoded from token ids <code className="font-mono">[{data.tokenIds.join(", ")}]</code>
              </p>
            </CardContent>
          </Card>

          <div className="mt-6 grid gap-4 md:grid-cols-2">
            <ExpandableSection title="Why does the caption not match the image exactly?" variant="why" defaultOpen>
              <p>
                This demo computes every mathematical operation explicitly in your browser — there is no
                pretrained network with billions of weights. The final caption is produced by a
                deterministic heuristic that maps image statistics (mean RGB, brightness) to one of a
                small set of objects, actions, and locations.
              </p>
              <p className="mt-2">
                A real VLM (LLaVA, BLIP-2) would replace each stage with a learned neural network: a
                pretrained ViT, a trained projector, and a pretrained language model. The
                <em> mathematical structure</em> is identical — only the parameter count and training
                differ.
              </p>
            </ExpandableSection>

            <ExpandableSection title="How to read each stage" variant="how">
              <p>
                Each stage shows: the input/output dimensions as a <DimBadge dims="dim" variant="intermediate" />{" "}
                badge, the equation as a <MathBlock>{`\\LaTeX`}</MathBlock> block, and a numerical example
                computed from your uploaded image. Walk through stages 1 → 12 in order; each step builds on
                the previous one.
              </p>
              <p className="mt-2">
                Click <em>Show/Hide mathematics</em> to focus on the visualisations only, or expand any
                stage to see the full equation plus the explicit numerical tensor.
              </p>
            </ExpandableSection>
          </div>

          <div className="mt-6">
            <VivaPanel
              questions={[
                {
                  level: "Easy",
                  question: "Name the three broad sub-systems of a VLM (vision, bridge, language) and the role of each.",
                  hint: "What converts pixels to features? What converts features to language-ready tokens? What produces text?",
                  answer:
                    "Vision encoder (ViT or CNN) converts pixels to visual feature vectors. The projector / cross-attention bridge aligns visual features with the LM's embedding space. The language model produces logits over the vocabulary which decode to text.",
                  explanation:
                    "Every stage in this pipeline belongs to one of those three sub-systems. Knowing which one you are debugging speeds up development enormously.",
                },
                {
                  level: "Medium",
                  question: "Why is cosine similarity (rather than dot product) used for image-text alignment in stage 7?",
                  hint: "Think about what each metric measures and what it ignores.",
                  answer:
                    "Cosine similarity measures the angle between vectors, ignoring their magnitudes. Embeddings from different modalities can have very different scales; cosine normalises that away and focuses on direction, which is what semantic similarity is about.",
                  explanation:
                    "In CLIP, both image and text encoders are trained with a contrastive InfoNCE loss on the cosine similarity matrix. The temperature parameter then re-scales similarities before softmax.",
                },
                {
                  level: "Difficult",
                  question:
                    "If you removed the cross-attention stage (stage 8) and simply concatenated the visual tokens to the text embeddings, what would break and why?",
                  hint: "The LM weights expect tokens from a particular distribution.",
                  answer:
                    "The LM's self-attention layers were trained on token embeddings from a particular distribution (e.g. word2vec or learned embeddings in a known range). Concatenating un-aligned visual tokens would inject out-of-distribution vectors, producing garbage activations in the first few layers and likely divergent output.",
                  explanation:
                    "Cross-attention (or a learned projector) is the alignment mechanism that converts visual features into the LM's expected input space. Without it, the pipeline is mathematically valid but semantically meaningless — the LM would treat visual tokens as random noise.",
                },
              ]}
            />
          </div>
        </>
      )}
    </div>
  );
}

interface StagePanelProps {
  stage: number;
  data: ThesisData;
  image: UploadedImage;
  showMath: boolean;
}

function StagePanel({ stage, data, image, showMath }: StagePanelProps) {
  const meta = STAGES[stage];
  const math = STAGE_MATH[stage];
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-mono">
            {stage + 1}
          </span>
          {meta.title}
        </CardTitle>
        <p className="text-sm text-muted-foreground">{meta.desc}</p>
      </CardHeader>
      <CardContent className="space-y-3">
        {showMath && (
          <div className="rounded-md border bg-muted/30 p-3 space-y-2">
            <MathBlock block>{math.latex}</MathBlock>
            <p className="text-xs text-muted-foreground">{math.note}</p>
          </div>
        )}
        <StageVisual stage={stage} data={data} image={image} />
      </CardContent>
    </Card>
  );
}

const STAGE_MATH: { latex: string; note: string }[] = [
  {
    latex: `X \\in \\mathbb{R}^{H \\times W \\times 3}`,
    note: "An image is a 3-tensor of shape (H, W, 3). Each X[i,j,c] is the intensity of color channel c at pixel (i, j).",
  },
  {
    latex: `x_p^{(n)} \\in \\mathbb{R}^{P^2 C},\\ N = \\frac{H}{P} \\cdot \\frac{W}{P}`,
    note: "Split the image into N patches of size P×P. Each patch is flattened to a vector of length P²·C.",
  },
  {
    latex: `z_n = x_p^{(n)} W_E + b_E,\\quad z_n \\in \\mathbb{R}^{d}`,
    note: "A learned linear projection maps each flattened patch to a d-dimensional embedding.",
  },
  {
    latex: `\\tilde z_n = z_n + PE_n,\\quad PE(p, 2i) = \\sin(p / 10000^{2i/d})`,
    note: "Add positional encoding so the encoder knows the patch's location in the image.",
  },
  {
    latex: `\\text{Attn}(\\tilde Z) = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d}}\\right) V`,
    note: "Self-attention lets every patch attend to every other patch, mixing spatial context.",
  },
  {
    latex: `Z_{vis} = \\text{ViT}(\\tilde Z) \\in \\mathbb{R}^{N \\times d}`,
    note: "After one (or more) transformer layers, each patch is a contextualised visual token.",
  },
  {
    latex: `\\text{sim}(Z_{vis}, c_k) = \\cos\\!\\left(\\bar Z_{vis},\\, e_k\\right) = \\frac{\\bar Z_{vis} \\cdot e_k}{\\|\\bar Z_{vis}\\|\\, \\|e_k\\|}`,
    note: "Cosine similarity between the mean-pooled visual token and each candidate caption embedding.",
  },
  {
    latex: `\\text{CrossAttn}(Q_{txt}, K_{img}, V_{img}) = \\text{softmax}\\!\\left(\\frac{Q_{txt} K_{img}^\\top}{\\sqrt{d}}\\right) V_{img}`,
    note: "Text queries attend to image keys/values, fusing language context with visual content.",
  },
  {
    latex: `H_{LM} = \\text{ReLU}\\!\\left(\\text{CrossAttn output}\\right)`,
    note: "Activation introduces non-linearity; result is the multimodal context vector fed to the LM head.",
  },
  {
    latex: `\\text{logits} = H_{LM} W_{LM}^\\top + b_{LM},\\quad \\text{logits} \\in \\mathbb{R}^{T \\times V}`,
    note: "A linear head (weight tying optional) maps hidden states to vocabulary logits.",
  },
  {
    latex: `P(y_t) = \\frac{e^{z_t / \\tau}}{\\sum_{j} e^{z_j / \\tau}}`,
    note: "Softmax with temperature τ normalises logits into a probability distribution over the vocabulary.",
  },
  {
    latex: `y_t = \\arg\\max_i P(y_t = i),\\quad \\hat y = \\text{Decode}(y_1, \\dots, y_T)`,
    note: "Greedy decoding picks the highest-probability token at each step; the IDs are decoded to words.",
  },
];

function StageVisual({ stage, data, image }: { stage: number; data: ThesisData; image: UploadedImage }) {
  switch (stage) {
    case 0:
      return (
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            <DimBadge dims={`H = ${data.H}`} variant="input" />
            <DimBadge dims={`W = ${data.W}`} variant="input" />
            <DimBadge dims="C = 3" variant="input" />
            <DimBadge dims={`total = ${data.H * data.W * 3}`} variant="intermediate" />
          </div>
          <div className="grid gap-3 md:grid-cols-[auto_1fr]">
            <div>
              <p className="mb-1 text-xs text-muted-foreground">Input image</p>
              <img
                src={image.dataUrl}
                alt="uploaded"
                className="rounded border"
                style={{ imageRendering: "pixelated", width: 160, height: 160 * (data.H / data.W) }}
              />
            </div>
            <div className="grid grid-cols-3 gap-2">
              {[
                { name: "R", data: image.channels[0], color: "#ef4444" },
                { name: "G", data: image.channels[1], color: "#22c55e" },
                { name: "B", data: image.channels[2], color: "#0ea5e9" },
              ].map((c) => (
                <div key={c.name}>
                  <p className="mb-1 text-[10px]" style={{ color: c.color }}>{c.name}</p>
                  <Heatmap matrix={c.data} min={0} max={255} cellSize={Math.max(8, Math.floor(160 / Math.max(data.H, data.W)))} format={() => ""} />
                </div>
              ))}
            </div>
          </div>
        </div>
      );
    case 1: {
      const shown = data.patches.map((row) => row.slice(0, 12));
      return (
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2">
            <DimBadge dims={`P = ${data.P}`} variant="input" />
            <DimBadge dims={`N = ${data.N} patches`} variant="output" />
            <DimBadge dims={`P²·C = ${data.patchDim}`} variant="intermediate" />
          </div>
          <p className="text-xs text-muted-foreground">
            Showing first 12 of {data.patchDim} columns per patch.
          </p>
          <MatrixView
            matrix={shown}
            heatmap
            digits={0}
            cellSize="xs"
            rowLabels={Array.from({ length: data.N }, (_, i) => `p${i}`)}
          />
        </div>
      );
    }
    case 2:
      return (
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2">
            <DimBadge dims={`in: ${data.patchDim}`} variant="input" />
            <DimBadge dims={`out: d = ${data.d}`} variant="output" />
          </div>
          <MatrixView
            matrix={data.patchEmbed}
            heatmap
            diverging
            digits={3}
            cellSize="sm"
            rowLabels={Array.from({ length: data.N }, (_, i) => `p${i}`)}
          />
        </div>
      );
    case 3:
      return (
        <div className="grid gap-3 md:grid-cols-2">
          <div>
            <p className="mb-1 text-xs text-muted-foreground">Positional encoding PE</p>
            <MatrixView matrix={data.pe} heatmap diverging digits={3} cellSize="sm" />
          </div>
          <div>
            <p className="mb-1 text-xs text-muted-foreground">Z + PE</p>
            <MatrixView matrix={data.withPos} heatmap diverging digits={3} cellSize="sm" />
          </div>
        </div>
      );
    case 4:
      return (
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground">Attention weights — row n shows how strongly patch n attends to every other patch.</p>
          <Heatmap matrix={data.vitWeights} min={0} max={1} cellSize={28} format={(v) => v.toFixed(2)} />
        </div>
      );
    case 5:
      return (
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2">
            <DimBadge dims={`N = ${data.N}`} variant="input" />
            <DimBadge dims={`d = ${data.d}`} variant="output" />
          </div>
          <MatrixView
            matrix={data.visualTokens}
            heatmap
            diverging
            digits={3}
            cellSize="sm"
            rowLabels={Array.from({ length: data.N }, (_, i) => `t${i}`)}
          />
        </div>
      );
    case 6: {
      const bestIdx = data.sims.indexOf(Math.max(...data.sims));
      return (
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground">Cosine similarity of pooled visual tokens with each candidate caption.</p>
          <div className="space-y-1.5">
            {CANDIDATE_CAPTIONS.map((c, i) => (
              <div key={c} className="flex items-center gap-2">
                <code className="text-xs flex-1 truncate">{c}</code>
                <div className="w-40 h-3 rounded bg-muted overflow-hidden">
                  <div
                    className={`h-full ${i === bestIdx ? "bg-emerald-500" : "bg-primary/60"}`}
                    style={{ width: `${Math.max(0, Math.min(100, (data.sims[i] + 1) / 2 * 100))}%` }}
                  />
                </div>
                <code className="text-xs w-16 text-right">{data.sims[i].toFixed(4)}</code>
                {i === bestIdx && <Badge variant="secondary" className="text-[10px]">best</Badge>}
              </div>
            ))}
          </div>
        </div>
      );
    }
    case 7: {
      const [r, c] = shape(data.crossWeights);
      return (
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2">
            <DimBadge dims={`Q from text: ${r} tokens`} variant="input" />
            <DimBadge dims={`K, V from image: ${data.N} patches`} variant="intermediate" />
            <DimBadge dims={`weights: ${r} × ${c}`} variant="output" />
          </div>
          <p className="text-xs text-muted-foreground">Cross-attention weights (text rows × image columns).</p>
          <Heatmap matrix={data.crossWeights} min={0} max={1} cellSize={24} format={(v) => v.toFixed(2)} />
        </div>
      );
    }
    case 8:
      return (
        <div className="space-y-2">
          <DimBadge dims={`lm rep: ${shape(data.lmRep)[0]} × ${shape(data.lmRep)[1]}`} variant="output" />
          <MatrixView matrix={data.lmRep} heatmap diverging digits={3} cellSize="sm" />
        </div>
      );
    case 9:
      return (
        <div className="space-y-2">
          <DimBadge dims={`logits: ${shape(data.logits)[0]} × ${data.V}`} variant="output" />
          <MatrixView
            matrix={data.logits}
            heatmap
            diverging
            digits={2}
            cellSize="sm"
            rowLabels={Array.from({ length: shape(data.logits)[0] }, (_, i) => `t${i}`)}
            colLabels={VOCAB}
          />
        </div>
      );
    case 10:
      return (
        <div className="space-y-2">
          <DimBadge dims={`probs: ${shape(data.probs)[0]} × ${data.V}`} variant="output" />
          <Heatmap
            matrix={data.probs}
            min={0}
            max={1}
            cellSize={28}
            format={(v) => v.toFixed(2)}
            rowLabels={Array.from({ length: shape(data.probs)[0] }, (_, i) => `t${i}`)}
            colLabels={VOCAB}
          />
        </div>
      );
    case 11: {
      return (
        <div className="space-y-3">
          <div className="rounded-md border-2 border-primary/40 bg-primary/5 p-4 text-center">
            <p className="text-xs uppercase tracking-wider text-muted-foreground">Final caption</p>
            <p className="mt-1 text-xl font-mono">&ldquo;{data.caption}&rdquo;</p>
          </div>
          <div className="flex flex-wrap gap-2">
            {data.tokenIds.map((t, i) => (
              <div key={i} className="rounded-md border bg-card px-3 py-2 text-center">
                <p className="text-[10px] text-muted-foreground">step {i}</p>
                <p className="font-mono text-lg">{t}</p>
                <p className="text-[10px] text-muted-foreground">→ {VOCAB[t]}</p>
              </div>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            Tokens decoded via greedy argmax. The final caption is constructed from the image statistics
            heuristic (object=&ldquo;{data.captionParts.object}&rdquo;, action=&ldquo;{data.captionParts.action}&rdquo;, location=&ldquo;{data.captionParts.location}&rdquo;).
          </p>
        </div>
      );
    }
    default:
      return null;
  }
}
