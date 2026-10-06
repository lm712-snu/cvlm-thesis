"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import { ArrowRight, Info } from "lucide-react";
import { cn } from "@/lib/utils";

/* ------------------------------------------------------------------ */
/*  Type definitions                                                   */
/* ------------------------------------------------------------------ */

type StageKind =
  | "image"
  | "encoder"
  | "tokens"
  | "projector"
  | "lm"
  | "logits"
  | "softmax"
  | "text";

interface Stage {
  id: StageKind;
  label: string;
  sublabel?: string;
  dim: string;
  variant: "input" | "weight" | "output" | "intermediate" | "label";
  math: string;
  explanation: string;
}

interface ArchDef {
  id: string;
  short: string;
  name: string;
  year: string;
  oneLiner: string;
  stages: Stage[];
  highlights: string[];
}

/* ------------------------------------------------------------------ */
/*  Stage color tokens (kept consistent with DimBadge variants)       */
/* ------------------------------------------------------------------ */

const STAGE_COLORS: Record<StageKind, string> = {
  image: "border-emerald-500/50 bg-emerald-500/10 hover:bg-emerald-500/15",
  encoder: "border-sky-500/50 bg-sky-500/10 hover:bg-sky-500/15",
  tokens: "border-violet-500/50 bg-violet-500/10 hover:bg-violet-500/15",
  projector: "border-amber-500/50 bg-amber-500/10 hover:bg-amber-500/15",
  lm: "border-fuchsia-500/50 bg-fuchsia-500/10 hover:bg-fuchsia-500/15",
  logits: "border-cyan-500/50 bg-cyan-500/10 hover:bg-cyan-500/15",
  softmax: "border-rose-500/50 bg-rose-500/10 hover:bg-rose-500/15",
  text: "border-emerald-500/50 bg-emerald-500/10 hover:bg-emerald-500/15",
};

/* ------------------------------------------------------------------ */
/*  The five architecture definitions                                  */
/* ------------------------------------------------------------------ */

const ARCHES: ArchDef[] = [
  {
    id: "resnet",
    short: "ResNet + LM",
    name: "CNN-based Vision Encoder (ResNet)",
    year: "2014–2017",
    oneLiner:
      "A CNN (ResNet) extracts a single global feature vector. The LM consumes one image token.",
    stages: [
      {
        id: "image",
        label: "IMAGE",
        sublabel: "RGB",
        dim: "H × W × 3",
        variant: "input",
        math: "X \\in \\mathbb{R}^{H \\times W \\times 3}",
        explanation:
          "The raw RGB image tensor. For ResNet-50 typically 224×224×3 after resizing and normalization.",
      },
      {
        id: "encoder",
        label: "VISION ENCODER",
        sublabel: "ResNet-50",
        dim: "CNN",
        variant: "weight",
        math: "F_{\\text{cnn}}(X) = \\text{global-pool}\\!\\left(\\text{ConvBlocks}(X)\\right)",
        explanation:
          "A stack of convolutional blocks (residual) shrinks spatial resolution while increasing channel depth, then a global average pool collapses H'×W' into a single vector.",
      },
      {
        id: "tokens",
        label: "VISUAL TOKENS",
        sublabel: "1 token",
        dim: "1 × d_v",
        variant: "intermediate",
        math: "z_v \\in \\mathbb{R}^{1 \\times d_v}",
        explanation:
          "After global pooling the image is described by a single feature vector z_v. Early VQA models (e.g. CNN+LSTM) used exactly this representation.",
      },
      {
        id: "projector",
        label: "PROJECTOR",
        sublabel: "Linear",
        dim: "d_v × d_LM",
        variant: "weight",
        math: "z_p = z_v W_P + b",
        explanation:
          "A single linear layer maps the vision feature into the LM's embedding space.",
      },
      {
        id: "lm",
        label: "LANGUAGE MODEL",
        sublabel: "LSTM / Transformer",
        dim: "d_LM",
        variant: "weight",
        math: "h_t = \\text{LM}(\\text{text}_{<t}, z_p)",
        explanation:
          "The conditioned language model (LSTM or small transformer) produces hidden states h_t autoregressively, attending over z_p at every step.",
      },
      {
        id: "logits",
        label: "LOGITS",
        dim: "|V|",
        variant: "intermediate",
        math: "z^{(t)} = h_t W_E^\\top + b_E",
        explanation:
          "A linear projection (the LM head) maps the hidden state to scores over the vocabulary.",
      },
      {
        id: "softmax",
        label: "SOFTMAX",
        dim: "|V|",
        variant: "intermediate",
        math: "P(y_t \\mid y_{<t}, X) = \\frac{\\exp(z^{(t)}_i)}{\\sum_j \\exp(z^{(t)}_j)}",
        explanation:
          "Softmax turns logits into a probability distribution over the vocabulary.",
      },
      {
        id: "text",
        label: "TEXT",
        sublabel: "caption",
        dim: "T tokens",
        variant: "output",
        math: "y_{1:T} = \\arg\\max_{y_t} P(y_t \\mid y_{<t}, X)",
        explanation:
          "Greedy or sampled tokens, decoded autoregressively one at a time (see Modules 16-17).",
      },
    ],
    highlights: [
      "Single visual token — fine-grained spatial detail is lost after global pooling.",
      "Used in early VQA / image-captioning systems before transformers took over.",
      "Not the architecture used by GPT-4V or LLaVA; included for historical context.",
    ],
  },
  {
    id: "vit",
    short: "ViT + LM",
    name: "Vision Transformer (ViT)",
    year: "2020",
    oneLiner:
      "Image is split into patches and processed as a sequence by a transformer — producing N visual tokens.",
    stages: [
      {
        id: "image",
        label: "IMAGE",
        sublabel: "RGB",
        dim: "224 × 224 × 3",
        variant: "input",
        math: "X \\in \\mathbb{R}^{224 \\times 224 \\times 3}",
        explanation: "Pre-processed RGB image, typically 224×224 with ImageNet normalization.",
      },
      {
        id: "encoder",
        label: "VISION ENCODER",
        sublabel: "ViT",
        dim: "Transformer",
        variant: "weight",
        math: "Z = \\text{ViT}(\\text{patches}(X)),\\quad Z \\in \\mathbb{R}^{N \\times d_v}",
        explanation:
          "The image is split into N = (224/P)² patches (e.g. P=16 → N=196), each flattened and linearly projected to a d_v-dim token, then processed by a transformer encoder.",
      },
      {
        id: "tokens",
        label: "VISUAL TOKENS",
        sublabel: "N=196 tokens",
        dim: "196 × d_v",
        variant: "intermediate",
        math: "Z = (z_1, \\ldots, z_N),\\ z_i \\in \\mathbb{R}^{d_v}",
        explanation:
          "Each patch becomes one token; the [CLS] token may be prepended. Spatial structure is preserved as a sequence.",
      },
      {
        id: "projector",
        label: "PROJECTOR",
        sublabel: "Linear / MLP",
        dim: "d_v × d_LM",
        variant: "weight",
        math: "Z_p = Z W_P + b",
        explanation:
          "Each visual token is projected independently into the LM's embedding space.",
      },
      {
        id: "lm",
        label: "LANGUAGE MODEL",
        sublabel: "Transformer-decoder",
        dim: "d_LM",
        variant: "weight",
        math: "h_t = \\text{LM}(\\text{text}_{<t}, Z_p)",
        explanation:
          "The decoder attends over its own past text and over the projected visual tokens via cross-attention (or full self-attention if prefixes are concatenated).",
      },
      {
        id: "logits",
        label: "LOGITS",
        dim: "|V|",
        variant: "intermediate",
        math: "z^{(t)} = h_t W_E^\\top + b_E",
        explanation: "LM head produces vocabulary scores for the next token.",
      },
      {
        id: "softmax",
        label: "SOFTMAX",
        dim: "|V|",
        variant: "intermediate",
        math: "P(y_t \\mid y_{<t}, X) = \\text{softmax}(z^{(t)})",
        explanation: "Softmax over logits gives next-token probabilities.",
      },
      {
        id: "text",
        label: "TEXT",
        sublabel: "caption",
        dim: "T tokens",
        variant: "output",
        math: "y_{1:T} = \\prod_{t=1}^{T} P(y_t \\mid y_{<t}, X)",
        explanation: "Autoregressively decoded tokens form the output caption.",
      },
    ],
    highlights: [
      "Multiple visual tokens (196) preserve spatial detail — unlike the CNN approach.",
      "Each token is a 'word' in the image's own vocabulary.",
      "Foundation for PaLI, ViLT, and (with CLIP pre-training) LLaVA.",
    ],
  },
  {
    id: "clip",
    short: "CLIP",
    name: "Dual-Encoder (CLIP)",
    year: "2021",
    oneLiner:
      "Two parallel encoders map image and text into a shared embedding space via contrastive learning. Not a captioner on its own — used as a vision backbone.",
    stages: [
      {
        id: "image",
        label: "IMAGE",
        sublabel: "RGB",
        dim: "224 × 224 × 3",
        variant: "input",
        math: "X \\in \\mathbb{R}^{224 \\times 224 \\times 3}",
        explanation: "Input image, processed independently from any text.",
      },
      {
        id: "encoder",
        label: "VISION ENCODER",
        sublabel: "ViT (CLIP)",
        dim: "Transformer",
        variant: "weight",
        math: "v = \\text{pool}(\\text{ViT}(X)) \\in \\mathbb{R}^{d}",
        explanation:
          "A ViT encodes the image; the [CLS] token (or global pool) gives one image embedding v of dimension d.",
      },
      {
        id: "tokens",
        label: "VISUAL TOKENS",
        sublabel: "1 embedding",
        dim: "1 × d",
        variant: "intermediate",
        math: "v \\in \\mathbb{R}^d",
        explanation:
          "CLIP uses a single pooled embedding for contrastive matching — there is no token sequence in the original model. (Note: the underlying ViT produces a patch sequence CLIP simply pools over.)",
      },
      {
        id: "projector",
        label: "PROJECTOR",
        sublabel: "Shared head",
        dim: "d",
        variant: "weight",
        math: "\\tilde v = v / \\|v\\|",
        explanation:
          "Both image and text embeddings are L2-normalized so their dot product is a cosine similarity in [-1, 1].",
      },
      {
        id: "lm",
        label: "LANGUAGE MODEL",
        sublabel: "Text encoder",
        dim: "d",
        variant: "weight",
        math: "u = \\text{pool}(\\text{Transformer}(\\text{text}))",
        explanation:
          "CLIP's text encoder is NOT a generative LM — it is a transformer that produces a single pooled text embedding u for matching. This is what makes CLIP a dual-encoder, not a VLM captioner.",
      },
      {
        id: "logits",
        label: "SIMILARITY",
        dim: "B × B",
        variant: "intermediate",
        math: "S_{ij} = \\tilde v_i \\cdot \\tilde u_j",
        explanation:
          "For a batch of B image-text pairs, compute the B×B similarity matrix. The diagonal holds positive pairs; off-diagonal are negatives.",
      },
      {
        id: "softmax",
        label: "SOFTMAX",
        dim: "B × B",
        variant: "intermediate",
        math: "\\mathcal{L}_{\\text{InfoNCE}} = -\\frac{1}{B}\\sum_i \\log \\frac{\\exp(S_{ii}/\\tau)}{\\sum_j \\exp(S_{ij}/\\tau)}",
        explanation:
          "Bidirectional InfoNCE loss: each row and each column is a softmax. τ is a learned temperature. Maximizing diagonal probability pulls matched pairs together.",
      },
      {
        id: "text",
        label: "MATCHED TEXT",
        sublabel: "retrieval",
        dim: "1 text",
        variant: "output",
        math: "y^* = \\arg\\max_j S_{ij}",
        explanation:
          "At inference, CLIP retrieves the nearest text embedding for a query image (zero-shot classification). It does not generate text — that requires a decoder LM, as in BLIP-2 / LLaVA.",
      },
    ],
    highlights: [
      "Trained by contrastive learning — no token generation, only matching.",
      "Becomes the vision encoder backbone for BLIP-2 and LLaVA.",
      "Shared embedding space is what makes zero-shot classification work.",
    ],
  },
  {
    id: "blip2",
    short: "BLIP-2",
    name: "Encoder–Decoder with Q-Former (BLIP-2)",
    year: "2023",
    oneLiner:
      "A frozen image encoder + a frozen LLM, bridged by a small learnable Q-Former that compresses image features into a fixed number of query tokens.",
    stages: [
      {
        id: "image",
        label: "IMAGE",
        sublabel: "RGB",
        dim: "224 × 224 × 3",
        variant: "input",
        math: "X \\in \\mathbb{R}^{224 \\times 224 \\times 3}",
        explanation: "Pre-processed RGB image.",
      },
      {
        id: "encoder",
        label: "VISION ENCODER",
        sublabel: "ViT (frozen)",
        dim: "Transformer",
        variant: "weight",
        math: "F = \\text{ViT}(X) \\in \\mathbb{R}^{N \\times d_v}",
        explanation:
          "A frozen, pretrained ViT (typically EVA-CLIP) extracts N patch features. Weights are NOT updated during BLIP-2 training.",
      },
      {
        id: "tokens",
        label: "IMAGE FEATURES",
        sublabel: "N=256 patches",
        dim: "256 × d_v",
        variant: "intermediate",
        math: "F = (f_1, \\ldots, f_N)",
        explanation:
          "The raw patch features. Too many to feed directly into a frozen LLM efficiently — BLIP-2 compresses them.",
      },
      {
        id: "projector",
        label: "Q-FORMER",
        sublabel: "32 learnable queries",
        dim: "M=32 queries",
        variant: "weight",
        math: "Q = \\text{CrossAttn}(Q_0, F),\\quad Q \\in \\mathbb{R}^{M \\times d_q}",
        explanation:
          "A small transformer with M learnable query tokens (M=32) cross-attends to the N image features. The output is M compact query embeddings that summarize the image — a form of learned resampling.",
      },
      {
        id: "lm",
        label: "LANGUAGE MODEL",
        sublabel: "Opt / Flan-T5 (frozen)",
        dim: "d_LM",
        variant: "weight",
        math: "h_t = \\text{LM}(\\text{text}_{<t}, Q W_p)",
        explanation:
          "The query embeddings are linearly projected and prepended as soft prompts to a frozen LLM (Opt, Flan-T5). The LM is never fine-tuned; only the Q-Former learns.",
      },
      {
        id: "logits",
        label: "LOGITS",
        dim: "|V|",
        variant: "intermediate",
        math: "z^{(t)} = h_t W_E^\\top + b_E",
        explanation: "LM head produces next-token scores.",
      },
      {
        id: "softmax",
        label: "SOFTMAX",
        dim: "|V|",
        variant: "intermediate",
        math: "P(y_t \\mid y_{<t}, X) = \\text{softmax}(z^{(t)})",
        explanation: "Softmax over logits gives the next-token distribution.",
      },
      {
        id: "text",
        label: "TEXT",
        sublabel: "caption / VQA",
        dim: "T tokens",
        variant: "output",
        math: "y_{1:T} = \\arg\\max_{y_t} P(y_t \\mid y_{<t}, X)",
        explanation: "Generated answer or caption, decoded autoregressively.",
      },
    ],
    highlights: [
      "Q-Former compresses 256 patch features into 32 query tokens — efficient.",
      "Both ViT and LLM stay frozen; only the Q-Former is trained — parameter-efficient.",
      "Same architecture supports captioning, VQA, and image-text retrieval.",
    ],
  },
  {
    id: "llava",
    short: "LLaVA",
    name: "Multimodal LLM (LLaVA)",
    year: "2023",
    oneLiner:
      "A frozen CLIP-ViT + a simple 2-layer MLP projector + an autoregressive LLM (Vicuna), jointly instruction-tuned end-to-end.",
    stages: [
      {
        id: "image",
        label: "IMAGE",
        sublabel: "RGB",
        dim: "336 × 336 × 3",
        variant: "input",
        math: "X \\in \\mathbb{R}^{336 \\times 336 \\times 3}",
        explanation: "Resized RGB image (336×336 for CLIP-ViT-L/14).",
      },
      {
        id: "encoder",
        label: "VISION ENCODER",
        sublabel: "CLIP-ViT (frozen)",
        dim: "Transformer",
        variant: "weight",
        math: "Z = \\text{CLIP-ViT}(X) \\in \\mathbb{R}^{N \\times d_v}",
        explanation:
          "The ViT from CLIP (no [CLS] pooling at the output — the patch features are kept). Frozen during LLaVA training in stage 1.",
      },
      {
        id: "tokens",
        label: "VISUAL TOKENS",
        sublabel: "N=576 patches",
        dim: "576 × d_v",
        variant: "intermediate",
        math: "Z = (z_1, \\ldots, z_N)",
        explanation:
          "All patch tokens are passed through — no compression like BLIP-2's Q-Former. N = (336/14)² = 576 for CLIP-ViT-L/14.",
      },
      {
        id: "projector",
        label: "PROJECTOR",
        sublabel: "MLP (2 layers)",
        dim: "d_v × d_LM",
        variant: "weight",
        math: "Z_p = \\text{GELU}(Z W_1 + b_1) W_2 + b_2",
        explanation:
          "A 2-layer MLP (with GELU) maps each visual token into the LM's embedding space. Trained end-to-end with the LLM in stage 2.",
      },
      {
        id: "lm",
        label: "LANGUAGE MODEL",
        sublabel: "Vicuna / LLaMA",
        dim: "d_LM",
        variant: "weight",
        math: "h_t = \\text{LM}(Z_p, \\text{text}_{<t})",
        explanation:
          "The projected visual tokens are concatenated with the text-token embeddings as a prefix and consumed by the LM's self-attention — there is no separate cross-attention layer.",
      },
      {
        id: "logits",
        label: "LOGITS",
        dim: "|V|",
        variant: "intermediate",
        math: "z^{(t)} = h_t W_E^\\top + b_E",
        explanation: "LM head produces next-token logits.",
      },
      {
        id: "softmax",
        label: "SOFTMAX",
        dim: "|V|",
        variant: "intermediate",
        math: "P(y_t \\mid y_{<t}, X) = \\text{softmax}(z^{(t)})",
        explanation: "Softmax over logits gives the next-token distribution.",
      },
      {
        id: "text",
        label: "TEXT",
        sublabel: "response",
        dim: "T tokens",
        variant: "output",
        math: "y_{1:T} = \\prod_{t=1}^{T} P(y_t \\mid y_{<t}, X)",
        explanation: "The model's natural-language response, decoded autoregressively.",
      },
    ],
    highlights: [
      "Simplest design: an MLP is enough — no Q-Former required.",
      "Visual tokens are just prepended to the text tokens (prefix prompting).",
      "End-to-end fine-tuning of projector + LM yields strong instruction-following.",
    ],
  },
];

/* ------------------------------------------------------------------ */
/*  Pipeline component                                                  */
/* ------------------------------------------------------------------ */

function Pipeline({
  stages,
  selected,
  onSelect,
}: {
  stages: Stage[];
  selected: StageKind;
  onSelect: (id: StageKind) => void;
}) {
  return (
    <div className="flex flex-wrap items-stretch gap-2">
      {stages.map((s, i) => (
        <div key={s.id} className="flex items-stretch gap-2">
          <button
            onClick={() => onSelect(s.id)}
            className={cn(
              "flex w-28 flex-col items-center justify-center rounded-lg border-2 p-2 text-center transition-all",
              STAGE_COLORS[s.id],
              selected === s.id && "ring-2 ring-primary ring-offset-1 scale-[1.03]"
            )}
            aria-pressed={selected === s.id}
          >
            <span className="text-[10px] font-bold uppercase tracking-wide text-foreground">
              {s.label}
            </span>
            {s.sublabel && (
              <span className="mt-0.5 text-[10px] text-muted-foreground">{s.sublabel}</span>
            )}
            <DimBadge dims={s.dim} variant={s.variant} className="mt-1 text-[9px]" />
          </button>
          {i < stages.length - 1 && (
            <div className="flex items-center text-muted-foreground">
              <ArrowRight className="h-4 w-4" />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Detail panel shown when a stage is selected                       */
/* ------------------------------------------------------------------ */

function StageDetail({ stage, arch }: { stage: Stage; arch: ArchDef }) {
  return (
    <Card className="border-primary/30">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <span>{stage.label}</span>
          {stage.sublabel && (
            <Badge variant="secondary" className="text-[10px]">
              {stage.sublabel}
            </Badge>
          )}
          <span className="ml-auto text-xs text-muted-foreground">{arch.name}</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Shape</span>
          <DimBadge dims={stage.dim} variant={stage.variant} />
        </div>
        <div className="rounded-md border bg-muted/40 p-3">
          <MathBlock block>{stage.math}</MathBlock>
        </div>
        <p className="text-muted-foreground leading-relaxed">{stage.explanation}</p>
      </CardContent>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/*  Main module                                                         */
/* ------------------------------------------------------------------ */

export function Module13_Architecture() {
  const [archId, setArchId] = useState<string>("llava");
  const [selected, setSelected] = useState<StageKind>("projector");

  const arch = ARCHES.find((a) => a.id === archId)!;
  const stage = arch.stages.find((s) => s.id === selected) ?? arch.stages[0];

  return (
    <div>
      <ModuleHeader
        number={13}
        title="VLM Architecture: Vision-to-Language Pipeline"
        subtitle="From pixels to tokens to text. Click any stage to inspect its math and shape — switch tabs to compare five canonical architectures."
      >
        <MathBlock block>
          {`\\text{VLM}(X) = \\text{Decode}\\!\\left(\\text{LM}\\!\\left(\\text{Projector}(\\text{VisionEncoder}(X))\\right)\\right)`}
        </MathBlock>
      </ModuleHeader>

      <div className="mb-4 rounded-lg border border-amber-500/40 bg-amber-500/5 p-3 text-sm">
        <div className="flex items-start gap-2">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
          <p>
            <strong>Simplification note.</strong> This is a simplified educational representation;
            real models vary in their patch sizes, normalisation layers, prompt templates, training
            objectives, and many engineering details. The diagrams below abstract away those
            details to highlight the four canonical building blocks:{" "}
            <em>vision encoder → visual tokens → projector → language model</em>.
          </p>
        </div>
      </div>

      <Tabs value={archId} onValueChange={(v) => { setArchId(v); setSelected("projector"); }}>
        <TabsList className="grid w-full grid-cols-2 gap-1 md:grid-cols-5">
          {ARCHES.map((a) => (
            <TabsTrigger key={a.id} value={a.id} className="text-xs">
              <span className="flex flex-col items-center">
                <span>{a.short}</span>
                <span className="text-[9px] text-muted-foreground">{a.year}</span>
              </span>
            </TabsTrigger>
          ))}
        </TabsList>

        {ARCHES.map((a) => (
          <TabsContent key={a.id} value={a.id} className="mt-4 space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex flex-wrap items-baseline gap-2 text-base">
                  {a.name}
                  <Badge variant="outline" className="text-[10px]">{a.year}</Badge>
                </CardTitle>
                <p className="text-sm text-muted-foreground">{a.oneLiner}</p>
              </CardHeader>
              <CardContent>
                <Pipeline stages={a.stages} selected={selected} onSelect={setSelected} />
                <p className="mt-3 text-xs text-muted-foreground">
                  Click any box to inspect its math and tensor shape. The default selection is{" "}
                  <strong>Projector</strong> — the smallest module that connects vision to language.
                </p>
              </CardContent>
            </Card>

            <StageDetail stage={stage} arch={a} />

            <Card className="bg-muted/20">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Architecture highlights</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-1.5 text-sm">
                  {a.highlights.map((h, i) => (
                    <li key={i} className="flex gap-2">
                      <span className="text-primary">•</span>
                      <span>{h}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          </TabsContent>
        ))}
      </Tabs>

      <div className="mt-4 grid gap-3 md:grid-cols-2">
        <ExpandableSection title="The four shared building blocks of every VLM" variant="intuition" defaultOpen>
          <ol className="list-decimal space-y-2 pl-5">
            <li>
              <strong>Vision encoder</strong> — turns pixels into a sequence of visual feature
              vectors. Modern VLMs almost universally use a ViT (often CLIP-pretrained).
            </li>
            <li>
              <strong>Visual tokens</strong> — the encoder output. May be 1 vector (CNN+pool),
              N patches (ViT/LLaVA), or M learned queries (BLIP-2).
            </li>
            <li>
              <strong>Projector</strong> — a small bridge that converts visual features into the
              LM's embedding space. Linear, MLP, Q-Former, or resampler.
            </li>
            <li>
              <strong>Language model</strong> — an autoregressive transformer that consumes the
              visual tokens as a prefix and produces text one token at a time.
            </li>
          </ol>
          <p className="mt-2">
            Modules 14 and 15 zoom into the projector and into cross-attention respectively; Modules
            16 and 17 cover the LM head and decoding.
          </p>
        </ExpandableSection>

        <ExpandableSection title="Why freeze the vision encoder and LLM?" variant="why">
          <p>
            Modern VLMs (BLIP-2, LLaVA stage 1) freeze both the vision encoder and the LLM, training
            only the small projector. This is dramatically cheaper than full fine-tuning, preserves
            the pretrained knowledge of both large models, and reduces catastrophic forgetting.
          </p>
          <p className="mt-2">
            LLaVA stage 2 unfreezes the projector + LM head jointly on instruction-tuning data to
            unlock dialogue ability. BLIP-2 keeps the LLM frozen throughout.
          </p>
        </ExpandableSection>

        <ExpandableSection title="Why prepend visual tokens instead of using cross-attention?" variant="how">
          <p>
            LLaVA concatenates projected visual tokens with text tokens and feeds the combined
            sequence to a single decoder-only transformer. The LM's ordinary self-attention then
            mixes text and image — no architectural change is needed. This is the simplest possible
            integration and works surprisingly well at scale.
          </p>
          <p className="mt-2">
            BLIP-2 instead uses the Q-Former (which contains explicit cross-attention layers) before
            the LM. Encoder-decoder LMs (e.g. T5-based PaLI) also use cross-attention inside the
            decoder. See Module 15 for the math.
          </p>
        </ExpandableSection>

        <ExpandableSection title="CLIP vs LLaVA — both use CLIP-ViT, so what's the difference?" variant="why">
          <p>
            CLIP is a <em> dual encoder</em> trained with a contrastive loss to align image and text
            embeddings in a shared space. It can <em>retrieve</em> the best caption but cannot{" "}
            <em>generate</em> one — there is no autoregressive decoder.
          </p>
          <p className="mt-2">
            LLaVA <em>reuses</em> CLIP's image encoder as a frozen feature extractor, then trains a
            projector and a generative LLM (Vicuna) on top. The CLIP-ViT weights are the same; the
            downstream use is entirely different.
          </p>
        </ExpandableSection>
      </div>

      <VivaPanel
        questions={[
          {
            level: "Easy",
            question: "Name the four shared building blocks of a modern VLM and their roles.",
            hint: "Look at the pipeline labels left to right.",
            answer:
              "Vision encoder (pixels → visual feature vectors), visual tokens (the encoder output sequence), projector (vision space → LM space), language model (autoregressive text generation conditioned on the visual tokens).",
            explanation:
              "All five architectures in this module share these four blocks; what differs is the choice of encoder (CNN vs ViT vs CLIP-ViT), the number of visual tokens (1, N, or M), and the projector type (Linear, MLP, Q-Former).",
          },
          {
            level: "Medium",
            question:
              "Why does CLIP not generate captions even though it has both an image and a text encoder?",
            hint: "Think about the loss function and the direction of computation.",
            answer:
              "CLIP's text encoder produces a single pooled embedding u for contrastive matching (InfoNCE loss), not an autoregressive distribution over tokens. There is no causal mask, no LM head, and no P(y_t | y_<t). It can retrieve the closest caption but not produce one.",
            explanation:
              "BLIP-2 and LLaVA add an autoregressive decoder LM on top of a CLIP-style vision encoder to gain generation ability. The encoder alone, trained with contrastive learning, is insufficient for generation.",
          },
          {
            level: "Difficult",
            question:
              "BLIP-2's Q-Former compresses N=256 patch features into M=32 query embeddings before feeding the LLM. Give two reasons this is beneficial and one downside.",
            hint: "Consider the LLM's context length, training cost, and information loss.",
            answer:
              "Benefit 1: The frozen LLM only needs to process 32 visual tokens instead of 256 — smaller context, faster inference, lower memory. Benefit 2: The 32 learned queries act as a bottleneck that forces the Q-Former to extract task-relevant information (captioning, VQA) rather than memorise every patch. Downside: information loss — fine spatial detail (e.g. small objects, exact counts) can be discarded by the compression, hurting performance on detail-sensitive benchmarks.",
            explanation:
              "LLaVA chooses the opposite trade-off: pass all 576 patches through a simple MLP and let the LLM self-attention figure out what to attend to. This is more expensive but preserves spatial detail. The choice between Q-Former and 'just project everything' is an active design axis in modern VLMs.",
          },
        ]}
      />
    </div>
  );
}
