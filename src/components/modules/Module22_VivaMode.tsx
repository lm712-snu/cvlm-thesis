"use client";

import { useState, useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Search, BookOpen } from "lucide-react";
import { VivaCard, type VivaQuestion } from "@/components/cvlm/VivaCard";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { MODULES } from "@/data/modules";

interface BankQuestion extends VivaQuestion {
  moduleNumber: number;
  moduleId: string;
}

const QUESTIONS: BankQuestion[] = [
  // Module 1
  {
    moduleNumber: 1,
    moduleId: "image-repr",
    level: "Easy",
    question: "What is a pixel, mathematically?",
    hint: "Think about how many numbers describe one pixel.",
    answer: "A pixel in an RGB image is a 3-vector (R, G, B) ∈ ℝ³.",
    explanation:
      "Each colour channel stores an intensity. In an 8-bit image each component is an integer in [0, 255]. Conceptually a pixel is a point in a 3-dimensional colour space.",
  },
  {
    moduleNumber: 1,
    moduleId: "image-repr",
    level: "Medium",
    question: "Why divide image pixels by 255 before feeding them to a neural network?",
    hint: "Consider what range the weights were initialised for.",
    answer:
      "Normalisation keeps activations in a small range, prevents gradient saturation in sigmoid/tanh, and matches the input distribution that weight initialisations (Xavier, He) assume.",
    explanation:
      "ImageNet-style training additionally subtracts per-channel means and divides by per-channel std, so transfer learning requires the same normalisation at inference time.",
  },
  // Module 2
  {
    moduleNumber: 2,
    moduleId: "convolution",
    level: "Easy",
    question: "What is a convolution kernel and what does it compute at each output position?",
    hint: "Think of it as a small weight matrix that slides over the image.",
    answer:
      "A kernel is a small (k×k) weight matrix. At each output position it computes the element-wise product with the corresponding image window, then sums (optionally plus bias).",
    explanation:
      "Different kernels detect different features: edges (Sobel), blur (Gaussian), sharpening. CNNs learn these kernels automatically from data.",
  },
  {
    moduleNumber: 2,
    moduleId: "convolution",
    level: "Difficult",
    question: "With a 5×5 input, 3×3 kernel, stride 1, padding 0, what is the output size and why?",
    hint: "Apply the formula floor((H + 2p − k)/s) + 1.",
    answer:
      "Output is 3×3. The kernel can be placed at top-left positions (0,0), (0,1), (0,2), (1,0), … (2,2), giving 3 valid positions per axis.",
    explanation:
      "The output shrinks because the kernel cannot extend beyond the image boundary. Padding the input by 1 would preserve the 5×5 size.",
  },
  // Module 3
  {
    moduleNumber: 3,
    moduleId: "cnn",
    level: "Easy",
    question: "Why apply ReLU after every conv layer in a CNN?",
    hint: "What would happen if every layer were purely linear?",
    answer:
      "ReLU introduces non-linearity between linear conv layers. Without it, a stack of convolutions would collapse to a single linear map, losing expressiveness.",
    explanation:
      "ReLU also adds sparsity (zeroing negative activations) and helps with gradient flow compared to sigmoid, which saturates and vanishes.",
  },
  {
    moduleNumber: 3,
    moduleId: "cnn",
    level: "Medium",
    question: "What does max-pooling contribute to a CNN?",
    hint: "Think about spatial size and invariance.",
    answer:
      "Max-pooling reduces spatial dimensions (downsampling), provides a small amount of translation invariance, and keeps only the strongest activations in each window.",
    explanation:
      "By halving H and W each pool, the network gets cheaper and deeper. The translation invariance arises because a small shift in the input often leaves the max in the same pool.",
  },
  // Module 4
  {
    moduleNumber: 4,
    moduleId: "patches",
    level: "Medium",
    question: "Given a 224×224 image with patch size P=16, how many patches are extracted and what is each patch's flattened dimension?",
    hint: "Number of patches = (H/P)·(W/P).",
    answer:
      "N = (224/16)·(224/16) = 14·14 = 196 patches. Each flattened patch is P²·C = 16·16·3 = 768.",
    explanation:
      "The 14×14 grid of patches becomes the sequence length N=196 in the ViT. Each patch is then linearly projected to the embedding dimension d (e.g. 768).",
  },
  // Module 5
  {
    moduleNumber: 5,
    moduleId: "patch-embed",
    level: "Easy",
    question: "What is a patch embedding and how is it computed?",
    hint: "It is a linear projection of the flattened patch.",
    answer:
      "z_n = x_p^{(n)} W_E + b_E, where x_p^{(n)} ∈ ℝ^(P²C) is the flattened patch and W_E ∈ ℝ^(P²C × d) is a learned weight matrix. Output z_n ∈ ℝ^d.",
    explanation:
      "This is equivalent to a 1×1 conv2d with stride P in a CNN. Both produce the same numbers; the linear form is easier to interpret.",
  },
  {
    moduleNumber: 5,
    moduleId: "patch-embed",
    level: "Medium",
    question: "Why use a single linear projection rather than an MLP for patch embedding?",
    hint: "Compute cost vs. expressiveness trade-off.",
    answer:
      "A linear projection is cheap (one matmul per patch) and sufficient — the transformer encoder that follows adds the necessary non-linear transformations. An MLP at the embedding stage would be redundant.",
    explanation:
      "The embedding stage's job is to map pixels into the feature space; the encoder's job is to compute contextual representations. Splitting the labour this way is both efficient and architecturally clean.",
  },
  // Module 6
  {
    moduleNumber: 6,
    moduleId: "positional",
    level: "Easy",
    question: "Why do transformers need positional encoding?",
    hint: "Self-attention is permutation-invariant by default.",
    answer:
      "Without positional information, the transformer sees its input as an unordered set of tokens — it cannot tell token 0 from token 7. Adding PE injects position information.",
    explanation:
      "Adding (not concatenating) PE is the standard choice; it modifies the embedding without increasing the dimension.",
  },
  {
    moduleNumber: 6,
    moduleId: "positional",
    level: "Medium",
    question: "Write the formula for sinusoidal positional encoding.",
    hint: "Different formulas for even and odd indices.",
    answer:
      "PE(pos, 2i) = sin(pos / 10000^(2i/d)),  PE(pos, 2i+1) = cos(pos / 10000^(2i/d)).",
    explanation:
      "The wavelength forms a geometric progression from 2π to 10000·2π, so the model can learn to attend to relative positions via linear combinations of PEs.",
  },
  {
    moduleNumber: 6,
    moduleId: "positional",
    level: "Difficult",
    question: "Why use both sin and cos instead of just one?",
    hint: "Think about how a model might infer relative position.",
    answer:
      "Using both sin and cos allows PE(pos+k) to be expressed as a linear function of PE(pos) for any k. This makes relative position easy to recover from the absolute encoding.",
    explanation:
      "If only sin were used, the relative-position shift matrix would not be invertible. The pairing of sin and cos gives the rotation matrix, which preserves all positional information.",
  },
  // Module 7
  {
    moduleNumber: 7,
    moduleId: "vit",
    level: "Easy",
    question: "What is a Vision Transformer (ViT) and what does it output?",
    hint: "Patches + attention.",
    answer:
      "ViT splits an image into patches, embeds each, adds positional encoding, and runs a stack of transformer encoder layers. The output is a sequence of visual tokens (one per patch).",
    explanation:
      "Unlike CNNs, ViT has no inductive bias about locality — attention is global from layer 1. This makes ViT more data-hungry but more flexible on large datasets.",
  },
  {
    moduleNumber: 7,
    moduleId: "vit",
    level: "Medium",
    question: "Why does ViT prepend a [CLS] token to the patch sequence?",
    hint: "How does the model produce a single classification output?",
    answer:
      "The [CLS] token is a learned embedding that aggregates global information through attention. Its final hidden state is used as the image-level representation (e.g. for classification).",
    explanation:
      "Pooling across all patch tokens also works, but [CLS] lets the model learn what to attend to. In VLMs, all patch tokens are typically used as visual context.",
  },
  // Module 8
  {
    moduleNumber: 8,
    moduleId: "self-attention",
    level: "Easy",
    question: "What do Q, K, V stand for in attention, and what is each one's role?",
    hint: "Three roles in information retrieval.",
    answer:
      "Q = queries (what a token is looking for), K = keys (what a token offers to be matched against), V = values (the content actually carried forward).",
    explanation:
      "Attention computes a weighted sum of values, where weights are softmax-normalised dot products of queries with keys.",
  },
  {
    moduleNumber: 8,
    moduleId: "self-attention",
    level: "Medium",
    question: "Why scale QKᵀ by 1/√d_k before softmax?",
    hint: "Variance of a sum of independent random variables.",
    answer:
      "Raw q·k has variance d_k. Without scaling, large d_k pushes softmax into saturated regions where gradients vanish. Dividing by √d_k keeps variance ≈ 1.",
    explanation:
      "This is the 'scaled' in 'scaled dot-product attention'. Without it, training deep transformers becomes unstable for large d_k (e.g. 64 or 128).",
  },
  {
    moduleNumber: 8,
    moduleId: "self-attention",
    level: "Difficult",
    question: "Why must softmax be applied row-wise (along the key axis), not column-wise?",
    hint: "What does each row of the attention matrix represent?",
    answer:
      "Each row of QKᵀ describes how strongly one query token attends to every key token. Row-wise softmax normalises those weights into a probability distribution per query.",
    explanation:
      "If we softmaxed along columns, each key would have its own distribution over queries — a different (and less useful) operation. The convention is: queries attend to keys.",
  },
  // Module 9
  {
    moduleNumber: 9,
    moduleId: "multihead",
    level: "Easy",
    question: "What is multi-head attention?",
    hint: "Multiple attention computations running in parallel.",
    answer:
      "Multi-head attention runs h parallel attention computations (heads), each with its own Q/K/V projections, concatenates their outputs, and applies a final linear projection W_O.",
    explanation:
      "Formula: MultiHead(X) = Concat(head_1, …, head_h) W_O. Each head can specialise on different relations (e.g. syntactic, semantic, positional).",
  },
  {
    moduleNumber: 9,
    moduleId: "multihead",
    level: "Medium",
    question: "How is d_model split across heads, and what constraint does this impose?",
    hint: "Equal split.",
    answer:
      "d_k = d_v = d_model / h. This requires d_model to be divisible by h. Typical: d_model=512, h=8 → d_k=64.",
    explanation:
      "The total compute is roughly the same as single-head attention with the same d_model because we are splitting, not duplicating the dimension.",
  },
  {
    moduleNumber: 9,
    moduleId: "multihead",
    level: "Difficult",
    question: "Why use multiple heads instead of one wide head with the same total dimension?",
    hint: "Think about what each head can specialise on.",
    answer:
      "Multiple heads let the model attend to different relations in parallel (one head may track syntax, another semantics, another position). A single wide head must squeeze all of this into one attention pattern per layer.",
    explanation:
      "Empirically, ablating heads reveals that some heads consistently attend to specific syntactic patterns. Multiple heads also give a richer gradient signal during training.",
  },
  // Module 10
  {
    moduleNumber: 10,
    moduleId: "text-repr",
    level: "Easy",
    question: "What is tokenization and why is it used?",
    hint: "Words vs. sub-words vs. characters.",
    answer:
      "Tokenization splits text into discrete units (tokens), each mapped to an integer ID. Sub-word tokenizers (BPE, WordPiece) balance vocabulary size and sequence length.",
    explanation:
      "Tokens are the atoms a language model operates on. Choosing the right granularity (word, sub-word, character) trades off vocabulary size against sequence length.",
  },
  {
    moduleNumber: 10,
    moduleId: "text-repr",
    level: "Medium",
    question: "Why use learned embeddings instead of one-hot token IDs?",
    hint: "Think about distances between tokens.",
    answer:
      "One-hot encodings are orthogonal — no information about token similarity. Learned embeddings place semantically related tokens near each other in d-dimensional space, which the model can exploit.",
    explanation:
      "Embeddings are just a lookup table E ∈ ℝ^(V × d) where row i is the embedding of token i. They are learned end-to-end with the rest of the model.",
  },
  // Module 11
  {
    moduleNumber: 11,
    moduleId: "alignment",
    level: "Easy",
    question: "What is CLIP and what does it align?",
    hint: "Two encoders, one shared space.",
    answer:
      "CLIP is a contrastive image-text model. It trains an image encoder and a text encoder so that matching (image, caption) pairs have high cosine similarity in a shared embedding space.",
    explanation:
      "Once trained, CLIP can be used for zero-shot classification: encode the image and each candidate label, pick the highest-similarity label.",
  },
  {
    moduleNumber: 11,
    moduleId: "alignment",
    level: "Medium",
    question: "How does CLIP align image and text embeddings during training?",
    hint: "A 2N × 2N batch, diagonal positive, off-diagonal negative.",
    answer:
      "CLIP uses InfoNCE contrastive loss on cosine similarities within a batch of N image-text pairs. The diagonal (matching pairs) is positive; off-diagonal (mismatched) is negative. Softmax with temperature τ pushes positives together and negatives apart.",
    explanation:
      "Both encoders are trained jointly from scratch on ~400M (image, caption) pairs scraped from the web. No labels are needed — the supervision comes from natural pairing.",
  },
  // Module 12
  {
    moduleNumber: 12,
    moduleId: "contrastive",
    level: "Easy",
    question: "What is contrastive learning?",
    hint: "Pull similar things together, push different things apart.",
    answer:
      "Contrastive learning trains encoders so that semantically similar inputs are close in embedding space and dissimilar inputs are far apart, using positive and negative pairs.",
    explanation:
      "It is self-supervised: the labels come from data augmentations (SimCLR) or natural co-occurrence (CLIP), not human annotation.",
  },
  {
    moduleNumber: 12,
    moduleId: "contrastive",
    level: "Medium",
    question: "What is the InfoNCE loss and what is its purpose?",
    hint: "Categorical cross-entropy over a batch of candidates.",
    answer:
      "InfoNCE = −(1/B) Σ_i log( exp(sim(z_i, z_i⁺)/τ) / Σ_j exp(sim(z_i, z_j)/τ) ). It is the cross-entropy of identifying the true positive among the in-batch negatives.",
    explanation:
      "It is a categorical classification problem: the query must 'pick out' its positive from the batch. Lower temperature makes this harder, sharpening representations.",
  },
  {
    moduleNumber: 12,
    moduleId: "contrastive",
    level: "Difficult",
    question: "Why does temperature τ matter in InfoNCE, and what happens at the extremes?",
    hint: "Think of τ as controlling the sharpness of the softmax.",
    answer:
      "τ controls the sharpness of the softmax over similarities. Low τ → very sharp, the model focuses on the closest negative; high τ → uniform, gradients vanish. Optimal τ balances signal vs. noise.",
    explanation:
      "CLIP uses a learned temperature initialised near 0.07. The temperature is a hyperparameter (or learned) that controls how 'strict' the contrastive task is.",
  },
  // Module 13
  {
    moduleNumber: 13,
    moduleId: "architecture",
    level: "Easy",
    question: "What are the three main components of a VLM?",
    hint: "Vision, language, and the thing between them.",
    answer:
      "1) A vision encoder (ViT or CNN) producing visual features. 2) A bridge: a projector or cross-attention that aligns visual features to the LM's space. 3) A language model (transformer decoder) that produces text.",
    explanation:
      "Different VLMs instantiate these three pieces differently: LLaVA uses a simple MLP projector; BLIP-2 uses a learned Q-Former; Flamingo uses perceiver resampler + cross-attention.",
  },
  // Module 14
  {
    moduleNumber: 14,
    moduleId: "projector",
    level: "Easy",
    question: "What does the projector do in a VLM?",
    hint: "It is a dimension-matching linear (or MLP) layer.",
    answer:
      "The projector is a linear (or MLP) layer that maps visual features Z_vis ∈ ℝ^d_vision to LM-token-shaped vectors H ∈ ℝ^d_lm so the language model can read them.",
    explanation:
      "Formula: H = Z_vis W_P + b_P, with W_P ∈ ℝ^(d_vision × d_lm). Without it, the LM would receive vectors in a totally different basis and produce garbage.",
  },
  {
    moduleNumber: 14,
    moduleId: "projector",
    level: "Medium",
    question: "Why is the projector necessary? Can't we feed visual tokens directly to the LM?",
    hint: "What distribution does the LM expect its inputs to come from?",
    answer:
      "The LM's first layer was trained on token embeddings in a specific distribution (range, scale, direction). Raw visual tokens are in a different distribution. The projector is the learned alignment map.",
    explanation:
      "Empirically, even a single linear projector works surprisingly well (LLaVA-1.5). MLP projectors give a small boost. The projector is the cheapest part of the VLM to train, so it is usually the only part that is fine-tuned from scratch.",
  },
  // Module 15
  {
    moduleNumber: 15,
    moduleId: "cross-attention",
    level: "Easy",
    question: "What is cross-attention and how does it differ from self-attention?",
    hint: "Where do Q, K, V come from?",
    answer:
      "In self-attention Q, K, V all come from the same input. In cross-attention Q comes from one modality (e.g. text) and K, V come from another (e.g. image).",
    explanation:
      "Cross-attention is the mechanism that lets the language model 'look at' specific image patches while generating each token.",
  },
  {
    moduleNumber: 15,
    moduleId: "cross-attention",
    level: "Medium",
    question: "In a cross-attention layer, what is the shape of the attention weight matrix?",
    hint: "Rows are queries, columns are keys.",
    answer:
      "Attention weights A ∈ ℝ^(N_text × N_image), where N_text is the number of text tokens and N_image is the number of image patches. Each row of A is a probability distribution over image patches.",
    explanation:
      "A[i, j] is how strongly text token i attends to image patch j. This matrix can be visualised to see which parts of the image each word is grounded in.",
  },
  // Module 16
  {
    moduleNumber: 16,
    moduleId: "next-token",
    level: "Easy",
    question: "What is softmax and what is its role in next-token prediction?",
    hint: "Convert scores to probabilities.",
    answer:
      "Softmax converts a vector of logits z ∈ ℝ^V to a probability distribution P(y_i) = exp(z_i) / Σ_j exp(z_j). The token with the highest probability is the model's prediction.",
    explanation:
      "Softmax is differentiable, making it suitable for gradient-based training. The logits can be any real numbers; softmax maps them to the probability simplex.",
  },
  {
    moduleNumber: 16,
    moduleId: "next-token",
    level: "Medium",
    question: "Why use softmax instead of just dividing logits by their sum (normalisation)?",
    hint: "Softmax is differentiable and amplifies differences.",
    answer:
      "Softmax is differentiable and amplifies the largest logit through the exponent. Plain normalisation loses information about magnitude and has worse gradient properties.",
    explanation:
      "With softmax, a small difference in logits becomes a much larger difference in probabilities, which sharpens decisions. Softmax also connects naturally to cross-entropy loss.",
  },
  // Module 17
  {
    moduleNumber: 17,
    moduleId: "autoregressive",
    level: "Easy",
    question: "What is autoregressive generation?",
    hint: "One token at a time, conditioned on the past.",
    answer:
      "Autoregressive generation produces a sequence one token at a time, each token conditioned on the previously generated tokens. P(y_1, …, y_T | X) = Π_t P(y_t | y_<t, X).",
    explanation:
      "GPT-style models are autoregressive. BERT is bidirectional and cannot generate; T5 is encoder-decoder.",
  },
  {
    moduleNumber: 17,
    moduleId: "autoregressive",
    level: "Medium",
    question: "Why factorise the joint probability P(y_1, …, y_T) as a product of conditionals?",
    hint: "Chain rule of probability.",
    answer:
      "By the chain rule, P(y_1, …, y_T) = Π_t P(y_t | y_1, …, y_{t-1}). Each factor is a manageable categorical distribution that the model can learn.",
    explanation:
      "This factorisation makes generation tractable: you can sample y_1, then y_2 conditioned on y_1, and so on. The cost is that errors compound over long sequences.",
  },
  {
    moduleNumber: 17,
    moduleId: "autoregressive",
    level: "Difficult",
    question: "What is teacher forcing and why is it used during training but not inference?",
    hint: "During training we know the next token; during inference we don't.",
    answer:
      "Teacher forcing feeds the ground-truth previous token (not the model's prediction) as input when training the next-step predictor. This gives stable gradients but creates a train/test mismatch.",
    explanation:
      "At inference, the model must use its own predictions, which may differ from ground truth — known as 'exposure bias'. Scheduled sampling and reinforcement learning are partial fixes.",
  },
  // Module 18
  {
    moduleNumber: 18,
    moduleId: "pipeline",
    level: "Medium",
    question: "List the high-level stages of a VLM pipeline from image to text.",
    hint: "Image → … → text.",
    answer:
      "Image → Patches → Patch embed → Positional → ViT encoder → Visual tokens → Projector → Cross-attention (optional) → LM layers → Logits → Softmax → Decode.",
    explanation:
      "Each stage corresponds to a mathematical operation covered in Modules 1-17. The pipeline is fully differentiable end-to-end, allowing joint training of all components (or selective fine-tuning).",
  },
  // Module 19
  {
    moduleNumber: 19,
    moduleId: "sandbox",
    level: "Easy",
    question: "In matrix multiplication A·B, what must be true of A's columns and B's rows?",
    hint: "The inner dimensions.",
    answer:
      "A's column count must equal B's row count. (m, k) · (k, n) → (m, n). Otherwise the dot product is undefined.",
    explanation:
      "This is the inner-dimension rule. The output dimensions are the outer dimensions (m and n).",
  },
  // Module 20
  {
    moduleNumber: 20,
    moduleId: "dimension-tracker",
    level: "Easy",
    question: "What is the formula for Conv2d output spatial size?",
    hint: "Depends on input, kernel, stride, padding.",
    answer:
      "H' = floor((H + 2p − k) / s) + 1, and similarly for W'.",
    explanation:
      "Common special cases: k=3, s=1, p=1 preserves size; k=2, s=2, p=0 halves size.",
  },
  // Module 21
  {
    moduleNumber: 21,
    moduleId: "thesis-demo",
    level: "Medium",
    question: "Why is the synthesized caption in the thesis demo not accurate to the uploaded image?",
    hint: "What does the demo run vs. what a real VLM runs?",
    answer:
      "The demo runs each mathematical operation explicitly with random or fixed small weights — there is no pretrained network. The caption is generated by a deterministic heuristic based on image statistics, not by a learned decoder.",
    explanation:
      "A real VLM (LLaVA, BLIP-2) replaces each stage with a pretrained neural network. The mathematical structure is identical; only the parameters (and the volume of training data) differ.",
  },
];

const LEVELS = ["All", "Easy", "Medium", "Difficult"] as const;
type LevelFilter = (typeof LEVELS)[number];

export function Module22_VivaMode() {
  const [query, setQuery] = useState("");
  const [level, setLevel] = useState<LevelFilter>("All");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return QUESTIONS.filter((qq) => {
      if (level !== "All" && qq.level !== level) return false;
      if (!q) return true;
      return (
        qq.question.toLowerCase().includes(q) ||
        qq.hint.toLowerCase().includes(q) ||
        qq.answer.toLowerCase().includes(q) ||
        qq.explanation.toLowerCase().includes(q)
      );
    });
  }, [query, level]);

  // Group by module (preserve MODULES order).
  const grouped = useMemo(() => {
    return MODULES.map((m) => ({
      module: m,
      questions: filtered.filter((q) => q.moduleId === m.id),
    })).filter((g) => g.questions.length > 0);
  }, [filtered]);

  const counts = useMemo(() => {
    const c: Record<LevelFilter, number> = { All: QUESTIONS.length, Easy: 0, Medium: 0, Difficult: 0 };
    for (const q of QUESTIONS) c[q.level]++;
    return c;
  }, []);

  return (
    <div>
      <ModuleHeader
        number={22}
        title="Viva Mode"
        subtitle="A unified question bank covering every module. Filter by difficulty, search by keyword, reveal hints and answers when you are ready."
      >
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <DimBadge dims={`${QUESTIONS.length} questions`} variant="output" />
          <DimBadge dims={`${MODULES.length} modules`} variant="intermediate" />
          <DimBadge dims={`Easy: ${counts.Easy} · Med: ${counts.Medium} · Hard: ${counts.Difficult}`} variant="label" />
        </div>
      </ModuleHeader>

      <Card className="mb-4">
        <CardContent className="py-4 space-y-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by keyword (e.g. softmax, attention, projector, contrastive)…"
              className="pl-9"
            />
          </div>
          <Tabs value={level} onValueChange={(v) => setLevel(v as LevelFilter)}>
            <TabsList>
              {LEVELS.map((l) => (
                <TabsTrigger key={l} value={l}>
                  {l} <span className="ml-1 text-[10px] text-muted-foreground">({counts[l]})</span>
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
        </CardContent>
      </Card>

      {grouped.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center text-sm text-muted-foreground">
            No questions match your filters. Try a different keyword or difficulty.
          </CardContent>
        </Card>
      ) : (
        <Accordion type="multiple" defaultValue={grouped.map((g) => g.module.id)} className="space-y-2">
          {grouped.map((g) => (
            <AccordionItem
              key={g.module.id}
              value={g.module.id}
              className="rounded-lg border bg-card/40 px-3"
            >
              <AccordionTrigger className="hover:no-underline">
                <div className="flex flex-wrap items-center gap-2 text-left">
                  <Badge variant="outline" className="text-[10px] font-mono">M{g.module.number}</Badge>
                  <span className="font-medium">{g.module.title}</span>
                  <Badge variant="secondary" className="text-[10px]">
                    <BookOpen className="h-3 w-3 mr-1" /> {g.questions.length}
                  </Badge>
                </div>
              </AccordionTrigger>
              <AccordionContent>
                <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3 pt-2">
                  {g.questions.map((q, i) => (
                    <VivaCard key={i} question={q} />
                  ))}
                </div>
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      )}
    </div>
  );
}
