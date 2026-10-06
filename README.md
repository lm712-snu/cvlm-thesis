# Mathematics of Computer Vision Language Models

An interactive educational web application that **visualizes, calculates, and explains** the mathematics
between an input image and a language-model output. Built for an undergraduate Mathematics + AI/ML thesis.

> The goal is NOT a chatbot. The goal is to make every matrix, tensor, and operation (convolution,
> attention, softmax, cross-entropy) **explicit and inspectable** — no hidden `torch.nn.MultiheadAttention`
> calls in the educational path.

## What this is

A static Next.js web app with **22 interactive modules** that walk through the full vision-language pipeline:

**Image → Tensor → Convolution → Patches → Embeddings → Attention → Cross-Modal → Projector →
Language Model → Logits → Softmax → Generated Text**

Every module exposes:

1. The mathematical equation (rendered with KaTeX)
2. A step-by-step numerical computation on small explicit matrices
3. Interactive controls (sliders, kernels, dimensions)
4. Heatmaps / matrix views of every intermediate value
5. Expandable sections: *Mathematics*, *Why?*, *How?*, *Intuition*, *Numerical Example*
6. Three viva questions (Easy / Medium / Difficult) with hint + answer reveal

All math runs **client-side in TypeScript** — no Python server, no API keys, no model downloads.

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Framework | Next.js 16 (App Router) | Static export to GitHub Pages |
| Language | TypeScript 5 (strict) | Type-safe math |
| Styling | Tailwind CSS 4 + shadcn/ui | Clean academic look, dark/light mode |
| Math rendering | KaTeX + react-katex | LaTeX in the browser |
| Math engine | Custom TypeScript (`src/lib/math/`) | Explicit, no library black boxes |
| Charts | Recharts | Bar charts for softmax / similarity |
| State | Zustand | Shared image + theme across modules |

> **Note on Python/Streamlit:** The original brief mentioned Python + Streamlit. We deliberately
> replaced this with a static Next.js app because **GitHub Pages only hosts static files** —
> Streamlit requires a running Python server and cannot be deployed on `github.io`. All the
> math that would have been done with NumPy/PyTorch is instead implemented explicitly in
> `src/lib/math/*.ts`, which is actually *better* for the educational goal: every operation
> is auditable in the browser dev tools.

## Quick start (local dev)

```bash
git clone <your-repo-url>
cd <your-repo>
npm install        # or: bun install
npm run dev        # or: bun run dev
# open http://localhost:3000
```

## Deploy to GitHub Pages

The app is configured for static export. Several deployment paths:

### Option A — GitHub Actions (recommended, automatic)

A GitHub Actions workflow at `.github/workflows/deploy.yml` is included. It automatically:

1. Builds the static site on every push to `main`
2. Patches `next.config.ts` with the correct `basePath` for your repo name
3. Deploys to GitHub Pages

**To enable:**

1. Push this repo to GitHub (instructions below)
2. In repo settings → **Pages** → **Source**, select **GitHub Actions**
3. Push to `main` — the workflow runs and your site goes live at
   `https://<your-username>.github.io/cvlm-thesis/`

```bash
git init
git remote add origin https://github.com/lm712-snu/cvlm-thesis.git
git add .
git commit -m "Initial commit: Mathematics of CVLMs thesis tool"
git branch -M main
git push -u origin main
# → GitHub Actions deploys to https://lm712-snu.github.io/cvlm-thesis/
```

### Option B — Project pages (`https://<user>.github.io/<repo-name>/`) via `gh-pages` branch

```bash
./scripts/build-gh-pages.sh cvlm-thesis
# ./out now contains the static site
cd out
git init && git checkout -b gh-pages
git add .
git commit -m "Deploy to GitHub Pages"
git remote add origin https://github.com/lm712-snu/cvlm-thesis.git
git push -u origin gh-pages
# In repo settings → Pages, set source to the `gh-pages` branch
```

### Option C — User/org pages (`https://<user>.github.io/`)

```bash
# No repo name needed (served from root)
./scripts/build-gh-pages.sh
# Then push the contents of ./out to the main branch of <user>.github.io repo
```

### Option D — `docs/` folder

```bash
./scripts/build-gh-pages.sh cvlm-thesis
mkdir -p docs
cp -r out/* docs/
git add docs
git commit -m "Add static site to docs/"
git push
# Then enable Pages in repo settings → Pages → Source: main / docs
```

### Manual configuration

If you prefer not to use the script, edit `next.config.ts`:

```ts
import type { NextConfig } from "next";
const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  basePath: "/cvlm-thesis",          // omit for user pages
  assetPrefix: "/cvlm-thesis/",      // omit for user pages
};
export default nextConfig;
```

Then run `npm run build:static` — the `out/` directory is your deployable site.

## Project Structure

```
.
├── src/
│   ├── app/
│   │   ├── layout.tsx           # Root layout (fonts, metadata)
│   │   ├── page.tsx             # Module router (maps activeModule → component)
│   │   └── globals.css          # Tailwind + theme tokens (light/dark)
│   ├── components/
│   │   ├── ui/                  # shadcn/ui components (Card, Button, Slider, …)
│   │   ├── cvlm/                # Shared educational UI:
│   │   │   ├── MathBlock.tsx       # KaTeX renderer (with error boundary)
│   │   │   ├── MatrixView.tsx      # 2D matrix as HTML table + heatmap
│   │   │   ├── Heatmap.tsx         # Diverging/sequential heatmap
│   │   │   ├── DimBadge.tsx        # Tensor dimension pill (196 × 768)
│   │   │   ├── ExpandableSection.tsx  # "Mathematics" / "Why?" / "How?" blocks
│   │   │   ├── ModuleHeader.tsx     # Per-module title + equation
│   │   │   ├── VivaCard.tsx         # Viva question with hint/answer reveal
│   │   │   ├── ImageUploader.tsx    # Client-side image → tensor
│   │   │   └── Shell.tsx            # Sidebar + topbar + theme toggle
│   │   └── modules/             # 22 module implementations
│   │       ├── HomeDashboard.tsx
│   │       ├── Module1_ImageRepresentation.tsx
│   │       ├── Module2_Convolution.tsx
│   │       ├── ... (Modules 3–17)
│   │       ├── Module18_Pipeline.tsx
│   │       ├── Module19_Sandbox.tsx
│   │       ├── Module20_DimensionTracker.tsx
│   │       ├── Module21_ThesisDemo.tsx
│   │       └── Module22_VivaMode.tsx
│   ├── data/
│   │   └── modules.ts            # Module catalog (id, number, title, category)
│   ├── lib/
│   │   ├── math/                 # EXPLICIT math library (no black boxes):
│   │   │   ├── matrix.ts         # matmul, transpose, relu, softmax, layerNorm, cosineSim, …
│   │   │   ├── convolution.ts    # conv2d, maxPool2x2, KERNELS (Sobel, Blur, …)
│   │   │   ├── attention.ts       # selfAttention, crossAttention, multiHead, sinusoidalPE, infoNCE, topK, topP
│   │   │   └── index.ts
│   │   └── utils.ts
│   ├── store/
│   │   └── app-store.ts          # Zustand: activeModule, theme, sharedImage
│   └── hooks/
├── scripts/
│   └── build-gh-pages.sh         # Static-export build script for GitHub Pages
├── next.config.ts                # Dev config (standalone)
├── next.config.gh-pages.ts       # Reference config for static export
├── package.json
└── README.md
```

## Module Catalog

| # | Module | Category | What it shows |
|---|---|---|---|
| – | Home Dashboard | – | Overview + highlights |
| 1 | Image Representation | Image | Pixel → RGB channels → tensor X ∈ ℝ^(H×W×3) |
| 2 | Convolution | Image | Sliding kernel animation, stride, padding |
| 3 | CNN Feature Extraction | Image | Conv → ReLU → MaxPool stacked pipeline |
| 4 | Image Patches | Vision | 224×224 image → 196 patches of 16×16 |
| 5 | Patch Embeddings | Vision | Linear projection z = x_p · W_E + b |
| 6 | Positional Embeddings | Vision | Sinusoidal PE: sin/cos(pos/10000^(2i/d)) heatmap |
| 7 | Vision Transformer | Vision | Encoder block: LN → Attn → Residual → MLP |
| 8 | **Self-Attention** | Attention | Q, K, V, scores, weights — fully explicit |
| 9 | Multi-Head Attention | Attention | h heads → concat → W_O |
| 10 | Text Representation | Language | Tokens → IDs → embeddings + PE |
| 11 | Image-Text Alignment (CLIP) | Alignment | Cosine similarity in shared space |
| 12 | Contrastive Learning | Alignment | InfoNCE loss with temperature τ |
| 13 | VLM Architecture | Pipeline | Clickable diagram; CNN/ViT/CLIP/BLIP-2/LLaVA |
| 14 | Vision-to-Language Projector | Pipeline | Z_visual · W_P + b: dimension transformation |
| 15 | Cross-Modal Attention | Attention | Q from text, K/V from image |
| 16 | Next Token Prediction | Language | Logits → softmax → probabilities |
| 17 | Autoregressive Generation | Language | Greedy / top-k / top-p decoding |
| 18 | End-to-End Pipeline | Pipeline | 14-stage flowchart with inspectable tensors |
| 19 | Mathematical Sandbox | Tools | Manual matrix calculator |
| 20 | Dimension Tracker | Tools | Auto dim propagation + mismatch detection |
| 21 | Thesis Demo | Pipeline | Full story: image → 12 stages → caption |
| 22 | Viva Mode | Tools | 41 thesis-defense questions with reveal answers |

---

# Mathematical Foundations

This section documents every mathematical concept used in the application, organized by topic. It is
intended as a thesis companion — read it alongside the corresponding interactive module.

## 1. Linear Algebra

### 1.1 Vectors

A vector **x** ∈ ℝⁿ is an ordered tuple of n real numbers. In this app, an image patch of size 16×16×3
becomes a vector of length 16·16·3 = 768.

The **dot product** of two vectors a, b ∈ ℝⁿ is:

  a · b = Σᵢ aᵢ bᵢ

The **L2 norm** measures vector length:

  ‖a‖ = √(Σᵢ aᵢ²) = √(a · a)

The **cosine similarity** between two vectors measures their angle, independent of magnitude:

  cos(a, b) = (a · b) / (‖a‖ ‖b‖)

This is the core operation in CLIP-style image-text alignment (Module 11).

### 1.2 Matrices

A matrix A ∈ ℝ^(m×n) has m rows and n columns. The entry A[i, j] is the element at row i, column j.

**Matrix multiplication** A (m×k) · B (k×n) → C (m×n):

  C[i, j] = Σₗ A[i, l] · B[l, j]

The inner dimension k must match. This is the **inner-dimension rule**, enforced by the Dimension
Tracker (Module 20).

The **transpose** Aᵀ swaps rows and columns: Aᵀ[i, j] = A[j, i]. Attention uses Kᵀ to turn
column-keys into row-keys for the dot product.

### 1.3 Tensors

A tensor is a multi-dimensional array. An RGB image is a 3-tensor X ∈ ℝ^(H×W×C). A batch of images
is a 4-tensor ℝ^(B×H×W×C). In this app we mostly work with 2D matrices for clarity, but every
matrix dimension is annotated with its semantic role (sequence length, embedding dim, etc.).

## 2. Image Representation

### 2.1 Pixel

A pixel in an RGB image is a 3-vector (R, G, B) ∈ [0, 255]³ (8-bit) or [0, 1]³ (normalized).
Conceptually a pixel is a point in 3D colour space (Module 1).

### 2.2 Image as Tensor

An image of height H, width W, and 3 colour channels is a 3-tensor X ∈ ℝ^(H×W×3). The entry
X[i, j, c] is the intensity of colour channel c at row i, column j.

### 2.3 Normalization

Pixel values are divided by 255 to lie in [0, 1]. Production pipelines further normalize
per-channel using statistics (e.g. ImageNet: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
so that pretrained weights receive inputs matching their training distribution.

### 2.4 Flattening

To feed a tensor into a linear layer, we reshape it into a vector:

  vec(X) ∈ ℝ^(H·W·C)

This destroys spatial structure. CNNs preserve it with convolution; Vision Transformers preserve it
by flattening *patches* rather than the whole image.

## 3. Convolution

### 3.1 Definition

2D convolution (cross-correlation form, as used in CNNs) of input X ∈ ℝ^(H×W) with kernel
K ∈ ℝ^(k×k) and bias b ∈ ℝ:

  Y[i, j] = Σₘ Σₙ X[i+m, j+n] · K[m, n] + b

The kernel slides across the input; at each position, it computes a weighted sum of the underlying
patch (Module 2).

### 3.2 Stride, Padding, Output Size

  out = ⌊(H + 2p − k) / s⌋ + 1

- **Stride s**: kernel step size. s=2 halves the output.
- **Padding p**: zeros added around the input. p = (k−1)/2 preserves the input size for odd k.

### 3.3 Multiple Channels and Filters

For C_in input channels and C_out filters, the weight tensor is W ∈ ℝ^(k×k×C_in×C_out). The output
becomes a 3-tensor Y ∈ ℝ^(H'×W'×C_out). Each output channel is a learned feature map.

### 3.4 Receptive Field

Each output pixel "sees" a k×k region of the input. Stacking two 3×3 convolutions gives a 5×5
receptive field — same as one 5×5 conv — but with 18 parameters instead of 25 and an extra nonlinearity.
This is why deep CNNs prefer stacks of small kernels (Module 3).

## 4. Functions and Activation

### 4.1 ReLU

  ReLU(x) = max(0, x)

ReLU is the most common activation in CNNs. It is non-saturating for positive inputs (gradient = 1),
which helps gradient flow. It introduces the nonlinearity that makes stacks of linear layers
expressive.

### 4.2 Softmax

  softmax(z)_i = exp(z_i) / Σⱼ exp(z_j)

Softmax converts a vector of logits into a probability distribution (sums to 1). With temperature τ:

  softmax(z; τ)_i = exp(z_i / τ) / Σⱼ exp(z_j / τ)

- Low τ → sharp distribution (confident predictions)
- High τ → flat distribution (more random sampling)
- τ → 0 → argmax (greedy)
- τ → ∞ → uniform

Softmax is used in attention (over keys) and in next-token prediction (over vocabulary).

### 4.3 Layer Normalization

For a vector x ∈ ℝ^d:

  LayerNorm(x) = γ · (x − μ) / √(σ² + ε) + β

where μ and σ² are the mean and variance of x, and γ, β are learnable scale and shift parameters.
LayerNorm stabilizes training by keeping activations in a consistent range. Used inside every
Transformer block.

## 5. Probability

### 5.1 Probability Distribution

A vector p ∈ ℝ^V is a probability distribution if p_i ≥ 0 and Σ p_i = 1. Softmax produces such a
distribution from logits.

### 5.2 Cross-Entropy Loss

For a predicted probability distribution p and a true label y (one-hot):

  L_CE = − Σᵢ yᵢ log(pᵢ) = −log(p_y)

When computed from logits directly (numerically stable):

  L_CE_from_logits = −log( softmax(z)_y )

This is the standard classification loss and the per-token loss in language models.

### 5.3 InfoNCE (Contrastive Loss)

For a batch of B image-text pairs with cosine similarity matrix S ∈ ℝ^(B×B) and temperature τ:

  p(image_i → text_j) = exp(S[i, j] / τ) / Σⱼ exp(S[i, j] / τ)

  L_image = −(1/B) Σᵢ log p(image_i → text_i)
  L_text  = −(1/B) Σⱼ log p(text_j → image_j)
  L = (L_image + L_text) / 2

The diagonal (matching pairs) is maximized; off-diagonal (mismatched) is minimized. This is how CLIP
is trained (Module 12).

## 6. Optimization (Background)

Although this app focuses on **inference** (forward pass), the trained weights came from optimization:

### 6.1 Gradient Descent

  θ ← θ − η · ∇L(θ)

where η is the learning rate. Stochastic gradient descent (SGD) and Adam are the most common
optimizers.

### 6.2 Backpropagation

The chain rule applied to a computation graph. For a layer y = f(x; θ):

  ∂L/∂x = ∂L/∂y · ∂f/∂x
  ∂L/∂θ = ∂L/∂y · ∂f/∂θ

This allows gradients to flow backward through every layer, including attention and convolution.
Although we don't compute gradients in this app (it's inference-only), understanding backprop is
essential for the thesis.

## 7. Attention

### 7.1 Scaled Dot-Product Self-Attention

Given input X ∈ ℝ^(n×d_model) and learnable weights W_Q, W_K ∈ ℝ^(d_model×d_k), W_V ∈ ℝ^(d_model×d_v):

  Q = X · W_Q        ∈ ℝ^(n×d_k)
  K = X · W_K        ∈ ℝ^(n×d_k)
  V = X · W_V        ∈ ℝ^(n×d_v)

  S = Q · Kᵀ / √d_k  ∈ ℝ^(n×n)
  A = softmax(S, axis=−1)  ∈ ℝ^(n×n)
  Output = A · V     ∈ ℝ^(n×d_v)

(Module 8 — every intermediate matrix is displayed.)

**Why divide by √d_k?** If q, k have entries with mean 0 and variance 1, the dot product q · k
has variance d_k. For large d_k, softmax saturates and gradients vanish. Dividing by √d_k keeps
the variance ≈ 1.

### 7.2 Multi-Head Attention

Split d_model into h heads, each with d_k = d_model / h:

  head_i = Attention(X · W_Q_i, X · W_K_i, X · W_V_i)
  MultiHead(X) = Concat(head_1, …, head_h) · W_O

Different heads learn different relationships (syntactic, semantic, positional). The concat is
projected back to d_model via W_O ∈ ℝ^(h·d_v × d_model) (Module 9).

### 7.3 Cross-Modal Attention

Q comes from one modality (text), K and V from another (image):

  Q_text = X_text · W_Q
  K_image = X_image · W_K
  V_image = X_image · W_V

  Attention = softmax(Q_text · K_imageᵀ / √d_k) · V_image

The attention matrix is (n_text × n_image): each text token attends to image patches. This is how
a language model "looks at" an image (Module 15).

### 7.4 Causal Masking (Autoregressive)

To prevent a token from attending to future tokens, set S[i, j] = −∞ for j > i before softmax.
After softmax, those weights become 0. Used inside decoder-only language models.

## 8. Embeddings

### 8.1 Token Embeddings

A vocabulary V is mapped to vectors of dimension d via an embedding matrix E ∈ ℝ^(V×d). Token id t
is looked up: x = E[t] ∈ ℝ^d. This is a learnable lookup table; gradient descent updates E.

### 8.2 Patch Embeddings

For Vision Transformers, each patch is a vector x_p ∈ ℝ^(P²C). A linear projection maps it to the
embedding space:

  z = x_p · W_E + b, where W_E ∈ ℝ^(P²C × d), b ∈ ℝ^d

(Module 5.) This is mathematically equivalent to a single linear layer on flattened patches.

### 8.3 Sinusoidal Positional Embeddings

Because attention is permutation-invariant, transformers add positional information:

  PE(pos, 2i)   = sin(pos / 10000^(2i/d))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

  Z = PatchEmbedding + PE    (element-wise add)

(Module 6.) The sinusoidal form has a nice property: PE(pos+k) is a linear function of PE(pos),
so the model can learn to attend to relative positions.

Modern transformers often use learnable positional embeddings (a matrix P ∈ ℝ^(max_len×d) trained
from scratch) or rotary embeddings (RoPE) which rotate Q and K by a position-dependent angle.

## 9. Vision-Language Model Architecture

### 9.1 General Pipeline

  IMAGE → VISION ENCODER → VISUAL TOKENS → PROJECTOR → LANGUAGE MODEL → LOGITS → SOFTMAX → TEXT

(Module 13.) Each component has variants:

| Component | Variants |
|---|---|
| Vision encoder | CNN (ResNet), ViT, CLIP image encoder |
| Projector | Linear, MLP, Q-Former (BLIP-2), Resampler (Flamingo) |
| Language model | Decoder-only (LLaMA), Encoder-decoder (T5), masked LM |
| Fusion | Prefix (LLaVA), cross-attention (Flamingo), Q-Former tokens (BLIP-2) |

### 9.2 Vision-to-Language Projector

The visual representation Z_visual ∈ ℝ^(N×d_v) lives in the vision encoder's space. The language
model expects inputs in ℝ^d_LM. A projector aligns them:

  Z_projected = Z_visual · W_P + b, where W_P ∈ ℝ^(d_v × d_LM), b ∈ ℝ^d_LM

(Module 14.) This is a single linear layer (LLaVA) or an MLP (two linear layers + GELU). Q-Former
(BLIP-2) additionally uses learnable query tokens that attend to image features.

### 9.3 Dual-Encoder Alignment (CLIP)

CLIP trains two independent encoders — image and text — to produce vectors in a shared d-dimensional
space. Image I and text T are compared by cosine similarity:

  sim(I, T) = (I · T) / (‖I‖ ‖T‖)

Training uses InfoNCE contrastive loss (Module 11 + 12). The result is a model that can rank text
descriptions by image relevance, without ever generating text.

## 10. Autoregressive Generation

### 10.1 Joint Probability

A sequence y_1, …, y_T given input X is generated token-by-token:

  P(y_1, …, y_T | X) = Πₜ P(y_t | y_<t, X)

Each step conditions on all previous tokens + the image. This is **autoregressive** decoding.

### 10.2 Greedy Decoding

At each step, pick the token with the highest probability:

  y_t = argmax_t P(y_t | y_<t, X)

Simple but prone to repetition and local optima.

### 10.3 Temperature Sampling

  P(y_t = i) = exp(z_i / τ) / Σⱼ exp(z_j / τ)

- τ < 1 → sharper (more deterministic)
- τ > 1 → flatter (more diverse)
- τ = 0 → greedy

### 10.4 Top-k Sampling

Keep only the k highest-probability tokens, renormalize, and sample. Reduces the chance of
low-probability "garbage" tokens.

### 10.5 Top-p (Nucleus) Sampling

Keep the smallest set of tokens whose cumulative probability ≥ p (e.g. p = 0.9). Adapts to the
shape of the distribution: if one token dominates, only 1-2 are kept; if many are tied, more are kept.

(Module 17 implements all four strategies on a small synthetic example.)

## 11. How These Concepts Connect

```
  Image (Module 1: tensor X ∈ ℝ^(H×W×3))
    │
    ├── [CNN path] Conv + ReLU + Pool (Modules 2, 3)
    │   → feature maps → flatten → image embedding
    │
    └── [ViT path] Patches (Module 4)
         → Patch Embeddings (Module 5)
         → + Positional Embeddings (Module 6)
         → Vision Transformer = Self-Attention + MLP (Modules 7, 8, 9)
         → Visual tokens

  Visual tokens ─┐
                 ├── Projector (Module 14) → Multimodal representation
  Text tokens  ─┘    (or Cross-Attention, Module 15)

  Multimodal representation
    → Language Model (Module 13)
    → Logits over vocabulary (Module 16)
    → Softmax → probabilities
    → Autoregressive sampling (Module 17)
    → Generated text

  Training signal (background):
    Contrastive loss (Module 12, CLIP) for alignment
    Cross-entropy loss (Section 5.2) for next-token prediction
```

## 12. Educational Simplifications

This is a **simplified educational representation**. Real production VLMs differ in several ways:

- **Scale**: real ViTs use d_model = 768–1024 and 12+ layers; we use d = 2–8.
- **Tokenization**: real tokenizers use BPE or WordPiece (50k+ vocab); we use whitespace splitting
  on a 10-word vocab.
- **Positional encoding**: real models use learnable PE or RoPE; we use the original sinusoidal form
  for pedagogical clarity.
- **Normalization**: real models add LayerNorm, dropout, and stochastic depth.
- **Optimization**: real training uses AdamW with cosine LR schedules and warmup; we don't train at all.
- **Architecture**: BLIP-2's Q-Former, Flamingo's perceiver resampler, LLaVA's MLP projector are
  more complex than our single-linear projector.

Every module flags where simplifications are made. The math, however, is real: every matrix multiply,
every softmax, every attention weight is computed explicitly from the definitions above.

## 13. References

- Vaswani et al. (2017). *Attention Is All You Need.* — original Transformer.
- Dosovitskiy et al. (2021). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* — ViT.
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* — CLIP.
- Li et al. (2023). *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.*
- Liu et al. (2023). *Visual Instruction Tuning.* — LLaVA.
- Goodfellow, Bengio, Courville. *Deep Learning.* — background linear algebra, optimization.

## 14. Viva Preparation

Module 22 (Viva Mode) contains 41 thesis-defense questions across all modules, each with:
- Difficulty (Easy / Medium / Difficult)
- Hint
- Answer
- Mathematical explanation

Sample questions:

- *Why is √d_k used in scaled dot-product attention?* (Module 8)
- *What happens if we remove positional embeddings from a transformer?* (Module 6)
- *Why can image and text embeddings be compared using cosine similarity?* (Module 11)
- *What is the mathematical purpose of the projector in a VLM?* (Module 14)
- *How does a language model convert logits into probabilities?* (Module 16)

Use Viva Mode in **presentation mode** (full-screen, large fonts) during your defense.

## License

This is an educational project. Use freely for academic purposes. Cite the original papers (Section 13)
for the underlying mathematics.
