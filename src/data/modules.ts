/**
 * Catalog of all modules in the application.
 * Used by the sidebar for navigation, and by Viva Mode to filter questions.
 */
export interface ModuleMeta {
  id: string;
  number: number;
  title: string;
  shortTitle: string;
  category: "image" | "vision" | "attention" | "language" | "alignment" | "pipeline" | "tools";
  description: string;
}

export const MODULES: ModuleMeta[] = [
  {
    id: "image-repr",
    number: 1,
    title: "Mathematical Representation of an Image",
    shortTitle: "Image → Tensor",
    category: "image",
    description: "Pixel, RGB channels, image as a tensor X ∈ ℝ^(H×W×3).",
  },
  {
    id: "convolution",
    number: 2,
    title: "Convolution",
    shortTitle: "Convolution",
    category: "image",
    description: "Sliding kernel, element-wise multiply, sum, feature maps.",
  },
  {
    id: "cnn",
    number: 3,
    title: "CNN Feature Extraction",
    shortTitle: "CNN Pipeline",
    category: "image",
    description: "Conv → ReLU → Pool, hierarchical features.",
  },
  {
    id: "patches",
    number: 4,
    title: "Image Patches",
    shortTitle: "Patches",
    category: "vision",
    description: "Splitting an image into N patches of size P×P.",
  },
  {
    id: "patch-embed",
    number: 5,
    title: "Patch Embeddings",
    shortTitle: "Patch Embed",
    category: "vision",
    description: "Linear projection: x_p ∈ ℝ^(P²C) → z ∈ ℝ^d.",
  },
  {
    id: "positional",
    number: 6,
    title: "Positional Embeddings",
    shortTitle: "Position",
    category: "vision",
    description: "Sinusoidal PE: sin(pos/10000^(2i/d)).",
  },
  {
    id: "vit",
    number: 7,
    title: "Vision Transformer",
    shortTitle: "ViT",
    category: "vision",
    description: "Patches + attention encoder = visual representation.",
  },
  {
    id: "self-attention",
    number: 8,
    title: "Self-Attention",
    shortTitle: "Self-Attention",
    category: "attention",
    description: "Q, K, V; softmax(QKᵀ/√d_k)V — the heart of transformers.",
  },
  {
    id: "multihead",
    number: 9,
    title: "Multi-Head Attention",
    shortTitle: "Multi-Head",
    category: "attention",
    description: "Concat(head₁,...,head_h) W_O.",
  },
  {
    id: "text-repr",
    number: 10,
    title: "Text Representation",
    shortTitle: "Text → Tokens",
    category: "language",
    description: "Tokens → IDs → embeddings.",
  },
  {
    id: "alignment",
    number: 11,
    title: "Image-Text Alignment (CLIP)",
    shortTitle: "Alignment",
    category: "alignment",
    description: "Cosine similarity in a shared embedding space.",
  },
  {
    id: "contrastive",
    number: 12,
    title: "Contrastive Learning",
    shortTitle: "Contrastive",
    category: "alignment",
    description: "InfoNCE loss, temperature, similarity matrix.",
  },
  {
    id: "architecture",
    number: 13,
    title: "VLM Architecture",
    shortTitle: "Architecture",
    category: "pipeline",
    description: "Encoder → Projector → Language Model.",
  },
  {
    id: "projector",
    number: 14,
    title: "Vision-to-Language Projector",
    shortTitle: "Projector",
    category: "pipeline",
    description: "Z_visual W_P + b: dimensional transformation.",
  },
  {
    id: "cross-attention",
    number: 15,
    title: "Cross-Modal Attention",
    shortTitle: "Cross-Attn",
    category: "attention",
    description: "Q from text, K/V from image.",
  },
  {
    id: "next-token",
    number: 16,
    title: "Next Token Prediction",
    shortTitle: "Softmax",
    category: "language",
    description: "Logits → softmax → probability distribution.",
  },
  {
    id: "autoregressive",
    number: 17,
    title: "Autoregressive Generation",
    shortTitle: "Generation",
    category: "language",
    description: "P(y₁,...,y_T|X) = Π P(y_t | y_<t, X).",
  },
  {
    id: "pipeline",
    number: 18,
    title: "End-to-End Pipeline",
    shortTitle: "Pipeline",
    category: "pipeline",
    description: "Image → ... → generated text.",
  },
  {
    id: "sandbox",
    number: 19,
    title: "Mathematical Sandbox",
    shortTitle: "Sandbox",
    category: "tools",
    description: "Manual matrix operations calculator.",
  },
  {
    id: "dimension-tracker",
    number: 20,
    title: "Dimension Tracker",
    shortTitle: "Dims",
    category: "tools",
    description: "Automatic dimension tracking through every operation.",
  },
  {
    id: "thesis-demo",
    number: 21,
    title: "Thesis Demo",
    shortTitle: "Demo",
    category: "pipeline",
    description: "Full story: image → math → language.",
  },
  {
    id: "viva",
    number: 22,
    title: "Viva Mode",
    shortTitle: "Viva",
    category: "tools",
    description: "Question bank for thesis defense.",
  },
];

export const CATEGORY_LABELS: Record<ModuleMeta["category"], string> = {
  image: "Image Foundations",
  vision: "Vision Pipeline",
  attention: "Attention",
  language: "Language Model",
  alignment: "Alignment",
  pipeline: "Architecture & Pipeline",
  tools: "Tools",
};
