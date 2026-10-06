/**
 * Attention utilities — explicit, no torch.nn.MultiheadAttention.
 * All intermediate matrices (Q, K, V, scores, weights, output) are returned for visualization.
 */
import type { Matrix, Vector } from "./matrix";
import { matmul, zeros, shape, transpose } from "./matrix";
import { softmaxRows } from "./matrix";

export interface AttentionResult {
  Q: Matrix;
  K: Matrix;
  V: Matrix;
  scores: Matrix; // QK^T / sqrt(d_k)
  weights: Matrix; // softmax(scores)
  output: Matrix; // weights @ V
  dK: number;
}

/**
 * Scaled dot-product self-attention.
 *
 *  Q = X W_Q         (n x d_k)
 *  K = X W_K         (n x d_k)
 *  V = X W_V         (n x d_v)
 *  S = Q K^T / sqrt(d_k)   (n x n)
 *  A = softmax(S, axis=-1) (n x n)
 *  Out = A V              (n x d_v)
 *
 * @param X       input sequence of shape (n, d_model)
 * @param W_Q     (d_model, d_k)
 * @param W_K     (d_model, d_k)
 * @param W_V     (d_model, d_v)
 */
export function selfAttention(X: Matrix, W_Q: Matrix, W_K: Matrix, W_V: Matrix): AttentionResult {
  const Q = matmul(X, W_Q);
  const K = matmul(X, W_K);
  const V = matmul(X, W_V);
  const [n, dK] = shape(Q);
  const scale = Math.sqrt(dK);
  const Kt = transpose(K);
  const scores = matmul(Q, Kt).map((row) => row.map((v) => v / scale));
  const weights = softmaxRows(scores);
  const output = matmul(weights, V);
  return { Q, K, V, scores, weights, output, dK };
}

/**
 * Cross-modal attention: queries come from text, keys/values come from image.
 *
 *  Q = X_text W_Q      (n_text x d_k)
 *  K = X_image W_K     (n_image x d_k)
 *  V = X_image W_V     (n_image x d_v)
 *  S = Q K^T / sqrt(d_k)   (n_text x n_image)
 *  Out = softmax(S) V      (n_text x d_v)
 */
export function crossAttention(
  X_text: Matrix,
  X_image: Matrix,
  W_Q: Matrix,
  W_K: Matrix,
  W_V: Matrix
): AttentionResult {
  const Q = matmul(X_text, W_Q);
  const K = matmul(X_image, W_K);
  const V = matmul(X_image, W_V);
  const [_, dK] = shape(Q);
  const scale = Math.sqrt(dK);
  const scores = matmul(Q, transpose(K)).map((row) => row.map((v) => v / scale));
  const weights = softmaxRows(scores);
  const output = matmul(weights, V);
  return { Q, K, V, scores, weights, output, dK };
}

export interface MultiHeadResult {
  heads: AttentionResult[];
  concat: Matrix; // concatenated head outputs (n x h*d_v)
  output: Matrix; // concat @ W_O  (n x d_model)
  W_O: Matrix;
}

/**
 * Multi-head attention:
 *
 *  head_i = Attention(X W_Q_i, X W_K_i, X W_V_i)
 *  MultiHead(X) = Concat(head_1, ..., head_h) W_O
 *
 * d_model is split evenly across heads (d_k = d_v = d_model / h).
 */
export function multiHeadAttention(
  X: Matrix,
  W_Qs: Matrix[], // each (d_model x d_k)
  W_Ks: Matrix[],
  W_Vs: Matrix[],
  W_O: Matrix // (h*d_v x d_model)
): MultiHeadResult {
  const heads = W_Qs.map((Wq, i) => selfAttention(X, Wq, W_Ks[i], W_Vs[i]));
  const [n] = shape(X);
  const h = heads.length;
  const dv = shape(heads[0].V)[1];

  // concatenate along the feature axis
  const concat = zeros(n, h * dv);
  for (let i = 0; i < n; i++) {
    let col = 0;
    for (let hd = 0; hd < h; hd++) {
      for (let k = 0; k < dv; k++) {
        concat[i][col++] = heads[hd].output[i][k];
      }
    }
  }
  const output = matmul(concat, W_O);
  return { heads, concat, output, W_O };
}

/**
 * Sinusoidal positional encoding (Transformer original paper).
 *
 *  PE(pos, 2i)   = sin(pos / 10000^(2i/d))
 *  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
 *
 * Returns (maxLen x d) matrix.
 */
export function sinusoidalPE(maxLen: number, d: number): Matrix {
  const pe = zeros(maxLen, d);
  for (let pos = 0; pos < maxLen; pos++) {
    for (let i = 0; i < Math.floor(d / 2); i++) {
      const denom = Math.pow(10000, (2 * i) / d);
      pe[pos][2 * i] = Math.sin(pos / denom);
      if (2 * i + 1 < d) pe[pos][2 * i + 1] = Math.cos(pos / denom);
    }
  }
  return pe;
}

/** Cross-entropy loss given predicted probabilities and a target index. */
export function crossEntropy(probs: Vector, targetIdx: number): number {
  const p = Math.max(probs[targetIdx], 1e-12);
  return -Math.log(p);
}

/** Cross-entropy with logits (numerically stable): -log(softmax(logits)[target]). */
export function crossEntropyFromLogits(logits: Vector, targetIdx: number): number {
  const mx = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - mx));
  const s = exps.reduce((a, b) => a + b, 0);
  const p = exps[targetIdx] / s;
  return -Math.log(Math.max(p, 1e-12));
}

/** InfoNCE loss for contrastive learning, given a (B x B) similarity matrix and temperature τ. */
export function infoNCE(sim: Matrix, temperature: number): { loss: number; rowProbs: Matrix } {
  const [B, B2] = shape(sim);
  if (B !== B2) throw new Error("sim must be square");
  // treat diagonal as positive pairs
  const scaled = sim.map((r) => r.map((v) => v / temperature));
  const rowProbs = softmaxRows(scaled);
  let total = 0;
  for (let i = 0; i < B; i++) total += -Math.log(Math.max(rowProbs[i][i], 1e-12));
  return { loss: total / B, rowProbs };
}

/** Greedy decoding: pick argmax of logits. */
export function argmax(v: Vector): number {
  let mi = 0;
  let mv = -Infinity;
  for (let i = 0; i < v.length; i++) {
    if (v[i] > mv) {
      mv = v[i];
      mi = i;
    }
  }
  return mi;
}

/** Top-k filtering: keep top-k logits, set others to -Infinity. */
export function topKMask(v: Vector, k: number): Vector {
  const idx = v.map((x, i) => [x, i] as [number, number]).sort((a, b) => b[0] - a[0]);
  const keep = new Set(idx.slice(0, k).map(([, i]) => i));
  return v.map((x, i) => (keep.has(i) ? x : -Infinity));
}

/** Nucleus (top-p) filtering: keep the smallest set whose cumulative prob >= p. */
export function topPMask(probs: Vector, p: number): Vector {
  const idx = probs.map((x, i) => [x, i] as [number, number]).sort((a, b) => b[0] - a[0]);
  let cum = 0;
  const keep = new Set<number>();
  for (const [prob, i] of idx) {
    keep.add(i);
    cum += prob;
    if (cum >= p) break;
  }
  return probs.map((x, i) => (keep.has(i) ? x : 0));
}
