/**
 * Matrix / linear algebra utilities.
 * All operations are explicit and educational — no hidden library calls.
 * Every function documents the dimensions of inputs and outputs.
 */

export type Matrix = number[][];
export type Vector = number[];

/** Create an m x n matrix filled with a value (default 0). */
export function zeros(m: number, n: number, fill = 0): Matrix {
  return Array.from({ length: m }, () => Array.from({ length: n }, () => fill));
}

/** Create an m x n matrix filled with random values in [min, max]. */
export function randMatrix(m: number, n: number, min = -1, max = 1, seed?: number): Matrix {
  const rng = seed !== undefined ? mulberry32(seed) : Math.random;
  return Array.from({ length: m }, () =>
    Array.from({ length: n }, () => rng() * (max - min) + min)
  );
}

/** Deterministic PRNG so demos are reproducible across renders. */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Return matrix dimensions [rows, cols]. */
export function shape(A: Matrix): [number, number] {
  return [A.length, A[0]?.length ?? 0];
}

/** Transpose A (m x n) -> (n x m). */
export function transpose(A: Matrix): Matrix {
  const [m, n] = shape(A);
  const out = zeros(n, m);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[j][i] = A[i][j];
  return out;
}

/** Matrix multiply: A (m x k) * B (k x n) -> (m x n). Throws on dimension mismatch. */
export function matmul(A: Matrix, B: Matrix): Matrix {
  const [m, k1] = shape(A);
  const [k2, n] = shape(B);
  if (k1 !== k2) {
    throw new Error(`Dimension mismatch: A is ${m}x${k1} but B is ${k2}x${n}`);
  }
  const out = zeros(m, n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < k1; k++) s += A[i][k] * B[k][j];
      out[i][j] = s;
    }
  }
  return out;
}

/** Element-wise addition; broadcasts bias vector across rows. */
export function addBias(A: Matrix, b: Vector): Matrix {
  const [m, n] = shape(A);
  if (b.length !== n) throw new Error(`Bias length ${b.length} != cols ${n}`);
  return A.map((row) => row.map((v, j) => v + b[j]));
}

/** Dot product of two equal-length vectors. */
export function dot(a: Vector, b: Vector): number {
  if (a.length !== b.length) throw new Error(`dot: length mismatch ${a.length} vs ${b.length}`);
  return a.reduce((s, v, i) => s + v * b[i], 0);
}

/** L2 norm of a vector. */
export function norm(a: Vector): number {
  return Math.sqrt(a.reduce((s, v) => s + v * v, 0));
}

/** Cosine similarity between two vectors: (a·b) / (||a|| ||b||). */
export function cosineSimilarity(a: Vector, b: Vector): number {
  const na = norm(a);
  const nb = norm(b);
  if (na === 0 || nb === 0) return 0;
  return dot(a, b) / (na * nb);
}

/** Apply f to every element. */
export function map(A: Matrix, f: (v: number, i: number, j: number) => number): Matrix {
  return A.map((row, i) => row.map((v, j) => f(v, i, j)));
}

/** ReLU(x) = max(0, x), element-wise. */
export function relu(A: Matrix): Matrix {
  return map(A, (v) => Math.max(0, v));
}

/** Element-wise sum across rows -> single vector of length n. */
export function sumRows(A: Matrix): Vector {
  const [m, n] = shape(A);
  const out = new Array(n).fill(0);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[j] += A[i][j];
  return out;
}

/** Row-wise softmax with temperature; numerically stable (subtract row max). */
export function softmaxRows(A: Matrix, temperature = 1): Matrix {
  return A.map((row) => softmaxVec(row, temperature));
}

/** Softmax over a vector with temperature τ: exp(x_i/τ) / Σ exp(x_j/τ). */
export function softmaxVec(v: Vector, temperature = 1): Vector {
  const t = temperature <= 0 ? 1e-9 : temperature;
  const scaled = v.map((x) => x / t);
  const mx = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - mx));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}

/** Layer norm across the last axis (per-row). */
export function layerNorm(A: Matrix, eps = 1e-5): Matrix {
  return A.map((row) => {
    const mean = row.reduce((a, b) => a + b, 0) / row.length;
    const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length;
    const denom = Math.sqrt(variance + eps);
    return row.map((v) => (v - mean) / denom);
  });
}

/** Pretty-print a matrix (for debugging / tooltips). */
export function fmtMatrix(A: Matrix, digits = 3): string {
  return A.map((row) => row.map((v) => v.toFixed(digits)).join("  ")).join("\n");
}

/** Format a vector as a compact string. */
export function fmtVec(v: Vector, digits = 3): string {
  return "[" + v.map((x) => x.toFixed(digits)).join(", ") + "]";
}

/** Flatten matrix row-major into a vector. */
export function flatten(A: Matrix): Vector {
  return A.flat();
}

/** Reshape a vector into an m x n matrix (row-major). */
export function reshape(v: Vector, m: number, n: number): Matrix {
  if (v.length !== m * n) throw new Error(`reshape: ${v.length} != ${m}*${n}`);
  const out = zeros(m, n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[i][j] = v[i * n + j];
  return out;
}
