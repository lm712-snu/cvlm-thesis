/**
 * 2D convolution utilities — explicit, educational.
 * Demonstrates: sliding window, element-wise multiply, sum, stride, padding.
 */
import type { Matrix } from "./matrix";
import { zeros, shape } from "./matrix";

export interface ConvOptions {
  stride?: number;
  padding?: number;
}

export interface ConvStep {
  outRow: number;
  outCol: number;
  window: Matrix; // the input slice
  products: Matrix; // window .* kernel
  sum: number; // the conv output for this position
}

/** Pad a matrix with `p` zeros on every side. */
export function pad(A: Matrix, p: number): Matrix {
  const [m, n] = shape(A);
  const M = m + 2 * p;
  const N = n + 2 * p;
  const out = zeros(M, N);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[i + p][j + p] = A[i][j];
  return out;
}

/**
 * 2D convolution (cross-correlation form, as used in CNNs).
 * Output(i,j) = Σ_m Σ_n X(i+m, j+n) * K(m,n) + b
 *
 * Returns the output feature map AND the step-by-step trace (for visualization).
 */
export function conv2d(
  input: Matrix,
  kernel: Matrix,
  bias = 0,
  opts: ConvOptions = {}
): { output: Matrix; steps: ConvStep[]; outH: number; outW: number } {
  const stride = opts.stride ?? 1;
  const padding = opts.padding ?? 0;
  const padded = padding > 0 ? pad(input, padding) : input;
  const [H, W] = shape(padded);
  const [kh, kw] = shape(kernel);
  if (kh !== kw) throw new Error("Kernel must be square");
  const k = kh;

  const outH = Math.floor((H - k) / stride) + 1;
  const outW = Math.floor((W - k) / stride) + 1;
  const output = zeros(outH, outW);
  const steps: ConvStep[] = [];

  for (let i = 0; i < outH; i++) {
    for (let j = 0; j < outW; j++) {
      const window = zeros(k, k);
      const products = zeros(k, k);
      let sum = 0;
      for (let m = 0; m < k; m++) {
        for (let n = 0; n < k; n++) {
          const v = padded[i * stride + m][j * stride + n];
          window[m][n] = v;
          products[m][n] = v * kernel[m][n];
          sum += v * kernel[m][n];
        }
      }
      sum += bias;
      output[i][j] = sum;
      steps.push({ outRow: i, outCol: j, window, products, sum });
    }
  }
  return { output, steps, outH, outW };
}

/** 2x2 max pool with stride 2. Returns pooled matrix and the argmax indices (for visualization). */
export function maxPool2x2(input: Matrix): { output: Matrix; indices: [number, number][][] } {
  const [H, W] = shape(input);
  const outH = Math.floor(H / 2);
  const outW = Math.floor(W / 2);
  const output = zeros(outH, outW);
  const indices: [number, number][][] = Array.from({ length: outH }, () =>
    Array.from({ length: outW }, () => [0, 0] as [number, number])
  );
  for (let i = 0; i < outH; i++) {
    for (let j = 0; j < outW; j++) {
      let mx = -Infinity;
      let mi = 0;
      let mj = 0;
      for (let m = 0; m < 2; m++) {
        for (let n = 0; n < 2; n++) {
          const v = input[2 * i + m][2 * j + n];
          if (v > mx) {
            mx = v;
            mi = 2 * i + m;
            mj = 2 * j + n;
          }
        }
      }
      output[i][j] = mx;
      indices[i][j] = [mi, mj];
    }
  }
  return { output, indices };
}

/** 2x2 average pool with stride 2. */
export function avgPool2x2(input: Matrix): Matrix {
  const [H, W] = shape(input);
  const outH = Math.floor(H / 2);
  const outW = Math.floor(W / 2);
  const output = zeros(outH, outW);
  for (let i = 0; i < outH; i++) {
    for (let j = 0; j < outW; j++) {
      let sum = 0;
      for (let m = 0; m < 2; m++) for (let n = 0; n < 2; n++) sum += input[2 * i + m][2 * j + n];
      output[i][j] = sum / 4;
    }
  }
  return output;
}

/** Common 3x3 kernels for demonstration. */
export const KERNELS: Record<string, Matrix> = {
  identity: [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
  ],
  sobelX: [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
  ],
  sobelY: [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1],
  ],
  edge: [
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1],
  ],
  sharpen: [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
  ],
  blur: [
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
  ],
  gaussian: [
    [1 / 16, 2 / 16, 1 / 16],
    [2 / 16, 4 / 16, 2 / 16],
    [1 / 16, 2 / 16, 1 / 16],
  ],
  emboss: [
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2],
  ],
};
