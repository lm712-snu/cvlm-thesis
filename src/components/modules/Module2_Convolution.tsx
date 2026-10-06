"use client";

import { useState, useMemo, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight, Play, RotateCcw } from "lucide-react";
import { MatrixView } from "@/components/cvlm/MatrixView";
import { Heatmap } from "@/components/cvlm/Heatmap";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import { conv2d, KERNELS, type ConvStep } from "@/lib/math/convolution";
import { randMatrix, shape } from "@/lib/math/matrix";
import { useAppStore } from "@/store/app-store";

const KERNEL_NAMES = Object.keys(KERNELS);

// 5x5 sample image used in the educational demo (deterministic, easy to read).
const SAMPLE_5x5: number[][] = [
  [10, 20, 30, 40, 50],
  [20, 30, 40, 50, 60],
  [30, 40, 50, 60, 70],
  [40, 50, 60, 70, 80],
  [50, 60, 70, 80, 90],
];

export function Module2_Convolution() {
  const sharedImage = useAppStore((s) => s.sharedImage);
  const [kernelName, setKernelName] = useState<string>("sobelX");
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState(0);
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);

  // Source matrix: use uploaded image's grayscale (down-sampled to <= 8x8 for readability) or the sample 5x5.
  const source = useMemo(() => {
    if (sharedImage && sharedImage.gray.length <= 8 && sharedImage.gray[0].length <= 8) {
      return sharedImage.gray;
    }
    return SAMPLE_5x5;
  }, [sharedImage]);

  const kernel = KERNELS[kernelName];
  const result = useMemo(
    () => conv2d(source, kernel, 0, { stride, padding }),
    [source, kernel, stride, padding]
  );
  const steps = result.steps;
  const currentStep: ConvStep | undefined = steps[stepIdx];
  const [H, W] = shape(source);
  const [kH, kW] = shape(kernel);
  const outH = result.outH;
  const outW = result.outW;

  // Auto-advance when playing.
  useEffect(() => {
    if (!playing) return;
    if (stepIdx >= steps.length - 1) {
      const id = setTimeout(() => setPlaying(false), 0);
      return () => clearTimeout(id);
    }
    const t = setTimeout(() => setStepIdx((i) => i + 1), 600);
    return () => clearTimeout(t);
  }, [playing, stepIdx, steps.length]);

  return (
    <div>
      <ModuleHeader
        number={2}
        title="Convolution"
        subtitle="Sliding a kernel across an image; element-wise multiply + sum = feature map."
      >
        <MathBlock block>
          {`Y_{ij} = \\sum_{m=0}^{k-1} \\sum_{n=0}^{k-1} X_{i+m,\\, j+n} \\cdot K_{m,n} + b`}
        </MathBlock>
      </ModuleHeader>

      <div className="grid gap-4 lg:grid-cols-[300px_1fr]">
        <Card className="h-fit">
          <CardHeader><CardTitle className="text-base">Controls</CardTitle></CardHeader>
          <CardContent className="space-y-4 text-sm">
            <div>
              <Label className="text-xs">Kernel</Label>
              <Select value={kernelName} onValueChange={(v) => { setKernelName(v); setStepIdx(0); }}>
                <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {KERNEL_NAMES.map((k) => (
                    <SelectItem key={k} value={k}>{k}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <div className="flex justify-between"><Label className="text-xs">Stride</Label><span className="font-mono text-xs">{stride}</span></div>
              <Slider value={[stride]} min={1} max={3} onValueChange={([v]) => { setStride(v ?? 1); setStepIdx(0); }} />
            </div>
            <div>
              <div className="flex justify-between"><Label className="text-xs">Padding</Label><span className="font-mono text-xs">{padding}</span></div>
              <Slider value={[padding]} min={0} max={3} onValueChange={([v]) => { setPadding(v ?? 0); setStepIdx(0); }} />
            </div>
            <div className="rounded border bg-muted/30 p-2 text-xs space-y-1">
              <div className="flex justify-between"><span>Input</span><DimBadge dims={`${H} × ${W}`} variant="input" /></div>
              <div className="flex justify-between"><span>Kernel</span><DimBadge dims={`${kH} × ${kW}`} variant="weight" /></div>
              <div className="flex justify-between"><span>Output</span><DimBadge dims={`${outH} × ${outW}`} variant="output" /></div>
            </div>
            <div className="rounded border bg-muted/30 p-2 text-xs">
              <p className="font-mono">out = ⌊(H + 2·p − k) / s⌋ + 1</p>
              <p className="text-muted-foreground mt-1">
                = ⌊({H} + {2 * padding} − {kH}) / {stride}⌋ + 1 = {outH}
              </p>
            </div>
            <p className="text-[10px] text-muted-foreground">
              {sharedImage
                ? "Using your uploaded image (grayscale, small dims)."
                : "Using a built-in 5×5 demo image. Upload an image ≤ 64×64 in Module 1 to use it here."}
            </p>
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2 flex flex-row items-center justify-between">
              <CardTitle className="text-sm">Sliding Window Animation</CardTitle>
              <div className="flex gap-1">
                <Button size="sm" variant="ghost" onClick={() => setStepIdx(0)} title="Restart">
                  <RotateCcw className="h-3.5 w-3.5" />
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setStepIdx((i) => Math.max(0, i - 1))}>
                  <ChevronLeft className="h-3.5 w-3.5" />
                </Button>
                <Button size="sm" variant={playing ? "default" : "outline"} onClick={() => setPlaying((p) => !p)}>
                  <Play className="h-3.5 w-3.5" /> {playing ? "Pause" : "Play"}
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setStepIdx((i) => Math.min(steps.length - 1, i + 1))}>
                  <ChevronRight className="h-3.5 w-3.5" />
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-3">
                {/* Input with sliding window highlighted */}
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">Input ({H}×{W})</p>
                  <div className="relative inline-block">
                    <MatrixView
                      matrix={source}
                      digits={0}
                      cellSize="sm"
                      highlight={
                        currentStep
                          ? // highlight the 3x3 region centered around the window
                            [
                              Math.min(H - 1, Math.max(0, currentStep.outRow + (padding > 0 ? 0 : 0))),
                              Math.min(W - 1, Math.max(0, currentStep.outCol + (padding > 0 ? 0 : 0))),
                            ]
                          : undefined
                      }
                    />
                    {/* Overlay the window as a rectangle */}
                    {currentStep && (
                      <div
                        className="pointer-events-none absolute border-2 border-primary"
                        style={{
                          left: `${currentStep.outCol * 36 + 1}px`,
                          top: `${currentStep.outRow * 32 + 1}px`,
                          width: `${kW * 36}px`,
                          height: `${kH * 32}px`,
                        }}
                      />
                    )}
                  </div>
                </div>
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">Kernel ({kH}×{kW})</p>
                  <MatrixView matrix={kernel} digits={2} heatmap diverging />
                </div>
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">Output ({outH}×{outW})</p>
                  <MatrixView
                    matrix={result.output}
                    digits={2}
                    heatmap
                    diverging
                    highlight={currentStep ? [currentStep.outRow, currentStep.outCol] : undefined}
                  />
                </div>
              </div>

              {currentStep && (
                <div className="mt-4 rounded border bg-muted/30 p-3 text-xs">
                  <p className="font-mono">
                    Y[{currentStep.outRow}, {currentStep.outCol}] = Σ X · K ={" "}
                    <span className="font-bold text-primary">{currentStep.sum.toFixed(3)}</span>
                  </p>
                  <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-3">
                    {currentStep.products.flatMap((row, m) =>
                      row.map((v, n) => (
                        <div key={`${m}-${n}`} className="rounded bg-background p-1 text-center font-mono">
                          {currentStep.window[m][n].toFixed(0)} × {kernel[m][n].toFixed(2)} = <span className="text-primary">{v.toFixed(2)}</span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}
              <p className="mt-2 text-[10px] text-muted-foreground">
                Step {stepIdx + 1} / {steps.length}. Click ▶ to animate the kernel sliding across the image.
              </p>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <ExpandableSection title="Stride and padding: how do they change the output size?" variant="how" defaultOpen>
              <p>
                <strong>Stride</strong> is how many pixels the kernel moves per step. Stride 2 halves the output
                size. <strong>Padding</strong> adds zeros around the input so the kernel can visit edge positions.
              </p>
              <MathBlock block>{`\\text{out} = \\left\\lfloor \\frac{H + 2p - k}{s} \\right\\rfloor + 1`}</MathBlock>
              <p>
                With <code>H=5, k=3, p=0, s=1</code>: <code>out = (5 − 3)/1 + 1 = 3</code>.
                With <code>p=1</code>: <code>out = (5 + 2 − 3)/1 + 1 = 5</code> (same size).
              </p>
            </ExpandableSection>

            <ExpandableSection title="Why are Sobel kernels useful for edges?" variant="why">
              <p>
                Sobel-X is <code>[-1,0,1; -2,0,2; -1,0,1]</code>. It computes a finite-difference approximation of the
                horizontal gradient ∂X/∂x. Large output = sharp intensity change in x = vertical edge.
              </p>
              <p className="mt-2">
                Stacking Sobel-X and Sobel-Y gives the gradient magnitude{" "}
                <MathBlock>{`|\\nabla X| = \\sqrt{(S_x)^2 + (S_y)^2}`}</MathBlock> — the basis of classical edge detection.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Multiple filters → multiple channels" variant="math">
              <p>
                A single kernel produces one feature map. Applying F kernels produces F stacked maps — i.e. the
                output is now <MathBlock>{`Y \\in \\mathbb{R}^{H' \\times W' \\times F}`}</MathBlock>.
              </p>
              <p className="mt-2">
                The weight tensor is then <MathBlock>{`W \\in \\mathbb{R}^{k \\times k \\times C_{in} \\times C_{out}}`}</MathBlock>.
                In modern CNNs, the first layer might have <code>3 → 64</code> channels (RGB → 64 feature maps).
              </p>
            </ExpandableSection>

            <ExpandableSection title="Receptive field" variant="intuition">
              <p>
                Each output pixel "sees" a k×k region of the input. After stacking two 3×3 convolutions, each output
                pixel effectively sees a 5×5 region — this is the <strong>receptive field</strong>.
              </p>
              <p className="mt-2">
                Deeper layers see larger regions, which is why deep CNNs can recognize whole objects even though
                each convolution is local. Hierarchical features emerge: pixels → edges → textures → parts → objects.
              </p>
            </ExpandableSection>
          </div>

          <VivaPanel
            questions={[
              {
                level: "Easy",
                question: "What is a kernel (filter) in convolution?",
                hint: "It is a small matrix of learnable numbers.",
                answer: "A small matrix K ∈ ℝ^(k×k) of learnable weights that slides over the input, computing a weighted sum at each position.",
                explanation: "Different kernels detect different features (edges, blurs, color gradients). CNNs learn these kernels automatically from data.",
              },
              {
                level: "Medium",
                question: "Why is convolution a good inductive bias for images?",
                hint: "Think locality and translation.",
                answer:
                  "Convolution exploits two properties of natural images: locality (nearby pixels are correlated) and translation invariance (a cat is a cat regardless of where it appears). The same kernel is applied at every position, sharing weights.",
                explanation:
                  "A fully-connected layer treating pixels as independent features would need O(H·W·C) parameters per layer and would not generalize across positions. Convolution shares weights spatially, drastically reducing parameters and encoding the right prior.",
              },
              {
                level: "Difficult",
                question:
                  "If you stack two 3×3 convolutions with stride 1, what is the receptive field of an output pixel? Compare with a single 5×5 convolution.",
                hint: "Draw the overlap.",
                answer:
                  "Two stacked 3×3 convs give a 5×5 receptive field — same as one 5×5 conv — but use 2·9 = 18 parameters instead of 25, and add an extra nonlinearity (ReLU) between them.",
                explanation:
                  "This is why VGG and modern CNNs prefer stacks of small 3×3 kernels: same effective receptive field, fewer parameters, more nonlinearity, better optimization.",
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
}
