"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { MatrixView } from "@/components/cvlm/MatrixView";
import { Heatmap } from "@/components/cvlm/Heatmap";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import { conv2d, maxPool2x2, KERNELS } from "@/lib/math/convolution";
import { relu, shape } from "@/lib/math/matrix";
import { useAppStore } from "@/store/app-store";
import type { Matrix } from "@/lib/math/matrix";
import { Layers, ArrowDown, ArrowRight } from "lucide-react";

const KERNEL_NAMES = Object.keys(KERNELS);

/**
 * 8x8 deterministic demo image — chosen large enough that two
 * (Conv 3x3 + ReLU + MaxPool 2x2) blocks still leave a visible 2x2 feature map.
 * Same monotonically increasing pattern as Module 2's 5x5 sample for continuity.
 */
const SAMPLE_8x8: Matrix = [
  [10, 20, 30, 40, 50, 60, 70, 80],
  [20, 30, 40, 50, 60, 70, 80, 90],
  [30, 40, 50, 60, 70, 80, 90, 100],
  [40, 50, 60, 70, 80, 90, 100, 110],
  [50, 60, 70, 80, 90, 100, 110, 120],
  [60, 70, 80, 90, 100, 110, 120, 130],
  [70, 80, 90, 100, 110, 120, 130, 140],
  [80, 90, 100, 110, 120, 130, 140, 150],
];

interface Stage {
  name: string;
  kind: "input" | "conv" | "relu" | "pool" | "feature";
  data: Matrix;
  /** optional kernel that produced this stage (for conv stages, the previous one). */
  kernelName?: string;
  /** short description shown next to the stage. */
  note: string;
}

export function Module3_CNN() {
  const sharedImage = useAppStore((s) => s.sharedImage);
  const [kernelName1, setKernelName1] = useState<string>("sobelX");
  const [kernelName2, setKernelName2] = useState<string>("sobelY");
  const [bias1, setBias1] = useState(0);
  const [bias2, setBias2] = useState(0);
  const [padding, setPadding] = useState(1);

  // Source matrix: prefer shared image if small enough, otherwise fall back to the 8x8 sample.
  const source: Matrix = useMemo(() => {
    if (sharedImage && sharedImage.gray.length >= 8 && sharedImage.gray.length <= 16 &&
        sharedImage.gray[0].length >= 8 && sharedImage.gray[0].length <= 16) {
      return sharedImage.gray;
    }
    return SAMPLE_8x8;
  }, [sharedImage]);

  // Run the full stacked CNN pipeline (Conv1 → ReLU → Pool → Conv2 → ReLU → Pool).
  const stages = useMemo<Stage[]>(() => {
    const k1 = KERNELS[kernelName1];
    const k2 = KERNELS[kernelName2];
    const conv1 = conv2d(source, k1, bias1, { stride: 1, padding });
    const relu1 = relu(conv1.output);
    const pool1 = maxPool2x2(relu1);
    const conv2 = conv2d(pool1.output, k2, bias2, { stride: 1, padding });
    const relu2 = relu(conv2.output);
    const pool2 = maxPool2x2(relu2);
    return [
      { name: "Input Image (X)", kind: "input", data: source, note: "Grayscale pixel intensities." },
      { name: "Conv1 (Z₁ = X * K₁ + b₁)", kind: "conv", data: conv1.output, kernelName: kernelName1, note: "Sliding 3×3 kernel, sum of element-wise products." },
      { name: "ReLU1 (A₁ = max(0, Z₁))", kind: "relu", data: relu1, note: "Negative values clipped to 0 — introduces nonlinearity." },
      { name: "MaxPool1 (P₁)", kind: "pool", data: pool1.output, note: "2×2 max → halves each spatial dimension." },
      { name: "Conv2 (Z₂ = P₁ * K₂ + b₂)", kind: "conv", data: conv2.output, kernelName: kernelName2, note: "Second feature extractor — composes with the first." },
      { name: "ReLU2 (A₂ = max(0, Z₂))", kind: "relu", data: relu2, note: "Nonlinearity after the second conv." },
      { name: "MaxPool2 — Final Feature Map", kind: "feature", data: pool2.output, note: "Final compact representation fed downstream." },
    ];
  }, [source, kernelName1, kernelName2, bias1, bias2, padding]);

  const [H, W] = shape(source);

  return (
    <div>
      <ModuleHeader
        number={3}
        title="CNN Feature Extraction"
        subtitle="Stacked Conv → ReLU → MaxPool blocks turn raw pixels into hierarchical features."
      >
        <MathBlock block>
          {`Z = X * K + b, \\quad A = \\text{ReLU}(Z) = \\max(0, Z), \\quad P_{ij} = \\max_{m,n \\in \\{0,1\\}} A_{2i+m,\\,2j+n}`}
        </MathBlock>
      </ModuleHeader>

      <div className="grid gap-4 lg:grid-cols-[300px_1fr]">
        <Card className="h-fit">
          <CardHeader><CardTitle className="text-base">Pipeline Controls</CardTitle></CardHeader>
          <CardContent className="space-y-4 text-sm">
            <div>
              <Label className="text-xs">Layer 1 kernel (K₁)</Label>
              <Select value={kernelName1} onValueChange={setKernelName1}>
                <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {KERNEL_NAMES.map((k) => (<SelectItem key={k} value={k}>{k}</SelectItem>))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-xs">Layer 2 kernel (K₂)</Label>
              <Select value={kernelName2} onValueChange={setKernelName2}>
                <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {KERNEL_NAMES.map((k) => (<SelectItem key={k} value={k}>{k}</SelectItem>))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <div className="flex justify-between"><Label className="text-xs">Padding (p)</Label><span className="font-mono text-xs">{padding}</span></div>
              <Slider value={[padding]} min={0} max={2} onValueChange={([v]) => setPadding(v ?? 1)} />
            </div>
            <div>
              <div className="flex justify-between"><Label className="text-xs">Bias layer 1 (b₁)</Label><span className="font-mono text-xs">{bias1}</span></div>
              <Slider value={[bias1]} min={-50} max={50} onValueChange={([v]) => setBias1(v ?? 0)} />
            </div>
            <div>
              <div className="flex justify-between"><Label className="text-xs">Bias layer 2 (b₂)</Label><span className="font-mono text-xs">{bias2}</span></div>
              <Slider value={[bias2]} min={-50} max={50} onValueChange={([v]) => setBias2(v ?? 0)} />
            </div>

            <div className="rounded border bg-muted/30 p-2 text-xs space-y-1">
              <div className="flex justify-between"><span>Input</span><DimBadge dims={`${H} × ${W}`} variant="input" /></div>
              <div className="flex justify-between"><span>After Conv1+ReLU</span><DimBadge dims={`${stages[1].data.length} × ${stages[1].data[0].length}`} variant="intermediate" /></div>
              <div className="flex justify-between"><span>After Pool1</span><DimBadge dims={`${stages[2].data.length} × ${stages[2].data[0].length}`} variant="intermediate" /></div>
              <div className="flex justify-between"><span>After Conv2+ReLU</span><DimBadge dims={`${stages[4].data.length} × ${stages[4].data[0].length}`} variant="intermediate" /></div>
              <div className="flex justify-between"><span>Final feature map</span><DimBadge dims={`${stages[6].data.length} × ${stages[6].data[0].length}`} variant="output" /></div>
            </div>
            <p className="text-[10px] text-muted-foreground">
              {sharedImage
                ? "Using your uploaded image's grayscale channel."
                : "Using a built-in 8×8 demo image. Upload a small image (≤ 16×16) in Module 1 to use it here."}
            </p>
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2"><Layers className="h-4 w-4" /> Stacked CNN pipeline — every stage shown numerically and as a heatmap</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex flex-wrap items-start gap-3">
                {stages.map((s, idx) => {
                  const [h, w] = shape(s.data);
                  const cellSize = Math.max(14, Math.min(32, Math.floor(220 / Math.max(h, w))));
                  const isInput = s.kind === "input";
                  const isOutput = s.kind === "feature";
                  const variant = isInput ? "input" : isOutput ? "output" : "intermediate";
                  const diverging = s.kind === "conv" || s.kind === "relu";
                  return (
                    <div key={idx} className="flex items-center gap-3">
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-medium">{s.name}</span>
                          <DimBadge dims={`${h} × ${w}`} variant={variant} />
                        </div>
                        {/* Heatmap view */}
                        <Heatmap
                          matrix={s.data}
                          diverging={diverging}
                          cellSize={cellSize}
                          format={() => ""}
                          min={diverging ? undefined : 0}
                        />
                        <p className="text-[10px] text-muted-foreground max-w-[180px]">{s.note}</p>
                        {s.kernelName && (
                          <p className="text-[10px] font-mono text-muted-foreground">kernel: {s.kernelName}</p>
                        )}
                      </div>
                      {idx < stages.length - 1 && (
                        <ArrowRight className="h-4 w-4 text-muted-foreground hidden md:block" />
                      )}
                    </div>
                  );
                })}
              </div>
              <p className="text-[10px] text-muted-foreground">
                Each heatmap is the literal numeric output of one stage. Note the spatial dimension shrinking through pooling.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-sm">Numeric matrices — final feature map and key intermediate</CardTitle></CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-3">
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">Input (X)</p>
                  <MatrixView matrix={source} digits={0} heatmap cellSize="xs" />
                </div>
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">After Conv1 + ReLU</p>
                  <MatrixView matrix={stages[2].data} digits={2} heatmap diverging cellSize="xs" />
                </div>
                <div>
                  <p className="mb-1 text-xs uppercase text-muted-foreground">Final feature map</p>
                  <MatrixView matrix={stages[6].data} digits={2} heatmap diverging cellSize="sm" />
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <ExpandableSection title="Mathematics of one CNN block" variant="math" defaultOpen>
              <p>A single block performs three operations in sequence:</p>
              <ol className="list-decimal pl-5 space-y-1 mt-2">
                <li>
                  <strong>Convolution:</strong> <MathBlock>{`Z_{ij} = \\sum_{m,n} X_{i+m, j+n} K_{mn} + b`}</MathBlock>
                </li>
                <li>
                  <strong>ReLU activation:</strong> <MathBlock>{`A_{ij} = \\max(0, Z_{ij})`}</MathBlock>
                </li>
                <li>
                  <strong>2×2 MaxPool:</strong> <MathBlock>{`P_{ij} = \\max_{m,n \\in \\{0,1\\}} A_{2i+m,\\,2j+n}`}</MathBlock>
                </li>
              </ol>
              <p className="mt-2">
                Convolution extracts local patterns, ReLU keeps only positive activations (a
                differentiable switch), and pooling downsamples — keeping the most salient
                response in each 2×2 region.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Hierarchical features: pixels → edges → textures → shapes → objects" variant="why" defaultOpen>
              <p>
                Stacking blocks builds a <strong>hierarchy of features</strong>. The first
                convolution sees 3×3 windows of raw pixels and learns edges. The second sees
                3×3 windows of <em>edges</em> and learns textures / corner patterns. Deeper
                layers see combinations of textures → shapes → object parts.
              </p>
              <div className="mt-2 grid grid-cols-5 gap-1 text-center text-[10px] font-medium">
                {["Pixels", "Edges", "Textures", "Shapes", "Objects"].map((label, i) => (
                  <div key={label} className="rounded border bg-muted/30 p-2">
                    <div className="font-mono text-muted-foreground">L{i}</div>
                    <div className="mt-1">{label}</div>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-muted-foreground">
                Each block both <em>increases semantic abstraction</em> and{" "}
                <em>decreases spatial resolution</em> — information becomes "what" rather than "where".
              </p>
            </ExpandableSection>

            <ExpandableSection title="Why ReLU and not sigmoid/tanh?" variant="how">
              <p>
                ReLU <MathBlock>{`f(x) = \\max(0, x)`}</MathBlock> is cheap, sparse (many exact zeros
                → efficient representations), and avoids the <strong>vanishing gradient</strong> problem
                of sigmoid/tanh, whose derivatives shrink to near-zero for large |x|.
              </p>
              <p className="mt-2">
                The downside: a neuron can "die" (always output 0 if its weights push the pre-activation
                negative everywhere). Variants like LeakyReLU and GELU mitigate this; modern Vision
                Transformers typically use GELU.
              </p>
            </ExpandableSection>

            <ExpandableSection title="MaxPool vs AveragePool vs Strided Conv" variant="intuition">
              <p>
                <strong>MaxPool</strong> keeps the strongest activation in each window — translation
                robustness (small shifts barely change the max). <strong>AveragePool</strong> smooths
                the feature map. <strong>Strided convolution</strong> (stride 2) merges pooling into the
                conv itself, making it a learnable downsampler.
              </p>
              <p className="mt-2">
                Modern architectures (ResNet, ConvNeXt) increasingly replace MaxPool with strided convs
                so the network learns <em>what</em> to discard rather than always taking the max.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Numerical example: trace one Conv1 + ReLU + Pool cycle" variant="numerical">
              <p>
                Input X is 8×8. With padding = {padding} and a 3×3 kernel, Conv1 output is{" "}
                <DimBadge dims={`${stages[1].data.length} × ${stages[1].data[0].length}`} variant="intermediate" />.
                ReLU clips negatives (same shape). MaxPool 2×2 halves each dimension to{" "}
                <DimBadge dims={`${stages[2].data.length} × ${stages[2].data[0].length}`} variant="intermediate" />.
              </p>
              <p className="mt-2">
                Each output cell <MathBlock>{`P_{ij}`}</MathBlock> is the maximum of the four cells{" "}
                <MathBlock>{`A_{2i,2j}, A_{2i,2j+1}, A_{2i+1,2j}, A_{2i+1,2j+1}`}</MathBlock> — explicitly
                selecting the most salient 2×2 response.
              </p>
              <div className="mt-2 text-[11px] text-muted-foreground">
                After two blocks: spatial dimension goes {H}×{W} → {stages[2].data.length}×{stages[2].data[0].length} → {stages[6].data.length}×{stages[6].data[0].length}.
                This is the cascaded downsampling that turns a {H}×{W} image into a compact feature vector.
              </div>
            </ExpandableSection>

            <ExpandableSection title="How this connects to Vision Transformers" variant="how">
              <p>
                CNNs hard-code the locality and weight-sharing prior into the architecture. Vision
                Transformers (Modules 4–7) remove this inductive bias: they split the image into patches
                and let <em>attention</em> learn spatial relationships from data. ViTs often outperform
                CNNs at scale, but CNNs win on smaller datasets thanks to their prior.
              </p>
              <p className="mt-2 text-muted-foreground">
                Hybrid designs (ConvNeXt, ViT with convolutional stems) combine the best of both worlds.
              </p>
            </ExpandableSection>
          </div>

          <div className="rounded-lg border bg-muted/30 p-3 text-xs text-muted-foreground flex items-center gap-2">
            <ArrowDown className="h-4 w-4" />
            <span>
              Output of this CNN stage becomes the input to either the next conv block, or — in a ViT
              context — gets reshaped into patches (Module 4).
            </span>
          </div>

          <VivaPanel
            questions={[
              {
                level: "Easy",
                question: "What is the role of ReLU in a CNN?",
                hint: "Think about negative values and gradients.",
                answer:
                  "ReLU applies f(x) = max(0, x) element-wise, introducing nonlinearity so the network can model non-linear functions. It also keeps gradients healthy for positive activations.",
                explanation:
                  "Without a nonlinearity, a stack of convolutions would collapse to a single linear map. ReLU is cheap, sparse, and avoids the vanishing-gradient problem of sigmoid/tanh.",
              },
              {
                level: "Medium",
                question:
                  "After Conv1 (3×3, padding=1, stride=1) and MaxPool 2×2 on an 8×8 input, what is the output shape?",
                hint: "Padding preserves size; pooling halves it.",
                answer:
                  "Conv1 output is 8×8 (padding 1 keeps the size). MaxPool 2×2 then halves both dimensions: 8/2 = 4. Final shape: 4×4.",
                explanation:
                  "Convolution with padding p and stride 1 produces output size H + 2p − k + 1 = 8 + 2 − 3 + 1 = 8. MaxPool 2×2 with stride 2 produces floor(H/2).",
              },
              {
                level: "Difficult",
                question:
                  "Why do deep CNNs build a hierarchy (edges → textures → shapes → objects) even though every layer uses the same 3×3 convolution operator?",
                hint: "Consider what each layer's input actually represents.",
                answer:
                  "Each layer's input is the previous layer's output. Layer 1 sees raw pixels, so its 3×3 convolutions detect pixel-level edges. Layer 2 sees edges, so its 3×3 convolutions combine edges into corner/texture detectors. Each layer composes on top of the previous, producing progressively more abstract features.",
                explanation:
                  "The 3×3 receptive field of a single conv is fixed, but the effective receptive field grows multiplicatively with depth: two stacked 3×3 convs give a 5×5 effective receptive field. Combined with nonlinearities and pooling, this produces the observed hierarchy.",
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
}
