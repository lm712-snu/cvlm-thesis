"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { MatrixView } from "@/components/cvlm/MatrixView";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import { matmul, addBias, randMatrix, shape, fmtVec } from "@/lib/math/matrix";
import { Dice5, RefreshCw, ArrowRight, ArrowDown } from "lucide-react";

// Small deterministic example: N=4 patches, P²C=4, d=3.
// Patch values are intentionally simple to make the matmul traceable by hand.
const PATCH_X_4x4: number[][] = [
  [1, 2, 3, 4],
  [2, 3, 4, 5],
  [3, 4, 5, 6],
  [4, 5, 6, 7],
];
const BIAS_DEMO = [0.5, 0.5, 0.5];

export function Module5_PatchEmbedding() {
  const [n, setN] = useState(4);
  const [ppC, setPpC] = useState(4);
  const [d, setD] = useState(3);
  const [seed, setSeed] = useState(7);
  const [showViTScale, setShowViTScale] = useState(false);

  // Build a small patch matrix (N × P²C). When N=4 and P²C=4 we use the
  // hand-authored matrix so users can verify by hand; otherwise we generate
  // a deterministic random one so they can experiment with dimensions.
  const { X_p, W_E, b, Z } = useMemo(() => {
    const X_p =
      n === 4 && ppC === 4
        ? PATCH_X_4x4
        : randMatrix(n, ppC, -2, 2, seed);
    const W_E = randMatrix(ppC, d, -1, 1, seed + 1);
    const b = Array.from({ length: d }, (_, i) =>
      // deterministic bias from the same seed family
      Number((((seed + 2 + i) * 17) % 100) / 100 - 0.5)
    );
    const Z = addBias(matmul(X_p, W_E), b);
    return { X_p, W_E, b, Z };
  }, [n, ppC, d, seed]);

  const [nRows, nCols] = shape(X_p);
  const [wRows, wCols] = shape(W_E);

  return (
    <div>
      <ModuleHeader
        number={5}
        title="Patch Embeddings"
        subtitle="Linear projection: each flattened patch x_p ∈ ℝ^(P²C) is projected to a d-dim embedding z ∈ ℝ^d via a learnable W_E."
      >
        <MathBlock block>
          {`Z = X_p \\, W_E + b, \\quad X_p \\in \\mathbb{R}^{N \\times P^2 C},\\ W_E \\in \\mathbb{R}^{P^2 C \\times d},\\ Z \\in \\mathbb{R}^{N \\times d}`}
        </MathBlock>
      </ModuleHeader>

      <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
        <Card className="h-fit">
          <CardHeader><CardTitle className="text-base">Configuration</CardTitle></CardHeader>
          <CardContent className="space-y-4 text-sm">
            <div>
              <div className="flex justify-between"><Label className="text-xs">Patches (N)</Label><span className="font-mono text-xs">{n}</span></div>
              <Slider value={[n]} min={2} max={8} onValueChange={([v]) => setN(v ?? 4)} />
            </div>
            <div>
              <div className="flex justify-between"><Label className="text-xs">Patch dim P²·C</Label><span className="font-mono text-xs">{ppC}</span></div>
              <Slider value={[ppC]} min={2} max={8} onValueChange={([v]) => setPpC(v ?? 4)} />
            </div>
            <div>
              <div className="flex justify-between"><Label className="text-xs">Embed dim d</Label><span className="font-mono text-xs">{d}</span></div>
              <Slider value={[d]} min={2} max={8} onValueChange={([v]) => setD(v ?? 3)} />
            </div>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={() => setSeed((s) => s + 1)}>
                <Dice5 className="h-3.5 w-3.5" /> New weights
              </Button>
              <Button size="sm" variant="ghost" onClick={() => { setN(4); setPpC(4); setD(3); setSeed(7); }}>
                <RefreshCw className="h-3.5 w-3.5" /> Reset
              </Button>
            </div>
            <div className="rounded border bg-muted/30 p-2 text-xs space-y-1">
              <div className="flex justify-between"><span>Patches X_p</span><DimBadge dims={`${nRows} × ${nCols}`} variant="input" /></div>
              <div className="flex justify-between"><span>Weights W_E</span><DimBadge dims={`${wRows} × ${wCols}`} variant="weight" /></div>
              <div className="flex justify-between"><span>Bias b</span><DimBadge dims={`${b.length}`} variant="weight" /></div>
              <div className="flex justify-between"><span>Embeddings Z</span><DimBadge dims={`${n} × ${d}`} variant="output" /></div>
            </div>
            <Button size="sm" variant={showViTScale ? "default" : "outline"} className="w-full" onClick={() => setShowViTScale((s) => !s)}>
              {showViTScale ? "Hide ViT-scale dims" : "Show ViT-scale dims"}
            </Button>
            {showViTScale && (
              <div className="rounded border border-violet-500/40 bg-violet-500/10 p-2 text-[11px] text-violet-700 dark:text-violet-300 space-y-1">
                <p className="font-semibold">ViT-Base / ImageNet</p>
                <p>X_p: <DimBadge dims="196 × 768" variant="input" /></p>
                <p>W_E: <DimBadge dims="768 × 768" variant="weight" /></p>
                <p>Z: <DimBadge dims="196 × 768" variant="output" /></p>
                <p className="text-muted-foreground mt-1">~590k learnable parameters in W_E alone.</p>
              </div>
            )}
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-sm">Projection flow: patches → W_E → embeddings</CardTitle></CardHeader>
            <CardContent>
              <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-start">
                {/* Patches */}
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <p className="text-xs uppercase text-muted-foreground">Patches X_p</p>
                    <DimBadge dims={`${nRows} × ${nCols}`} variant="input" />
                  </div>
                  <MatrixView matrix={X_p} digits={2} heatmap diverging cellSize="sm" />
                  <p className="text-[10px] text-muted-foreground max-w-[180px]">Flattened patch values (one patch per row).</p>
                </div>

                <ArrowRight className="hidden md:block h-6 w-6 text-muted-foreground mx-1" />
                <ArrowDown className="md:hidden h-5 w-5 text-muted-foreground" />

                {/* W_E */}
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <p className="text-xs uppercase text-muted-foreground">Weights W_E</p>
                    <DimBadge dims={`${wRows} × ${wCols}`} variant="weight" />
                  </div>
                  <MatrixView matrix={W_E} digits={2} heatmap diverging cellSize="sm" />
                  <p className="text-[10px] text-muted-foreground max-w-[180px]">Learnable projection. Trained by back-prop.</p>
                </div>

                <ArrowRight className="hidden md:block h-6 w-6 text-muted-foreground mx-1" />
                <ArrowDown className="md:hidden h-5 w-5 text-muted-foreground" />

                {/* Output */}
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <p className="text-xs uppercase text-muted-foreground">Embeddings Z</p>
                    <DimBadge dims={`${n} × ${d}`} variant="output" />
                  </div>
                  <MatrixView matrix={Z} digits={2} heatmap diverging cellSize="sm" />
                  <p className="text-[10px] text-muted-foreground max-w-[180px]">One d-dim embedding per patch — input to the transformer.</p>
                </div>
              </div>

              <div className="mt-4 rounded border bg-muted/30 p-3 text-xs font-mono space-y-1">
                <p>Z = X_p · W_E + b</p>
                <p>Z = [{nRows}×{nCols}] · [{wRows}×{wCols}] + [{b.length}] = [{n}×{d}]</p>
                <p className="text-muted-foreground">b (broadcast across rows) = {fmtVec(b, 2)}</p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-sm">One embedding computed by hand</CardTitle></CardHeader>
            <CardContent className="space-y-2 text-sm">
              <p className="text-muted-foreground">
                The first patch <MathBlock>{`x_p^{(0)}`}</MathBlock> = <code className="text-xs">{fmtVec(X_p[0], 2)}</code>.
                Its embedding is the dot-product with each column of W_E, plus bias:
              </p>
              <div className="rounded border bg-muted/20 p-3 text-xs font-mono space-y-1">
                {Array.from({ length: d }).map((_, j) => (
                  <div key={j}>
                    z[0, {j}] = {X_p[0].map((v, i) => `${v.toFixed(2)}·${W_E[i][j].toFixed(2)}`).join(" + ")} + {b[j].toFixed(2)}
                    <span className="ml-2 text-primary">= {Z[0][j].toFixed(4)}</span>
                  </div>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                Each output cell is a weighted sum of all {ppC} input values. The weights come from a
                <em> learned</em> column of <MathBlock>{`W_E`}</MathBlock> — this is exactly a fully-connected layer applied to each patch independently.
              </p>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <ExpandableSection title="Mathematics: linear projection" variant="math" defaultOpen>
              <p>
                A patch embedding is a single linear layer applied row-wise to the patch matrix:
              </p>
              <MathBlock block>{`Z = X_p W_E + b`}</MathBlock>
              <ul className="list-disc pl-5 space-y-1 mt-2">
                <li><MathBlock>{`X_p \\in \\mathbb{R}^{N \\times P^2 C}`}</MathBlock> — N patches, each of dim P²C.</li>
                <li><MathBlock>{`W_E \\in \\mathbb{R}^{P^2 C \\times d}`}</MathBlock> — learnable projection.</li>
                <li><MathBlock>{`b \\in \\mathbb{R}^{d}`}</MathBlock> — per-channel bias, broadcast across rows.</li>
                <li><MathBlock>{`Z \\in \\mathbb{R}^{N \\times d}`}</MathBlock> — embedded sequence.</li>
              </ul>
              <p className="mt-2">
                Equivalent to a fully-connected layer <MathBlock>{`\\text{Linear}(P^2 C, d)`}</MathBlock> applied independently to each of the N patches.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Why is W_E learnable? What does it learn?" variant="why" defaultOpen>
              <p>
                The weights <MathBlock>{`W_E`}</MathBlock> are <strong>learned end-to-end</strong> by
                back-propagation from the language-modelling (or classification) loss. After training,
                the rows of <MathBlock>{`W_E^\\top`}</MathBlock> become <em>feature detectors</em>:
                each output dimension extracts one learned pattern from the patch.
              </p>
              <p className="mt-2">
                Visualizing learned W_E rows of real ViTs reveals Gabor-like filters (edges at specific
                orientations and scales) — the network rediscovers, from data, the same low-level
                features that hand-crafted computer vision used for decades.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Equivalence to a 1×1 convolution" variant="how">
              <p>
                Applying <MathBlock>{`W_E`}</MathBlock> to the patch grid is mathematically identical
                to a single <MathBlock>{`P \\times P`}</MathBlock> convolution with stride <MathBlock>{`P`}</MathBlock>{" "}
                and <MathBlock>{`d`}</MathBlock> output channels. The conv kernel is exactly the reshape of <MathBlock>{`W_E`}</MathBlock> into <MathBlock>{`P \\times P \\times C \\times d`}</MathBlock>.
              </p>
              <p className="mt-2 text-muted-foreground">
                Many implementations (e.g. timm) implement patch embedding as a Conv2d for performance, but the math is the same.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Intuition: from pixels to a 'word vector'" variant="intuition">
              <p>
                A word embedding maps a discrete token (e.g. "cat") to a dense vector capturing its meaning.
                Patch embedding does the same for visual tokens: a 16×16 patch of pixels becomes a 768-dim
                vector that captures its visual content.
              </p>
              <p className="mt-2">
                The transformer then attends over these embeddings just like a BERT attends over word embeddings.
                Patches and words are unified as <em>tokens</em>.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Numerical example: trace one output cell" variant="numerical">
              <p>
                For the first patch <MathBlock>{`x_p^{(0)}`}</MathBlock> = {fmtVec(X_p[0], 2)} and
                W_E column 0 = <code className="text-xs">[{W_E.map((r) => r[0].toFixed(2)).join(", ")}]</code>:
              </p>
              <MathBlock block>{`z_0^{(0)} = \\sum_{i=1}^{P^2 C} x_{p,i}^{(0)} \\cdot W_{E,i,0} + b_0 = ${Z[0][0].toFixed(4)}`}</MathBlock>
              <p className="mt-2 text-muted-foreground">
                This is the inner product of the patch vector with a learned feature vector — high when the patch
                matches that feature, low otherwise.
              </p>
            </ExpandableSection>

            <ExpandableSection title="What about a CLS token?" variant="how">
              <p>
                Real ViTs prepend a learnable <MathBlock>{`[\\text{CLS}]`}</MathBlock> token to the
                sequence: <MathBlock>{`Z = [z_{\\text{cls}};\\, X_p W_E + b]`}</MathBlock>. The CLS token
                has no patch content; its final-layer embedding is used as the image-level summary for
                classification.
              </p>
              <p className="mt-2 text-muted-foreground">
                For VLMs, the CLS is often replaced with a global pooling, or simply all patch embeddings are passed to the cross-attention.
              </p>
            </ExpandableSection>
          </div>

          <VivaPanel
            questions={[
              {
                level: "Easy",
                question: "What is the role of the patch embedding layer?",
                hint: "It maps a high-dim patch vector to a lower-dim embedding.",
                answer:
                  "The patch embedding projects each flattened patch x_p ∈ ℝ^(P²C) to a d-dim embedding z ∈ ℝ^d via a learnable linear map W_E. It converts raw pixel blocks into dense token embeddings suitable for a transformer.",
                explanation:
                  "Without this projection, patches would be very high-dimensional (e.g. 768 for 16×16×3), forcing the attention layers to also work in 768-dim. By learning W_E, the network both reduces dimensionality and discovers useful low-level features simultaneously.",
              },
              {
                level: "Medium",
                question:
                  "Why can patch embedding be implemented as a single strided convolution? What are the kernel/stride dimensions?",
                hint: "Look at the shape of W_E.",
                answer:
                  "W_E has shape (P²C × d). Reshape its rows into a (P × P × C × d) kernel and apply it to the image with stride P and zero padding. Output: (H/P × W/P × d), which flattens to N × d.",
                explanation:
                  "The reshape and strided convolution compute the exact same inner products as the linear projection, but exploit efficient conv kernels. This is why timm uses Conv2d(P²C, d, kernel=P, stride=P) under the hood.",
              },
              {
                level: "Difficult",
                question:
                  "Patch embedding destroys spatial structure within a patch (it flattens P×P×C pixels into a vector). Is this a problem? Why don't ViTs suffer noticeably from it?",
                hint: "Think about what the first attention layer can recover.",
                answer:
                  "Within-patch structure is partially lost because flattening ignores the 2D layout. However: (1) the patch is small (P=16), so a linear projection can still learn position-aware features implicitly via W_E; (2) the first self-attention layer has access to all N patches and can re-derive spatial relationships through attention. With sinusoidal positional embeddings (Module 6), every patch also knows its grid location, restoring global spatial awareness.",
                explanation:
                  "Empirically, ViTs match or exceed CNNs at scale, suggesting that the lost within-patch locality is a minor price. Swin Transformers and similar hierarchical designs reintroduce local 2D structure within windows for further gains.",
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
}
