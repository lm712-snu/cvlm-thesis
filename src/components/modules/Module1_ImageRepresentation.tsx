"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ImageUploader, type UploadedImage } from "@/components/cvlm/ImageUploader";
import { MatrixView } from "@/components/cvlm/MatrixView";
import { Heatmap } from "@/components/cvlm/Heatmap";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import { useAppStore } from "@/store/app-store";
import { fmtVec } from "@/lib/math";

export function Module1_ImageRepresentation() {
  const sharedImage = useAppStore((s) => s.sharedImage);
  const setSharedImage = useAppStore((s) => s.setSharedImage);
  const [pixel, setPixel] = useState<[number, number]>([0, 0]);
  const [showNormalized, setShowNormalized] = useState(false);
  const [patchSize, setPatchSize] = useState(8);

  const img = sharedImage;
  const [py, px] = pixel;

  return (
    <div>
      <ModuleHeader
        number={1}
        title="Mathematical Representation of an Image"
        subtitle="Pixel → RGB channels → tensor X ∈ ℝ^(H×W×3) → normalization → flattening."
      >
        <MathBlock block>
          {`X \\in \\mathbb{R}^{H \\times W \\times 3}, \\quad X_{i,j,c} \\in [0, 255] \\text{ (uint8) or } [0, 1] \\text{ (normalized)}`}
        </MathBlock>
      </ModuleHeader>

      <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
        <Card>
          <CardHeader><CardTitle className="text-base">Input</CardTitle></CardHeader>
          <CardContent>
            <ImageUploader image={img} onImage={setSharedImage} maxSize={64} />
            {img && (
              <div className="mt-3 space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <span>Dimensions</span>
                  <DimBadge dims={`${img.height} × ${img.width} × 3`} variant="input" />
                </div>
                <div className="flex items-center justify-between">
                  <span>Total elements</span>
                  <code className="text-xs">{img.height * img.width * 3}</code>
                </div>
                <div className="flex items-center justify-between">
                  <span>Bytes (uint8)</span>
                  <code className="text-xs">{img.height * img.width * 3}</code>
                </div>
                <div className="flex items-center justify-between">
                  <span>Show normalized [0,1]</span>
                  <Switch checked={showNormalized} onCheckedChange={setShowNormalized} />
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <div className="space-y-4">
          {img ? (
            <>
              <Tabs defaultValue="image">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="image">Image</TabsTrigger>
                  <TabsTrigger value="pixels">Pixel grid</TabsTrigger>
                  <TabsTrigger value="channels">Channels</TabsTrigger>
                  <TabsTrigger value="tensor">Tensor</TabsTrigger>
                </TabsList>

                <TabsContent value="image" className="space-y-3">
                  <Card>
                    <CardContent className="pt-4">
                      <div className="flex flex-wrap gap-4">
                        <img
                          src={img.dataUrl}
                          alt="uploaded"
                          className="rounded border"
                          style={{ imageRendering: "pixelated", width: 256, height: 256 }}
                        />
                        <div className="flex-1 space-y-2 text-sm">
                          <p className="text-muted-foreground">
                            Click on any cell in the pixel grid below to inspect its RGB value. Each pixel is a
                            3-vector <MathBlock>{`(R, G, B) \\in \\mathbb{R}^3`}</MathBlock>.
                          </p>
                          <div className="rounded border bg-muted/30 p-3 space-y-1">
                            <p className="text-xs uppercase text-muted-foreground">Selected pixel</p>
                            <p className="font-mono">Position: ({py}, {px})</p>
                            <p className="font-mono">RGB: {fmtVec(img.pixels[py][px].slice(0, 3), 0)}</p>
                            <p className="font-mono">Normalized: {fmtVec(img.pixels[py][px].slice(0, 3).map((v) => v / 255), 3)}</p>
                            <div className="flex items-center gap-2 pt-1">
                              <span className="text-xs">Color:</span>
                              <div
                                className="h-6 w-6 rounded border"
                                style={{ backgroundColor: `rgb(${img.pixels[py][px][0]}, ${img.pixels[py][px][1]}, ${img.pixels[py][px][2]})` }}
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader><CardTitle className="text-sm">Interactive Pixel Grid (grayscale)</CardTitle></CardHeader>
                    <CardContent>
                      <Heatmap
                        matrix={img.gray.map((r) => r.map((v) => (showNormalized ? v / 255 : v)))}
                        rowLabels={img.gray.map((_, i) => String(i))}
                        colLabels={img.gray[0].map((_, j) => String(j))}
                        highlight={[py, px]}
                        onCellClick={(i, j) => setPixel([i, j])}
                        cellSize={Math.max(14, Math.floor(320 / Math.max(img.height, img.width)))}
                        min={0}
                        max={showNormalized ? 1 : 255}
                        format={(v) => ""}
                      />
                      <p className="mt-2 text-xs text-muted-foreground">
                        Click any cell to inspect. Brightness = gray value{" "}
                        <MathBlock>{`Y = 0.299 R + 0.587 G + 0.114 B`}</MathBlock>.
                      </p>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="pixels" className="space-y-3">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">Numeric Pixel Matrix</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex items-center gap-2">
                        <Label className="text-xs">Patch size:</Label>
                        <Slider
                          value={[patchSize]}
                          min={4}
                          max={Math.min(img.width, img.height)}
                          onValueChange={([v]) => setPatchSize(v ?? 8)}
                          className="max-w-xs"
                        />
                        <span className="font-mono text-xs">{patchSize}×{patchSize}</span>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Showing top-left <code>{patchSize}×{patchSize}</code> patch of the red channel (truncated for readability).
                      </p>
                      <MatrixView
                        matrix={
                          showNormalized
                            ? img.channels[0].slice(0, patchSize).map((r) => r.slice(0, patchSize).map((v) => v / 255))
                            : img.channels[0].slice(0, patchSize).map((r) => r.slice(0, patchSize))
                        }
                        heatmap
                        digits={showNormalized ? 3 : 0}
                        highlight={py < patchSize && px < patchSize ? [py, px] : undefined}
                        onCellClick={(i, j) => setPixel([i, j])}
                      />
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="channels" className="space-y-3">
                  <div className="grid gap-3 md:grid-cols-3">
                    {["Red", "Green", "Blue"].map((name, c) => (
                      <Card key={name}>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm flex items-center gap-2">
                            <span
                              className="h-3 w-3 rounded-full"
                              style={{ backgroundColor: ["#ef4444", "#22c55e", "#3b82f6"][c] }}
                            />
                            {name} channel
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <Heatmap
                            matrix={img.channels[c]}
                            min={0}
                            max={255}
                            cellSize={Math.max(6, Math.floor(220 / Math.max(img.height, img.width)))}
                            format={(v) => ""}
                          />
                          <p className="mt-1 text-[10px] text-muted-foreground font-mono">
                            shape = [{img.height}, {img.width}]
                          </p>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </TabsContent>

                <TabsContent value="tensor" className="space-y-3">
                  <Card>
                    <CardContent className="pt-4 space-y-3">
                      <p className="text-sm">
                        The full image is a 3-tensor <MathBlock>{`X \\in \\mathbb{R}^{${img.height} \\times ${img.width} \\times 3}`}</MathBlock>.
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Flattened, this is a vector of length <DimBadge dims={`${img.height * img.width * 3}`} variant="output" />.
                      </p>
                      <ExpandableSection title="Flattening a tensor" variant="math" defaultOpen>
                        <p>
                          To feed an image into a linear layer, we flatten the spatial dimensions:
                        </p>
                        <MathBlock block>{`\\text{vec}(X) \\in \\mathbb{R}^{H \\cdot W \\cdot C}`}</MathBlock>
                        <p>
                          For this image: H = {img.height}, W = {img.width}, C = 3, so the vector has{" "}
                          <strong>{img.height * img.width * 3}</strong> elements. Modern Vision Transformers
                          avoid flattening the whole image; they flatten <em>patches</em> instead (Module 4).
                        </p>
                      </ExpandableSection>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>

              <ExpandableSection title="What does X ∈ ℝ^(H×W×3) actually mean?" variant="why" defaultOpen>
                <p>
                  An image is a 3-dimensional array (a tensor) of shape H rows, W columns, and 3 colour channels
                  (Red, Green, Blue). The entry <MathBlock>{`X_{i,j,c}`}</MathBlock> is the intensity of colour{" "}
                  <MathBlock>{`c`}</MathBlock> at row <MathBlock>{`i`}</MathBlock>, column <MathBlock>{`j`}</MathBlock>.
                </p>
                <p className="mt-2">
                  For an 8-bit image each entry is an integer in <MathBlock>{`[0, 255]`}</MathBlock>. In deep
                  learning we usually normalize by dividing by 255 so that the tensor lies in{" "}
                  <MathBlock>{`[0, 1]`}</MathBlock>, which helps gradient-based optimization.
                </p>
              </ExpandableSection>

              <ExpandableSection title="Normalization: why divide by 255?" variant="how">
                <p>Two reasons:</p>
                <ol className="list-decimal pl-5 space-y-1">
                  <li>
                    <strong>Numerical stability.</strong> Activations like sigmoid/tanh saturate for large inputs.
                    Keeping values in <MathBlock>{`[0,1]`}</MathBlock> (or standardized to zero mean, unit variance)
                    keeps gradients healthy.
                  </li>
                  <li>
                    <strong>Weight initialization.</strong> Standard initializations (Xavier, He) assume inputs of
                    unit variance. Normalizing makes the assumption hold.
                  </li>
                </ol>
                <p className="mt-2">
                  Production pipelines often go further with per-channel mean/std normalization (e.g. ImageNet
                  statistics: mean <code>[0.485, 0.456, 0.406]</code>, std <code>[0.229, 0.224, 0.225]</code>).
                </p>
              </ExpandableSection>

              <VivaPanel
                questions={[
                  {
                    level: "Easy",
                    question: "What is a pixel, mathematically?",
                    hint: "Think about how many numbers describe one pixel.",
                    answer: "A pixel in an RGB image is a 3-vector (R, G, B) ∈ ℝ³.",
                    explanation:
                      "Each colour channel stores an intensity. In an 8-bit image each component is an integer in [0, 255]. Conceptually a pixel is a point in a 3-dimensional colour space.",
                  },
                  {
                    level: "Medium",
                    question: "Why do we flatten an image before feeding it to a linear layer?",
                    hint: "Linear layers expect 2D inputs.",
                    answer:
                      "Linear layers y = Wx + b require x to be a vector. A 2D image X ∈ ℝ^(H×W×C) must be reshaped into a vector of length H·W·C.",
                    explanation:
                      "This destroys spatial structure, which is why CNNs (Module 2-3) and Vision Transformers (Module 4-7) are preferred: they preserve locality and process the image as a structured tensor, not a flat vector.",
                  },
                  {
                    level: "Difficult",
                    question:
                      "Why does mean/std normalization (e.g. ImageNet statistics) outperform simple /255 normalization for transfer learning?",
                    hint: "Consider what distribution the pretrained weights expect.",
                    answer:
                      "Pretrained models were trained with inputs normalized to ~zero mean and unit variance using ImageNet statistics. Matching that distribution at inference keeps activations in the range the weights were tuned for.",
                    explanation:
                      "If you feed an ImageNet-trained model with inputs in [0, 1], the first layer's pre-activations will have a positive mean and different variance than during training, causing distribution shift and degraded features.",
                  },
                ]}
              />
            </>
          ) : (
            <Card>
              <CardContent className="py-12 text-center text-muted-foreground">
                Upload an image (or use the sample) to begin.
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
