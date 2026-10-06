"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import { useAppStore } from "@/store/app-store";
import type { UploadedImage } from "@/components/cvlm/ImageUploader";
import { Grid3x3, Move } from "lucide-react";

/** Sample synthetic image generator: returns a data URL plus image dimensions. */
function useSampleImage(): UploadedImage | null {
  // Lazy-initialised once on first client render; canvas drawing on a detached
  // element is a pure computation, so the useState initialiser is safe here.
  const [img] = useState<UploadedImage | null>(() => {
    if (typeof document === "undefined") return null;
    const w = 224, h = 224;
    const canvas = document.createElement("canvas");
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    const grad = ctx.createLinearGradient(0, 0, w, h);
    grad.addColorStop(0, "#0f766e");
    grad.addColorStop(0.5, "#f59e0b");
    grad.addColorStop(1, "#db2777");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
    // Some geometric shapes so patches are visibly different.
    ctx.fillStyle = "rgba(255,255,255,0.6)";
    ctx.beginPath(); ctx.arc(w * 0.3, h * 0.35, 28, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "rgba(20,20,20,0.7)";
    ctx.fillRect(w * 0.55, h * 0.55, 60, 60);
    ctx.strokeStyle = "rgba(255,255,255,0.8)";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(w * 0.1, h * 0.8);
    ctx.lineTo(w * 0.4, h * 0.55);
    ctx.stroke();
    const data = ctx.getImageData(0, 0, w, h).data;
    const pixels: number[][][] = Array.from({ length: h }, () => Array.from({ length: w }, () => [0, 0, 0, 0]));
    const channels: number[][][] = [
      Array.from({ length: h }, () => new Array(w).fill(0)),
      Array.from({ length: h }, () => new Array(w).fill(0)),
      Array.from({ length: h }, () => new Array(w).fill(0)),
    ];
    const gray: number[][] = Array.from({ length: h }, () => new Array(w).fill(0));
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      const r = data[idx], g = data[idx + 1], b = data[idx + 2];
      pixels[y][x] = [r, g, b, 255];
      channels[0][y][x] = r; channels[1][y][x] = g; channels[2][y][x] = b;
      gray[y][x] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
    return { dataUrl: canvas.toDataURL(), pixels, width: w, height: h, gray, channels };
  });
  return img;
}

/** Tiny 8×8 demo image (gradient) for the worked patch-extraction example. */
const TINY_8x8: number[][][] = (() => {
  const out: number[][][] = [];
  for (let i = 0; i < 8; i++) {
    const row: number[][] = [];
    for (let j = 0; j < 8; j++) {
      const v = (i * 8 + j) * 4;
      row.push([v, 255 - v, (i + j) * 16]);
    }
    out.push(row);
  }
  return out;
})();

export function Module4_Patches() {
  const sharedImage = useAppStore((s) => s.sharedImage);
  const sample = useSampleImage();
  // Default to ViT-canonical 224×224, P=16 → 196 patches.
  const [imgSize, setImgSize] = useState(224);
  const [patchSize, setPatchSize] = useState(16);
  const [channels, setChannels] = useState(3);

  const displayImage = sharedImage ?? sample;

  // Compute patch count and per-patch dimension for the current sliders.
  const patchesPerSide = Math.max(1, Math.floor(imgSize / patchSize));
  const numPatches = patchesPerSide * patchesPerSide;
  const patchDim = patchSize * patchSize * channels;
  // Approximate tokens-per-image scaled to common ViT config.
  const isCanonical = imgSize === 224 && patchSize === 16 && channels === 3;

  // Compute overlay grid dimensions as a percentage of the displayed image size.
  const overlayPct = (100 * patchSize) / imgSize;

  return (
    <div>
      <ModuleHeader
        number={4}
        title="Image Patches"
        subtitle="Splitting an image into N non-overlapping P×P patches — the bridge from convolutional thinking to sequence thinking."
      >
        <MathBlock block>
          {`N = \\left\\lfloor \\frac{H}{P} \\right\\rfloor^2, \\quad \\text{patch}_k \\in \\mathbb{R}^{P^2 \\cdot C}`}
        </MathBlock>
      </ModuleHeader>

      <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
        <Card className="h-fit">
          <CardHeader><CardTitle className="text-base">Patch Configuration</CardTitle></CardHeader>
          <CardContent className="space-y-4 text-sm">
            <div>
              <div className="flex justify-between"><Label className="text-xs">Image size (H = W)</Label><span className="font-mono text-xs">{imgSize}px</span></div>
              <Slider value={[imgSize]} min={32} max={256} step={16} onValueChange={([v]) => setImgSize(v ?? 224)} />
            </div>
            <div>
              <div className="flex justify-between"><Label className="text-xs">Patch size (P)</Label><span className="font-mono text-xs">{patchSize}px</span></div>
              <Slider value={[patchSize]} min={4} max={32} step={4} onValueChange={([v]) => setPatchSize(v ?? 16)} />
            </div>
            <div>
              <div className="flex justify-between"><Label className="text-xs">Channels (C)</Label><span className="font-mono text-xs">{channels}</span></div>
              <Slider value={[channels]} min={1} max={3} onValueChange={([v]) => setChannels(v ?? 3)} />
            </div>

            <div className="rounded border bg-muted/30 p-3 text-xs space-y-2">
              <div className="flex items-center justify-between">
                <span>Patches per side</span>
                <code className="font-mono">{patchesPerSide}</code>
              </div>
              <div className="flex items-center justify-between">
                <span>Number of patches N</span>
                <DimBadge dims={`${numPatches}`} variant="output" />
              </div>
              <div className="flex items-center justify-between">
                <span>Patch dimension P²·C</span>
                <DimBadge dims={`${patchDim}`} variant="intermediate" />
              </div>
              <div className="flex items-center justify-between">
                <span>Total elements preserved</span>
                <code className="font-mono">{imgSize * imgSize * channels}</code>
              </div>
            </div>

            {isCanonical ? (
              <p className="rounded border border-emerald-500/40 bg-emerald-500/10 p-2 text-[11px] text-emerald-700 dark:text-emerald-300">
                This is the canonical ViT-Base configuration: 224×224 image, 16×16 patches → 196 tokens of dim 768.
              </p>
            ) : (
              <p className="rounded border bg-muted/20 p-2 text-[11px] text-muted-foreground">
                Drag the sliders to see how N and patch dimension depend on H, P, and C.
              </p>
            )}

            <p className="text-[10px] text-muted-foreground">
              {sharedImage
                ? "Overlay shown on your uploaded image (from Module 1)."
                : "Overlay shown on a generated 224×224 sample image."}
            </p>
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Tabs defaultValue="overlay">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="overlay"><Grid3x3 className="h-3.5 w-3.5 mr-1" /> Grid Overlay</TabsTrigger>
              <TabsTrigger value="example">Worked Example</TabsTrigger>
              <TabsTrigger value="math">Math</TabsTrigger>
            </TabsList>

            <TabsContent value="overlay" className="space-y-3">
              <Card>
                <CardContent className="pt-4">
                  <div className="flex flex-col md:flex-row gap-4">
                    <div className="relative inline-block">
                      {displayImage ? (
                        <img
                          src={displayImage.dataUrl}
                          alt="source"
                          className="rounded border"
                          style={{ width: 320, height: 320, objectFit: "cover", imageRendering: "auto" }}
                        />
                      ) : (
                        <div className="h-[320px] w-[320px] rounded border bg-muted animate-pulse" />
                      )}
                      {/* Patch grid overlay — lines drawn using SVG, sized in % of the displayed image. */}
                      <svg
                        className="pointer-events-none absolute inset-0"
                        width="320"
                        height="320"
                        viewBox="0 0 100 100"
                        preserveAspectRatio="none"
                      >
                        {Array.from({ length: patchesPerSide + 1 }).map((_, i) => {
                          const pos = i * overlayPct;
                          return (
                            <g key={`g-${i}`}>
                              <line x1={pos} y1={0} x2={pos} y2={100} stroke="rgba(255,255,255,0.7)" strokeWidth={0.3} />
                              <line x1={0} y1={pos} x2={100} y2={pos} stroke="rgba(255,255,255,0.7)" strokeWidth={0.3} />
                            </g>
                          );
                        })}
                      </svg>
                      {/* Highlight the top-left patch as the "first token". */}
                      <div
                        className="pointer-events-none absolute border-2 border-amber-500"
                        style={{
                          left: 0, top: 0,
                          width: `${overlayPct}%`,
                          height: `${overlayPct}%`,
                        }}
                      />
                    </div>
                    <div className="flex-1 space-y-3 text-sm">
                      <div>
                        <p className="text-xs uppercase text-muted-foreground">Patch grid</p>
                        <p className="font-mono text-2xl">{patchesPerSide} × {patchesPerSide} = {numPatches} patches</p>
                      </div>
                      <div>
                        <p className="text-xs uppercase text-muted-foreground">Each patch</p>
                        <p className="font-mono">{patchSize} × {patchSize} × {channels} = {patchDim} numbers</p>
                        <DimBadge dims={`ℝ^${patchDim}`} variant="intermediate" className="mt-1" />
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Each highlighted square is flattened into a single vector and treated as one
                        "token" — exactly like a word in a sentence. The image becomes a sequence of N tokens.
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Yellow border = patch 0 (top-left). ViT processes all {numPatches} patches as a sequence.
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="example" className="space-y-3">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">8×8 image, P=4 → 4 patches of length 16</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  <p className="text-muted-foreground">
                    To make the patch extraction fully visible, we use a tiny 8×8 RGB image and patch size 4.
                    The image splits into (8/4)² = 4 non-overlapping patches, each containing 4×4×3 = 48 numbers.
                  </p>
                  <div className="flex flex-wrap gap-3">
                    {Array.from({ length: 4 }).map((_, p) => {
                      const pi = Math.floor(p / 2);
                      const pj = p % 2;
                      const colors = ["#0f766e", "#f59e0b", "#db2777", "#7c3aed"];
                      return (
                        <div key={p} className="space-y-1">
                          <p className="text-[10px] font-mono">Patch {p}  (row {pi}, col {pj})</p>
                          <div
                            className="grid"
                            style={{
                              gridTemplateColumns: "repeat(4, 24px)",
                              gap: 1,
                              border: `2px solid ${colors[p]}`,
                              borderRadius: 4,
                              padding: 2,
                            }}
                          >
                            {Array.from({ length: 16 }).map((_, k) => {
                              const r = pi * 4 + Math.floor(k / 4);
                              const c = pj * 4 + (k % 4);
                              const px = TINY_8x8[r][c];
                              const css = `rgb(${px[0]}, ${px[1]}, ${px[2]})`;
                              return (
                                <div
                                  key={k}
                                  className="h-6 w-6 rounded-sm"
                                  style={{ backgroundColor: css }}
                                  title={`pixel (${r},${c}) = rgb(${px[0]},${px[1]},${px[2]})`}
                                />
                              );
                            })}
                          </div>
                          <p className="text-[10px] text-muted-foreground">16 pixels × 3 channels = 48</p>
                        </div>
                      );
                    })}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Each patch is flattened row-major into a vector of length{" "}
                    <MathBlock>{`P^2 \\cdot C = 4^2 \\cdot 3 = 48`}</MathBlock>. The 4 vectors become the 4-token
                    sequence fed to the ViT.
                  </p>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="math" className="space-y-3">
              <Card>
                <CardContent className="pt-4 space-y-3 text-sm">
                  <p>
                    Given image <MathBlock>{`X \\in \\mathbb{R}^{H \\times W \\times C}`}</MathBlock> and patch
                    size <MathBlock>{`P`}</MathBlock>, the number of patches is:
                  </p>
                  <MathBlock block>{`N = \\left\\lfloor \\frac{H}{P} \\right\\rfloor \\cdot \\left\\lfloor \\frac{W}{P} \\right\\rfloor`}</MathBlock>
                  <p>
                    For a square image H = W, this simplifies to <MathBlock>{`N = \\lfloor H/P \\rfloor^2`}</MathBlock>.
                    Each patch <MathBlock>{`x_p^{(k)} \\in \\mathbb{R}^{P^2 \\cdot C}`}</MathBlock> is a flattened block
                    of <MathBlock>{`P \\times P`}</MathBlock> pixels across all <MathBlock>{`C`}</MathBlock> channels.
                  </p>
                  <p className="text-muted-foreground">
                    The output is a sequence <MathBlock>{`X_p = [x_p^{(1)}, \\ldots, x_p^{(N)}] \\in \\mathbb{R}^{N \\times P^2 C}`}</MathBlock> —
                    a 2D matrix, exactly the input shape a transformer expects.
                  </p>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          <div className="grid gap-4 md:grid-cols-2">
            <ExpandableSection title="Why patches? Why not flatten the whole image?" variant="why" defaultOpen>
              <p>
                Flattening a 224×224×3 image gives a vector of length 150,528. A self-attention layer over
                this vector would need an attention matrix of size 150,528² ≈ 2.3 × 10¹⁰ entries —
                completely infeasible.
              </p>
              <p className="mt-2">
                Splitting into 16×16 patches gives N = 196 tokens — attention cost is 196² = 38,416 entries,
                a reduction factor of ~600,000×. The trade-off: we lose some within-patch spatial structure,
                but each patch is small enough (16×16) that the loss is minimal.
              </p>
            </ExpandableSection>

            <ExpandableSection title="How does patch extraction preserve locality?" variant="how">
              <p>
                Each patch keeps the 2D spatial layout of <MathBlock>{`P \\times P`}</MathBlock> pixels —
                just flattened in row-major order. Adjacent pixels in the image stay adjacent in the
                flattened vector. The transformer's first attention layer can then learn which patches
                to attend to (typically spatially neighbouring ones).
              </p>
              <p className="mt-2 text-muted-foreground">
                This is a <em>weaker</em> inductive bias than convolution (which forces locality via the
                kernel), but enough for ViTs to learn excellent visual representations from large datasets.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Intuition: image as a sentence of patches" variant="intuition">
              <p>
                A language model processes a sentence as a sequence of word tokens. ViT processes an image
                as a sequence of <em>patch tokens</em>. Just as "the", "cat", "sat" combine via attention
                to encode meaning, patches "top-left", "center", "bottom-right" combine via attention to
                encode visual content.
              </p>
              <div className="mt-2 flex items-center gap-2 text-[11px] text-muted-foreground">
                <Move className="h-3 w-3" /> <span>Slider the patch size to see how the "vocabulary" of patches changes.</span>
              </div>
            </ExpandableSection>

            <ExpandableSection title="Numerical example: 224×224, P=16" variant="numerical" defaultOpen>
              <p>
                <MathBlock>{`H = W = 224,\\ P = 16,\\ C = 3`}</MathBlock>
              </p>
              <ul className="list-disc pl-5 space-y-1 mt-2">
                <li>Patches per side: <MathBlock>{`\\lfloor 224/16 \\rfloor = 14`}</MathBlock></li>
                <li>Total patches: <MathBlock>{`N = 14^2 = 196`}</MathBlock></li>
                <li>Per-patch dimension: <MathBlock>{`P^2 C = 16^2 \\cdot 3 = 768`}</MathBlock></li>
                <li>Token sequence shape: <DimBadge dims="196 × 768" variant="output" /></li>
              </ul>
              <p className="mt-2 text-muted-foreground">
                These 768-dimensional patch vectors are exactly the input dim of ViT-Base, and 196 is the
                sequence length — the same "context window" idea as a 196-token sentence.
              </p>
            </ExpandableSection>
          </div>

          <VivaPanel
            questions={[
              {
                level: "Easy",
                question: "How many patches does a 224×224 image produce with patch size 16?",
                hint: "Compute the patches per side first.",
                answer:
                  "224 / 16 = 14 patches per side. Total N = 14 × 14 = 196 patches.",
                explanation:
                  "Each patch is 16×16 = 256 pixels. With 3 RGB channels that's 768 numbers per patch, matching ViT-Base's hidden dimension.",
              },
              {
                level: "Medium",
                question: "Why does ViT use 16×16 patches and not 4×4 or 32×32?",
                hint: "Trade-off between attention cost and spatial resolution.",
                answer:
                  "Smaller patches (P=4) give N=3136 tokens — attention cost scales as O(N²), so 3136² ≈ 10⁷ operations vs 196² for P=16. Larger patches (P=32) reduce N to 49, but each patch is too coarse to capture fine visual detail.",
                explanation:
                  "P=16 is the sweet spot discovered by Dosovitskiy et al. (2020): 196 tokens is cheap enough for O(N²) attention, and 16×16 patches still contain enough pixels (256) to preserve visual detail.",
              },
              {
                level: "Difficult",
                question:
                  "If you halve the patch size (e.g. 16→8) on a fixed 224×224 image, how does the attention cost change? Is it just 2× or 4×?",
                hint: "Attention cost is O(N²·d).",
                answer:
                  "Halving P quadruples N (224/8 = 28 → N = 784 vs 196). Attention cost scales as N², so cost increases by 4² / 1² = 16×. The constant factor is also scaled by N (since each row sums to 1, but the matmul AV is now 4× longer), giving roughly 16× for QKᵀ and 4× for AV — total ~16× in practice.",
                explanation:
                  "This is why hierarchical ViTs (Swin, PVT) use windowed attention or downsample tokens between stages: to avoid the O(N²) blowup while still using small patches for fine detail.",
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
}
