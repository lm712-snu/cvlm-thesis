"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { RefreshCw, Dice5 } from "lucide-react";
import { MatrixView } from "@/components/cvlm/MatrixView";
import { Heatmap } from "@/components/cvlm/Heatmap";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import { infoNCE } from "@/lib/math/attention";
import { softmaxVec, transpose, cosineSimilarity, type Matrix, type Vector } from "@/lib/math/matrix";

const B = 3; // batch size: 3 images x 3 texts
const D = 4; // embedding dimension

const IMAGE_LABELS = ["img0: dog", "img1: cat", "img2: car"];
const TEXT_LABELS = [
  "txt0: a dog runs",
  "txt1: a cat sleeps",
  "txt2: a car drives",
];

export function Module12_Contrastive() {
  const [tau, setTau] = useState(0.1);
  const [seed, setSeed] = useState(5);

  // Synthetic embeddings: each image is somewhat aligned with its matched text (diagonal),
  // but with noise so the loss is non-trivial.
  const { imageEmb, textEmb } = useMemo(() => {
    const rng = mulberry32(seed * 23 + 7);
    const imageEmb: Vector[] = Array.from({ length: B }, () =>
      Array.from({ length: D }, () => rng() * 2 - 1)
    );
    // Text: each text_i is a noisy copy of image_i plus some random direction.
    const textEmb: Vector[] = imageEmb.map((v) => {
      const t: Vector = v.map((x) => 0.6 * x + (rng() * 2 - 1) * 0.6);
      return t;
    });
    return { imageEmb, textEmb };
  }, [seed]);

  // 3x3 similarity matrix S[i][j] = cos(image_i, text_j).
  const sim: Matrix = useMemo(() => {
    const S: Matrix = Array.from({ length: B }, () => new Array(B).fill(0));
    for (let i = 0; i < B; i++) {
      for (let j = 0; j < B; j++) {
        S[i][j] = cosineSimilarity(imageEmb[i], textEmb[j]);
      }
    }
    return S;
  }, [imageEmb, textEmb]);

  // Use the library function for the canonical loss.
  const { loss: libLoss, rowProbs } = useMemo(() => infoNCE(sim, tau), [sim, tau]);

  // Manually compute the same thing for the step-by-step view.
  const manual = useMemo(() => {
    // Row-wise softmax of (S / tau): for each image, distribution over texts.
    const rowScaled = sim.map((r) => r.map((v) => v / tau));
    const rowSoftmax = rowScaled.map((r) => softmaxVec(r));
    // Column-wise softmax: for each text, distribution over images.
    const simT = transpose(sim);
    const colScaled = simT.map((r) => r.map((v) => v / tau));
    const colSoftmax = colScaled.map((r) => softmaxVec(r)); // shape (B, B), but row index = text
    const colSoftmaxAsMatrix = transpose(colSoftmax); // back to (image, text) ordering

    // L_i2t = -log(rowSoftmax[i][i])
    const Li2t = rowSoftmax.map((r, i) => -Math.log(Math.max(r[i], 1e-12)));
    // L_t2i = -log(colSoftmax[i][i]) where colSoftmax[text][image]
    const Lt2i = colSoftmax.map((r, i) => -Math.log(Math.max(r[i], 1e-12)));
    const L_i2t = Li2t.reduce((a, b) => a + b, 0) / B;
    const L_t2i = Lt2i.reduce((a, b) => a + b, 0) / B;
    const totalLoss = (L_i2t + L_t2i) / 2;
    return {
      rowSoftmax,
      colSoftmaxAsMatrix,
      Li2t,
      Lt2i,
      L_i2t,
      L_t2i,
      totalLoss,
    };
  }, [sim, tau]);

  const [hovered, setHovered] = useState<[number, number] | null>(null);

  return (
    <div>
      <ModuleHeader
        number={12}
        title="Contrastive Learning (InfoNCE)"
        subtitle="Train encoders by pulling matched pairs together and pushing mismatched pairs apart. The CLIP objective in full."
      >
        <MathBlock block>
          {`\\mathcal{L} = \\tfrac{1}{2}(\\mathcal{L}_{I \\to T} + \\mathcal{L}_{T \\to I}), \\quad \\mathcal{L}_{I \\to T} = -\\frac{1}{B}\\sum_{i=1}^{B} \\log \\frac{\\exp(s_{ii}/\\tau)}{\\sum_{j=1}^{B} \\exp(s_{ij}/\\tau)}`}
        </MathBlock>
      </ModuleHeader>

      <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
        <Card className="h-fit">
          <CardHeader><CardTitle className="text-base">Configuration</CardTitle></CardHeader>
          <CardContent className="space-y-4 text-sm">
            <div>
              <div className="flex justify-between">
                <Label className="text-xs">Temperature τ</Label>
                <span className="font-mono text-xs">{tau.toFixed(3)}</span>
              </div>
              <Slider
                value={[tau]}
                min={0.01}
                max={1}
                step={0.01}
                onValueChange={([v]) => setTau(v ?? 0.1)}
              />
              <p className="mt-1 text-[10px] text-muted-foreground">
                Lower τ sharpens the softmax; higher τ flattens it.
              </p>
            </div>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={() => setSeed((s) => s + 1)}>
                <Dice5 className="h-3.5 w-3.5" /> New batch
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setSeed(5)}>
                <RefreshCw className="h-3.5 w-3.5" /> Reset
              </Button>
            </div>
            <div className="rounded border bg-muted/30 p-2 text-xs space-y-1">
              <div className="flex justify-between"><span>Batch size B</span><code>{B}</code></div>
              <div className="flex justify-between"><span>Embedding dim d</span><code>{D}</code></div>
              <div className="flex justify-between"><span>Sim matrix S</span><DimBadge dims={`${B} x ${B}`} variant="intermediate" /></div>
              <div className="flex justify-between"><span>Library loss</span><code>{libLoss.toFixed(4)}</code></div>
              <div className="flex justify-between"><span>Manual loss</span><code>{manual.totalLoss.toFixed(4)}</code></div>
              <div className="flex justify-between border-t pt-1 mt-1">
                <span>Loss L (avg of both)</span>
                <DimBadge dims={manual.totalLoss.toFixed(3)} variant="label" />
              </div>
            </div>
            <p className="text-[10px] text-muted-foreground">
              The diagonal of S (image_i vs text_i) is the <em>positive</em> pair. InfoNCE pushes
              diagonal probabilities toward 1.
            </p>
          </CardContent>
        </Card>

        <div className="space-y-4">
          {/* Embeddings */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Batch: {B} images × {B} texts (d = {D})</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="rounded border bg-muted/20 p-3">
                  <p className="mb-1 text-[10px] uppercase text-muted-foreground">Image embeddings I</p>
                  <MatrixView
                    matrix={imageEmb}
                    digits={2}
                    heatmap
                    diverging
                    cellSize="xs"
                    rowLabels={IMAGE_LABELS}
                    colLabels={["d0", "d1", "d2", "d3"]}
                  />
                </div>
                <div className="rounded border bg-muted/20 p-3">
                  <p className="mb-1 text-[10px] uppercase text-muted-foreground">Text embeddings T</p>
                  <MatrixView
                    matrix={textEmb}
                    digits={2}
                    heatmap
                    diverging
                    cellSize="xs"
                    rowLabels={TEXT_LABELS}
                    colLabels={["d0", "d1", "d2", "d3"]}
                  />
                </div>
              </div>
              <p className="text-[10px] text-muted-foreground">
                Synthetic — each text_i is a noisy copy of image_i, so the diagonal pairs are the easiest positives.
              </p>
            </CardContent>
          </Card>

          {/* Similarity matrix */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Step 1 — Similarity matrix S[i][j] = cos(image_i, text_j)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="rounded border bg-muted/20 p-3">
                <Heatmap
                  matrix={sim}
                  diverging
                  cellSize={44}
                  rowLabels={IMAGE_LABELS}
                  colLabels={TEXT_LABELS.map((t) => t.split(":")[0])}
                  format={(v) => v.toFixed(2)}
                  highlight={hovered ?? undefined}
                  onCellClick={(i, j) => setHovered([i, j])}
                />
                <div className="mt-2 flex flex-wrap gap-1">
                  {TEXT_LABELS.map((t, j) => (
                    <Badge key={j} variant="outline" className="text-[10px] font-mono">{t}</Badge>
                  ))}
                </div>
                <p className="mt-2 text-[10px] text-muted-foreground">
                  Diagonal cells (image_i ↔ text_i) are the <strong>positive</strong> pairs — the loss wants these
                  to be high relative to off-diagonal cells.
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Softmax distributions */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Step 2 — Softmax(S/τ) row-wise (I→T) and column-wise (T→I)</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="rounded border bg-muted/20 p-3">
                  <p className="mb-1 text-[10px] uppercase text-muted-foreground">
                    P(text | image_i) — row-wise softmax over texts
                  </p>
                  <Heatmap
                    matrix={manual.rowSoftmax}
                    min={0}
                    max={1}
                    cellSize={36}
                    rowLabels={IMAGE_LABELS}
                    colLabels={["txt0", "txt1", "txt2"]}
                    format={(v) => v.toFixed(2)}
                    highlight={hovered ?? undefined}
                    onCellClick={(i, j) => setHovered([i, j])}
                  />
                  <p className="mt-1 text-[10px] text-muted-foreground">
                    Each row is a probability distribution over the 3 texts. The diagonal is the target.
                  </p>
                </div>
                <div className="rounded border bg-muted/20 p-3">
                  <p className="mb-1 text-[10px] uppercase text-muted-foreground">
                    P(image | text_j) — column-wise softmax over images
                  </p>
                  <Heatmap
                    matrix={manual.colSoftmaxAsMatrix}
                    min={0}
                    max={1}
                    cellSize={36}
                    rowLabels={IMAGE_LABELS}
                    colLabels={["txt0", "txt1", "txt2"]}
                    format={(v) => v.toFixed(2)}
                    highlight={hovered ?? undefined}
                    onCellClick={(i, j) => setHovered([i, j])}
                  />
                  <p className="mt-1 text-[10px] text-muted-foreground">
                    Each column is a probability distribution over the 3 images.
                  </p>
                </div>
              </div>
              <div className="rounded border bg-primary/5 p-3">
                <p className="text-xs">
                  <strong>Effect of temperature:</strong> at τ = {tau.toFixed(2)}, the softmax is{" "}
                  {tau < 0.2 ? "very sharp — even small similarity differences produce nearly one-hot distributions" : ""}
                  {tau >= 0.2 && tau < 0.5 ? "moderately peaked" : ""}
                  {tau >= 0.5 ? "quite flat — distributions are close to uniform" : ""}.
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Loss computation */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Step 3 — Loss: -log(p_positive), averaged over batch and direction</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="overflow-x-auto">
                <table className="w-full border-collapse text-xs">
                  <thead>
                    <tr className="border-b">
                      <th className="px-2 py-1 text-left">Pair i</th>
                      <th className="px-2 py-1 text-right">P(text_i | img_i)</th>
                      <th className="px-2 py-1 text-right">L_i2t = -log(p)</th>
                      <th className="px-2 py-1 text-right">P(img_i | text_i)</th>
                      <th className="px-2 py-1 text-right">L_t2i = -log(p)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {manual.Li2t.map((_, i) => (
                      <tr key={i} className="border-b">
                        <td className="px-2 py-1">{IMAGE_LABELS[i].split(":")[0]} ↔ {TEXT_LABELS[i].split(":")[0]}</td>
                        <td className="px-2 py-1 text-right font-mono">{manual.rowSoftmax[i][i].toFixed(4)}</td>
                        <td className="px-2 py-1 text-right font-mono">{manual.Li2t[i].toFixed(4)}</td>
                        <td className="px-2 py-1 text-right font-mono">{manual.colSoftmaxAsMatrix[i][i].toFixed(4)}</td>
                        <td className="px-2 py-1 text-right font-mono">{manual.Lt2i[i].toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                  <tfoot>
                    <tr className="border-t-2 font-semibold">
                      <td className="px-2 py-1">Mean</td>
                      <td className="px-2 py-1 text-right font-mono">—</td>
                      <td className="px-2 py-1 text-right font-mono">{manual.L_i2t.toFixed(4)}</td>
                      <td className="px-2 py-1 text-right font-mono">—</td>
                      <td className="px-2 py-1 text-right font-mono">{manual.L_t2i.toFixed(4)}</td>
                    </tr>
                  </tfoot>
                </table>
              </div>
              <div className="rounded border-2 border-primary/40 bg-primary/5 p-3">
                <p className="font-mono text-sm">
                  L = (L_I→T + L_T→I) / 2 = ({manual.L_i2t.toFixed(4)} + {manual.L_t2i.toFixed(4)}) / 2 ={" "}
                  <span className="font-bold text-primary">{manual.totalLoss.toFixed(4)}</span>
                </p>
                <p className="mt-1 text-[10px] text-muted-foreground">
                  Library cross-check (row-wise only, one direction): <code>{libLoss.toFixed(4)}</code>.
                  The full CLIP loss is symmetric, so it is roughly the average of both directions.
                </p>
              </div>
              {hovered && (
                <div className="rounded border bg-muted/30 p-2 text-[11px] font-mono">
                  Cell ({hovered[0]}, {hovered[1]}): sim = {sim[hovered[0]][hovered[1]].toFixed(4)} →
                  {" "}scaled = {(sim[hovered[0]][hovered[1]] / tau).toFixed(4)} →
                  {" "}P(text | img_{hovered[0]}) = {manual.rowSoftmax[hovered[0]][hovered[1]].toFixed(4)}
                </div>
              )}
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <ExpandableSection title="What is the diagonal and why is it special?" variant="why" defaultOpen>
              <p>
                In the InfoNCE batch, the diagonal entries <MathBlock>{`S_{ii}`}</MathBlock> are similarities between
                matched (positive) image-text pairs. Off-diagonal entries <MathBlock>{`S_{ij}, i \\ne j`}</MathBlock>{" "}
                are similarities between mismatched (negative) pairs.
              </p>
              <p className="mt-2">
                The loss is minimized when every row of the softmax has all its mass on the diagonal — i.e. each
                image is more similar to its matching text than to any other text in the batch.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Why temperature τ?" variant="math" defaultOpen>
              <p>
                The raw cosine similarities <MathBlock>{`S_{ij} \\in [-1, 1]`}</MathBlock> live in a narrow range.
                Feeding them directly into softmax produces nearly uniform distributions — gradients vanish.
              </p>
              <p className="mt-2">
                Dividing by <MathBlock>{`\\tau < 1`}</MathBlock> amplifies the differences: small similarity gaps
                become large logit gaps, sharpening the distribution. <MathBlock>{`\\tau`}</MathBlock> is a
                hyperparameter (often learnable in CLIP) that controls how strict the contrastive task is.
              </p>
              <p className="mt-2">
                As <MathBlock>{`\\tau \\to 0`}</MathBlock>, the softmax approaches argmax and the loss becomes the
                hard "is the positive the maximum?" 0/1 loss. As <MathBlock>{`\\tau \\to \\infty`}</MathBlock>, the
                softmax becomes uniform and the loss saturates at <MathBlock>{`\\log B`}</MathBlock>.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Why symmetrize over both directions?" variant="how">
              <p>
                <MathBlock>{`\\mathcal{L}_{I \\to T}`}</MathBlock> asks: given an image, can the model pick its
                matching text? <MathBlock>{`\\mathcal{L}_{T \\to I}`}</MathBlock> asks the reverse: given a text, can
                it pick its image?
              </p>
              <p className="mt-2">
                These are <em>not</em> equivalent in general — a batch could be easy in one direction and hard in the
                other (e.g. one image is similar to several texts, but each text matches only one image). CLIP
                averages both to ensure the learned embedding space is useful for retrieval in either direction.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Connection to cross-entropy classification" variant="intuition">
              <p>
                InfoNCE <em>is</em> cross-entropy classification — the only twist is that the classes are
                "the other items in the batch". The image is the input, the B texts are the B classes, and the
                correct class is the matching text.
              </p>
              <p className="mt-2">
                This is why contrastive learning works so well: it converts representation learning into a familiar
                classification problem with a well-understood loss, gradient flow, and the right inductive biases
                (softmax + cross-entropy).
              </p>
            </ExpandableSection>
          </div>

          <VivaPanel
            questions={[
              {
                level: "Easy",
                question: "In the InfoNCE loss, which entries of the similarity matrix are 'positive' and which are 'negative'?",
                hint: "Think about the structure of the batch.",
                answer:
                  "The diagonal entries S_ii (image_i matched with its own text_i) are positives. All off-diagonal entries S_ij with i != j are negatives — image_i paired with the wrong text.",
                explanation:
                  "This is why the batch size matters: with B images and B texts, each image has 1 positive and B-1 negatives. Larger batches give more negatives and stronger gradients, which is one reason CLIP was trained with very large batches (thousands).",
              },
              {
                level: "Medium",
                question: "What does the temperature τ control, and what happens at the extremes τ → 0 and τ → ∞?",
                hint: "Think about the shape of softmax as you scale its inputs.",
                answer:
                  "τ scales the logits before softmax. At τ → 0, softmax approaches argmax — the loss becomes a hard 0/1 classification loss (only the top similarity matters). At τ → ∞, softmax becomes uniform and the loss saturates at log(B), the maximum entropy value — no learning signal.",
                explanation:
                  "Smaller τ makes the contrastive task harder (only the very-best match counts), which can sharpen representations but also risks instability. Larger τ makes it easier but less informative. CLIP makes τ a learnable parameter so the model can choose its own strictness.",
              },
              {
                level: "Difficult",
                question:
                  "InfoNCE uses softmax over the batch. Why is it called 'noise-contrastive estimation', and what does the 'noise' refer to?",
                hint: "The original NCE paper framed this as a binary classification problem.",
                answer:
                  "NCE recovers density estimation as a binary classification problem: distinguish samples from the true data distribution (the positive) from samples drawn from a noise distribution (the negatives). In InfoNCE, the positive is the matched pair and the negatives are all other in-batch pairs — they serve as samples from the 'noise' distribution of mismatched pairs.",
                explanation:
                  "The 'Info' in InfoNCE refers to a lower bound on mutual information: minimizing this loss maximizes a lower bound on the mutual information between the image representation and the text representation (for the matched pairs). The math connects to the InfoMax principle — learn representations that retain maximal information about a paired signal. The 'noise' negatives are essential because they define what 'informative' means: not just any text, but the specific matching text.",
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
}

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
