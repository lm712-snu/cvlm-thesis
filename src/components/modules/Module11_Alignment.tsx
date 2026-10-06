"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { RefreshCw } from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LabelList,
} from "recharts";
import { MatrixView } from "@/components/cvlm/MatrixView";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import { cosineSimilarity, fmtVec, type Vector } from "@/lib/math/matrix";

const D = 4;

const CAPTIONS = [
  "A dog is running",
  "A cat is sleeping",
  "A car drives fast",
  "Children play in the park",
] as const;

// Color palette for the bar chart (no indigo / blue per style guide).
const COLORS = [
  "#16a34a", // green — best match
  "#0d9488", // teal
  "#d97706", // amber
  "#dc2626", // red
];

export function Module11_Alignment() {
  const [seed, setSeed] = useState(3);

  // Synthetic image and text embeddings, d = 4. The "image" is a single pooled vector.
  // The "correct" caption for the synthetic image is index 0 ("A dog is running")
  // — but the model's encoders might not have been trained, so the bar chart reflects
  // whatever the random embeddings produce. Click "regenerate" to sample new pairs.
  const { imageVec, textVecs } = useMemo(() => {
    const rng = mulberry32(seed * 17 + 1);
    const imageVec: Vector = Array.from({ length: D }, () => rng() * 2 - 1);
    // Bias the first caption's embedding to be more aligned with the image
    // (simulating a *trained* encoder pairing).
    const textVecs: Vector[] = CAPTIONS.map((_, i) => {
      const v: Vector = Array.from({ length: D }, () => rng() * 2 - 1);
      // Make caption 0 progressively more aligned with the image
      if (i === 0) {
        for (let k = 0; k < D; k++) v[k] = 0.65 * imageVec[k] + 0.35 * v[k];
      } else if (i === 1) {
        for (let k = 0; k < D; k++) v[k] = 0.25 * imageVec[k] + 0.75 * v[k];
      }
      return v;
    });
    return { imageVec, textVecs };
  }, [seed]);

  // Compute similarities (already a perfect forward reference to Module 12 — here we use
  // cosine similarity as the alignment score).
  const ranked = useMemo(() => {
    const sims = textVecs.map((t, i) => ({
      idx: i,
      caption: CAPTIONS[i],
      sim: cosineSimilarity(imageVec, t),
    }));
    return [...sims].sort((a, b) => b.sim - a.sim);
  }, [imageVec, textVecs]);

  const chartData = ranked.map((r) => ({
    name: r.caption.length > 22 ? r.caption.slice(0, 20) + "…" : r.caption,
    fullName: r.caption,
    sim: r.sim,
    rank: ranked.indexOf(r),
  }));

  const best = ranked[0];

  return (
    <div>
      <ModuleHeader
        number={11}
        title="Image–Text Alignment (CLIP-style)"
        subtitle="Two encoders project images and text into a shared embedding space. Cosine similarity ranks candidate captions."
      >
        <MathBlock block>
          {`\\text{sim}(I, T) = \\frac{I \\cdot T}{\\|I\\|\\, \\|T\\|}, \\quad I = f_{\\text{image}}(x),\\ T = f_{\\text{text}}(t)`}
        </MathBlock>
      </ModuleHeader>

      <div className="grid gap-4 lg:grid-cols-[300px_1fr]">
        <Card className="h-fit">
          <CardHeader><CardTitle className="text-base">Shared embedding space</CardTitle></CardHeader>
          <CardContent className="space-y-3 text-sm">
            <p className="text-xs text-muted-foreground">
              Both encoders output vectors of dimension <DimBadge dims={`${D}`} variant="output" /> in the same space,
              so cosine similarity between an image vector and a text vector is well-defined.
            </p>
            <Button size="sm" variant="outline" className="w-full" onClick={() => setSeed((s) => s + 1)}>
              <RefreshCw className="h-3.5 w-3.5" /> Regenerate embeddings
            </Button>
            <Button size="sm" variant="ghost" className="w-full" onClick={() => setSeed(3)}>
              Reset to default
            </Button>
            <div className="rounded border bg-muted/30 p-2 text-xs space-y-1">
              <div className="flex justify-between"><span>Image embedding I</span><DimBadge dims={`${D}`} variant="input" /></div>
              <div className="flex justify-between"><span>Each text embedding T</span><DimBadge dims={`${D}`} variant="output" /></div>
              <div className="flex justify-between"><span>Caption candidates</span><code>{CAPTIONS.length}</code></div>
              <div className="flex justify-between"><span>Best match</span><code className="truncate max-w-[160px]">"{best.caption}"</code></div>
            </div>
            <p className="text-[10px] text-muted-foreground">
              These vectors are synthetic — in a real CLIP model, the encoders are deep neural networks trained with
              a contrastive loss (see Module 12) to make matching image-text pairs land close in this space.
            </p>
          </CardContent>
        </Card>

        <div className="space-y-4">
          {/* Encoders */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Image and text encoders → shared d-dim space</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded border bg-muted/20 p-3">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-xs uppercase text-muted-foreground">Image encoder</span>
                    <DimBadge dims="image → ℝ^d" variant="input" />
                  </div>
                  <p className="mb-2 text-[11px] text-muted-foreground">
                    Pools an image into a single vector I = {fmtVec(imageVec, 3)}
                  </p>
                  <MatrixView matrix={[imageVec]} digits={3} heatmap diverging cellSize="sm" rowLabels={["I"]} colLabels={["d0", "d1", "d2", "d3"]} />
                </div>
                <div className="rounded border bg-muted/20 p-3">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-xs uppercase text-muted-foreground">Text encoders</span>
                    <DimBadge dims="text → ℝ^d" variant="output" />
                  </div>
                  <MatrixView
                    matrix={textVecs}
                    digits={3}
                    heatmap
                    diverging
                    cellSize="xs"
                    rowLabels={CAPTIONS.map((c) => c.slice(0, 12))}
                    colLabels={["d0", "d1", "d2", "d3"]}
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Similarity ranking */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Caption ranking by cosine similarity</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="rounded border bg-muted/20 p-3">
                <p className="mb-2 text-xs text-muted-foreground">
                  sim(I, T) = (I · T) / (||I|| · ||T||). Higher = more aligned.
                </p>
                <div style={{ width: "100%", height: 220 }}>
                  <ResponsiveContainer>
                    <BarChart data={chartData} layout="vertical" margin={{ left: 30, right: 40, top: 8, bottom: 8 }}>
                      <XAxis type="number" domain={[-1, 1]} stroke="#9ca3af" fontSize={11} tickFormatter={(v) => v.toFixed(1)} />
                      <YAxis type="category" dataKey="name" stroke="#9ca3af" fontSize={11} width={130} />
                      <Tooltip
                        cursor={{ fill: "rgba(0,0,0,0.05)" }}
                        formatter={(v: number) => v.toFixed(4)}
                        labelFormatter={(_, payload) => payload && payload.length ? String(payload[0].payload.fullName) : ""}
                      />
                      <Bar dataKey="sim" radius={[0, 4, 4, 0]}>
                        {chartData.map((d, i) => (
                          <Cell key={i} fill={COLORS[d.rank]} />
                        ))}
                        <LabelList dataKey="sim" position="right" formatter={(v: number) => v.toFixed(3)} style={{ fontSize: 11, fill: "#6b7280" }} />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <p className="mt-2 text-[10px] text-muted-foreground">
                  Green = highest similarity (top match). Colors are fixed by rank; bar length encodes cosine similarity.
                </p>
              </div>

              <div className="rounded border bg-muted/20 p-3 space-y-2">
                <p className="text-xs uppercase text-muted-foreground">Ranked list</p>
                <ol className="space-y-1">
                  {ranked.map((r, i) => (
                    <li key={r.idx} className="flex items-center justify-between gap-2 text-sm">
                      <span className="flex items-center gap-2">
                        <Badge
                          variant="outline"
                          className="font-mono text-[10px]"
                          style={{ borderColor: COLORS[i], color: COLORS[i] }}
                        >
                          #{i + 1}
                        </Badge>
                        <span className="truncate">{r.caption}</span>
                      </span>
                      <code className="text-xs font-mono">{r.sim.toFixed(4)}</code>
                    </li>
                  ))}
                </ol>
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <ExpandableSection title="What is a 'shared embedding space'?" variant="why" defaultOpen>
              <p>
                Images and text live in very different worlds: one is a tensor of pixels, the other a sequence of
                tokens. To compare them, we need a common coordinate system.
              </p>
              <p className="mt-2">
                CLIP trains two separate encoders — one for images (a vision transformer or ResNet), one for text (a
                transformer) — but forces both to output vectors in the <em>same</em> <MathBlock>{`\\mathbb{R}^d`}</MathBlock>.
                Once both encoders live in the same space, distance and similarity are well-defined across modalities.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Why cosine similarity, not dot product?" variant="math" defaultOpen>
              <p>
                Dot product <MathBlock>{`I \\cdot T`}</MathBlock> depends on the magnitudes of <MathBlock>{`I`}</MathBlock>{" "}
                and <MathBlock>{`T`}</MathBlock>. A high-magnitude vector would score highly with anything, even
                semantically unrelated vectors.
              </p>
              <p className="mt-2">
                Dividing by <MathBlock>{`\\|I\\|\\,\\|T\\|`}</MathBlock> normalizes the score into{" "}
                <MathBlock>{`[-1, 1]`}</MathBlock>, measuring only the <em>angle</em> between vectors. This makes
                comparisons invariant to encoder scale, which is essential when the two encoders are architecturally
                different (their natural output norms differ).
              </p>
            </ExpandableSection>

            <ExpandableSection title="How does CLIP actually learn good encoders?" variant="how">
              <p>
                Random encoders (like this demo's) produce meaningless similarity scores. CLIP learns meaningful
                encoders via <em>contrastive learning</em>: given a batch of N image-text pairs, it pushes matched
                pairs together and pushes mismatched pairs apart using the InfoNCE loss.
              </p>
              <p className="mt-2">
                See <strong>Module 12 — Contrastive Learning</strong> for the full InfoNCE derivation, temperature
                scaling, and the symmetric image-to-text / text-to-image loss used in CLIP training.
              </p>
              <MathBlock block>
                {`\\mathcal{L}_{\\text{CLIP}} = \\tfrac{1}{2}\\big(\\mathcal{L}_{I \\to T} + \\mathcal{L}_{T \\to I}\\big), \\quad \\mathcal{L}_{I \\to T} = -\\log \\frac{\\exp(\\text{sim}(I, T^+)/\\tau)}{\\sum_{j=1}^{N} \\exp(\\text{sim}(I, T_j)/\\tau)}`}
              </MathBlock>
            </ExpandableSection>

            <ExpandableSection title="Why doesn't this demo's ranking always pick the 'dog' caption?" variant="intuition">
              <p>
                Because the text encoders in this demo are <em>untrained</em> random vectors. Without a contrastive
                training signal, there is no pressure for the correct pair to have a small angle.
              </p>
              <p className="mt-2">
                We partially bias the first caption's embedding toward the image to mimic a trained encoder, so the
                top match is usually — but not always — the "correct" pair. Hit <em>Regenerate</em> a few times: when
                the bias is overcome by noise, you'll see an unrelated caption win, exactly as a freshly initialized
                CLIP would behave before training.
              </p>
            </ExpandableSection>
          </div>

          <VivaPanel
            questions={[
              {
                level: "Easy",
                question: "What does it mean for an image and a text to 'live in the same embedding space'?",
                hint: "Think about what operations become valid when both are vectors of the same shape.",
                answer:
                  "Both the image encoder and the text encoder output vectors of the same dimension d. Because they share a coordinate system, geometric operations like dot product, cosine similarity, and Euclidean distance become meaningful across modalities.",
                explanation:
                  "Without a shared space, asking 'how similar is this image to this sentence?' is undefined. The encoders are the bridge: they translate pixels and tokens into a common ℝ^d where alignment can be measured.",
              },
              {
                level: "Medium",
                question: "Why use cosine similarity instead of raw dot product or Euclidean distance?",
                hint: "What happens when one encoder produces vectors with much larger magnitude than the other?",
                answer:
                  "Cosine similarity is invariant to vector magnitude, so a high-norm encoder cannot dominate the ranking. It measures only the angle between vectors, which is what semantic alignment actually corresponds to. Dot product conflates alignment with magnitude; Euclidean distance conflates alignment with both magnitude and origin offset.",
                explanation:
                  "CLIP normalizes embeddings to unit length before computing similarities, which makes dot product and cosine similarity numerically identical — but the design choice matters in unnormalized settings and motivates the architecture of the loss.",
              },
              {
                level: "Difficult",
                question:
                  "CLIP is trained on roughly 400M image-text pairs scraped from the web. Why is the contrastive objective sufficient to produce semantically meaningful embeddings, given that the supervision signal is just 'these came from the same webpage'?",
                hint: "Consider what the model has to learn to minimize the contrastive loss at scale.",
                answer:
                  "The contrastive loss only requires that matched pairs have higher similarity than mismatched ones. To satisfy this across a vast and diverse dataset, the model must learn features that genuinely distinguish the matched caption (objects, attributes, scene type, spatial relationships) from arbitrary alternatives. The web's natural distribution of image-text pairs provides enough signal that semantic concepts become the lowest-entropy way to separate positives from negatives.",
                explanation:
                  "The key is scale: a single noisy pair (e.g. an image of a dog with the caption 'cute puppy') tells the model little, but millions of such pairs force the model to discover that dogs, cuteness, and puppies co-occur. The model cannot cheat by memorizing specific pairs because the same concept appears across many different image-text instantiations. The contrastive loss becomes a powerful representation-learning objective precisely because it does not need labels — only co-occurrence.",
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
}

// Local copy of a deterministic PRNG (kept inside this module so we don't add an export to the math library).
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
