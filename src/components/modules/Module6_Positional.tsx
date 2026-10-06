"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Heatmap } from "@/components/cvlm/Heatmap";
import { MatrixView } from "@/components/cvlm/MatrixView";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import { sinusoidalPE } from "@/lib/math/attention";
import { randMatrix, shape } from "@/lib/math/matrix";
import { Plus, Equal, MousePointerClick } from "lucide-react";

export function Module6_Positional() {
  // PE configuration sliders.
  const [d, setD] = useState(16);
  const [maxPos, setMaxPos] = useState(16);
  // Selected cell for inspection.
  const [sel, setSel] = useState<[number, number] | null>(null);

  // For the "before/after" demo we use a small patch embedding matrix
  // of shape (8, d) so it lines up with PE's column dim d.
  const demoN = 8;
  const pe = useMemo(() => sinusoidalPE(maxPos, d), [maxPos, d]);
  const patchEmb = useMemo(() => randMatrix(Math.min(demoN, maxPos), d, -1, 1, 13), [d, maxPos]);
  // Take the first N=min(demoN, maxPos) rows of PE so shapes match.
  const peForDemo = useMemo(
    () => pe.slice(0, Math.min(demoN, maxPos)).map((r) => r.slice()),
    [pe, maxPos, d]
  );
  const combined = useMemo(
    () => patchEmb.map((row, i) => row.map((v, j) => v + peForDemo[i][j])),
    [patchEmb, peForDemo]
  );

  const selected = sel
    ? (() => {
        const [pos, idx] = sel;
        const i = Math.floor(idx / 2);
        const isSin = idx % 2 === 0;
        const denom = Math.pow(10000, (2 * i) / d);
        const arg = pos / denom;
        return { pos, idx, i, isSin, denom, arg, value: pe[pos][idx] };
      })()
    : null;

  return (
    <div>
      <ModuleHeader
        number={6}
        title="Positional Embeddings"
        subtitle="Patch embeddings alone are permutation-invariant — positional encoding tells the transformer where each patch came from."
      >
        <MathBlock block>
          {`PE_{(pos,\\,2i)} = \\sin\\!\\left(\\frac{pos}{10000^{2i/d}}\\right), \\quad PE_{(pos,\\,2i+1)} = \\cos\\!\\left(\\frac{pos}{10000^{2i/d}}\\right)`}
        </MathBlock>
      </ModuleHeader>

      <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
        <Card className="h-fit">
          <CardHeader><CardTitle className="text-base">Configuration</CardTitle></CardHeader>
          <CardContent className="space-y-4 text-sm">
            <div>
              <div className="flex justify-between"><Label className="text-xs">Embed dim d</Label><span className="font-mono text-xs">{d}</span></div>
              <Slider value={[d]} min={4} max={32} step={2} onValueChange={([v]) => { setD(v ?? 16); setSel(null); }} />
            </div>
            <div>
              <div className="flex justify-between"><Label className="text-xs">Max positions</Label><span className="font-mono text-xs">{maxPos}</span></div>
              <Slider value={[maxPos]} min={8} max={32} step={1} onValueChange={([v]) => { setMaxPos(v ?? 16); setSel(null); }} />
            </div>
            <div className="rounded border bg-muted/30 p-2 text-xs space-y-1">
              <div className="flex justify-between"><span>PE matrix</span><DimBadge dims={`${maxPos} × ${d}`} variant="intermediate" /></div>
              <div className="flex justify-between"><span>Value range</span><span className="font-mono">[-1, +1]</span></div>
              <div className="flex justify-between"><span>Frequencies</span><span className="font-mono">{Math.floor(d / 2)}</span></div>
              <div className="flex justify-between"><span>Sinusoid period (longest)</span><span className="font-mono">2π·10000</span></div>
            </div>
            <p className="rounded border bg-muted/20 p-2 text-[11px] text-muted-foreground">
              Columns alternate sin/cos pairs. Even columns = sin, odd columns = cos of the same frequency.
            </p>
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-2 flex flex-row items-center justify-between">
              <CardTitle className="text-sm">Sinusoidal positional encoding — click any cell</CardTitle>
              <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                <MousePointerClick className="h-3 w-3" /> click to inspect
              </span>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-3">
                <Heatmap
                  matrix={pe}
                  diverging
                  min={-1}
                  max={1}
                  cellSize={Math.max(12, Math.min(28, Math.floor(360 / d)))}
                  rowLabels={Array.from({ length: maxPos }, (_, i) => `pos ${i}`)}
                  colLabels={Array.from({ length: d }, (_, j) => `d${j}`)}
                  highlight={sel ?? undefined}
                  onCellClick={(i, j) => setSel([i, j])}
                  format={() => ""}
                />
                {/* Color legend */}
                <div className="flex flex-col gap-1 text-[10px]">
                  <div className="flex items-center gap-1">
                    <div className="h-3 w-6" style={{ background: "rgb(255, 0, 0)" }} />
                    <span>-1 (red)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="h-3 w-6" style={{ background: "rgb(255, 255, 255)" }} />
                    <span>0 (white)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="h-3 w-6" style={{ background: "rgb(0, 0, 255)" }} />
                    <span>+1 (blue)</span>
                  </div>
                  <p className="text-muted-foreground mt-2 max-w-[140px]">
                    Horizontal stripes near the left = high-frequency (fast change with position).
                    Near the right = low-frequency (slow change).
                  </p>
                </div>
              </div>

              {selected && (
                <div className="mt-4 rounded border-2 border-primary/40 bg-primary/5 p-3 text-xs font-mono space-y-1">
                  <p className="font-medium">PE[pos={selected.pos}, dim={selected.idx}]</p>
                  <p>
                    i = ⌊{selected.idx}/2⌋ = {selected.i}, so this is a{" "}
                    <span className="text-primary">{selected.isSin ? "sin" : "cos"}</span> cell
                  </p>
                  <p>denominator = 10000^(2·{selected.i}/{d}) = {selected.denom.toFixed(4)}</p>
                  <p>argument = pos/denom = {selected.pos}/{selected.denom.toFixed(4)} = {selected.arg.toFixed(4)}</p>
                  <p className="text-primary">
                    {selected.isSin ? "sin" : "cos"}({selected.arg.toFixed(4)}) = <strong>{selected.value.toFixed(6)}</strong>
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-sm">Adding PE to patch embeddings: Z = P + PE</CardTitle></CardHeader>
            <CardContent>
              <div className="grid gap-3 md:grid-cols-[1fr_auto_1fr_auto_1fr] md:items-center">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <p className="text-xs uppercase text-muted-foreground">Patch Embeddings P</p>
                    <DimBadge dims={`${Math.min(demoN, maxPos)} × ${d}`} variant="input" />
                  </div>
                  <Heatmap
                    matrix={patchEmb}
                    diverging
                    cellSize={Math.max(12, Math.min(28, Math.floor(220 / d)))}
                    format={() => ""}
                  />
                </div>

                <div className="flex items-center justify-center">
                  <Plus className="h-6 w-6 text-muted-foreground" />
                </div>

                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <p className="text-xs uppercase text-muted-foreground">Positional Enc. PE</p>
                    <DimBadge dims={`${Math.min(demoN, maxPos)} × ${d}`} variant="intermediate" />
                  </div>
                  <Heatmap
                    matrix={peForDemo}
                    diverging
                    min={-1}
                    max={1}
                    cellSize={Math.max(12, Math.min(28, Math.floor(220 / d)))}
                    format={() => ""}
                  />
                </div>

                <div className="flex items-center justify-center">
                  <Equal className="h-6 w-6 text-muted-foreground" />
                </div>

                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <p className="text-xs uppercase text-muted-foreground">Combined Z</p>
                    <DimBadge dims={`${Math.min(demoN, maxPos)} × ${d}`} variant="output" />
                  </div>
                  <Heatmap
                    matrix={combined}
                    diverging
                    cellSize={Math.max(12, Math.min(28, Math.floor(220 / d)))}
                    format={() => ""}
                  />
                </div>
              </div>
              <p className="mt-3 text-[11px] text-muted-foreground">
                Element-wise addition. The patch content (P) now carries positional fingerprints (PE) — the
                transformer can disambiguate "patch in top-left" from "patch in bottom-right" by reading PE.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-sm">Numeric Z (first few rows)</CardTitle></CardHeader>
            <CardContent>
              <MatrixView
                matrix={combined.slice(0, Math.min(6, combined.length))}
                digits={2}
                heatmap
                diverging
                cellSize="xs"
              />
              <p className="mt-2 text-[10px] text-muted-foreground">
                Each row is one patch's combined embedding. Spatial layout is now baked into the values.
              </p>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <ExpandableSection title="Why does a transformer need positional information?" variant="why" defaultOpen>
              <p>
                Self-attention (Module 8) treats its input as a <em>set</em> — it has no built-in notion of
                order. Permuting the patches would produce the same output (just permuted). For images, the
                spatial location of each patch is crucial: a "cat ear" patch in the top-right means very
                different things than one in the bottom-left.
              </p>
              <p className="mt-2">
                Positional encoding injects this information by adding a position-dependent vector to each
                patch embedding before attention sees it.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Why sin and cos of different frequencies?" variant="math" defaultOpen>
              <p>
                Each dimension pair (2i, 2i+1) encodes position using a sinusoid of frequency{" "}
                <MathBlock>{`\\omega_i = 1/10000^{2i/d}`}</MathBlock>. Low <MathBlock>{`i`}</MathBlock> →
                high frequency (changes fast with position); high <MathBlock>{`i`}</MathBlock> → low
                frequency (changes slowly).
              </p>
              <p className="mt-2">
                The combination of many frequencies lets the network <strong>uniquely identify positions</strong>{" "}
                from a low-dim vector — like a Fourier basis. Long wavelengths disambiguate macro position;
                short wavelengths disambiguate micro position.
              </p>
            </ExpandableSection>

            <ExpandableSection title="PE(pos+k) is a linear function of PE(pos)" variant="how">
              <p>
                A neat property of the sinusoidal form: for any fixed offset <MathBlock>{`k`}</MathBlock>,
                <MathBlock>{`PE(pos + k)`}</MathBlock> can be written as a linear function of{" "}
                <MathBlock>{`PE(pos)`}</MathBlock> — a 2D rotation per frequency pair.
              </p>
              <p className="mt-2">
                This means the network can learn to attend to <em>relative</em> positions ("look 2 patches
                to the right") by learning a linear projection on Q and K. This is why sinusoidal PE
                generalizes to longer sequences than the model was trained on.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Learned vs sinusoidal PE" variant="intuition">
              <p>
                The original Transformer and ViT papers used sinusoidal PE (no parameters). Many follow-ups
                (BERT, GPT, ViT-Large) use <em>learned</em> positional embeddings: a single{" "}
                <MathBlock>{`(N_{\\max} \\times d)`}</MathBlock> matrix of free parameters.
              </p>
              <p className="mt-2 text-muted-foreground">
                Sinusoidal PE has zero parameters and extrapolates to longer sequences; learned PE often
                gives marginally better accuracy but cannot exceed N_max. Recent large models (LLaMA, SigLIP)
                increasingly use rotary positional embeddings (RoPE), which bake rotation into Q/K directly.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Numerical example: d=4, pos=3" variant="numerical">
              <p>
                With <MathBlock>{`d = 4`}</MathBlock> there are 2 frequency pairs:
              </p>
              <ul className="list-disc pl-5 space-y-1 mt-2 text-xs font-mono">
                <li>i=0: denom = 10000^(0/4) = 1 → sin(3), cos(3)</li>
                <li>i=1: denom = 10000^(2/4) = 100 → sin(3/100), cos(3/100)</li>
              </ul>
              <p className="mt-2">
                <MathBlock block>{`PE(3, \\cdot) = [\\sin 3,\\ \\cos 3,\\ \\sin 0.03,\\ \\cos 0.03] = [0.141, -0.990, 0.030, 1.000]`}</MathBlock>
              </p>
              <p className="text-xs text-muted-foreground">
                The first two dimensions vary fast (sin/cos of integer positions); the last two barely change
                — they encode "macro" position.
              </p>
            </ExpandableSection>

            <ExpandableSection title="How 2D position is handled for images" variant="how">
              <p>
                Patches form a 2D grid, but standard PE is 1D. ViTs handle this by flattening the grid
                row-major (so position = row·gridW + col) and applying 1D PE. This is fine because attention
                is permutation-equivariant — the 1D PE just needs to disambiguate patches.
              </p>
              <p className="mt-2 text-muted-foreground">
                Some ViT variants (e.g. Swin, CvT) use explicit 2D PE (separate sin/cos for row and column),
                which can give marginal gains on small datasets.
              </p>
            </ExpandableSection>
          </div>

          <VivaPanel
            questions={[
              {
                level: "Easy",
                question: "Why does a transformer need positional encoding at all?",
                hint: "Consider what attention does to a permuted input.",
                answer:
                  "Self-attention is permutation-equivariant — it treats its input as a set. Without positional information, the model could not distinguish 'patch 5' from 'patch 17'. PE injects positional information so the transformer knows where each patch came from.",
                explanation:
                  "Convolutional networks don't have this issue because the convolution operation is position-aware. Transformers trade this inductive bias for generality, and recover position via PE.",
              },
              {
                level: "Medium",
                question:
                  "Why use both sin and cos, instead of just sin for every dimension?",
                hint: "Think about relative position and linear combinations.",
                answer:
                  "Using only sin would still encode position, but the 'PE(pos+k) is linear in PE(pos)' property requires the sin/cos pair to form a 2D rotation matrix per frequency. The cos component is what makes this rotation possible.",
                explanation:
                  "Each (sin, cos) pair at a given frequency is a 2D unit vector parametrized by angle = pos/denom. Shifting pos by k rotates both entries by k/denom, which is a linear map. Without the cos, you'd lose the rotation structure and the relative-position learnability.",
              },
              {
                level: "Difficult",
                question:
                  "Show that PE(pos+k) is a linear function of PE(pos). Why is this important?",
                hint: "Use the angle-addition formulas for sin and cos.",
                answer:
                  "For each frequency pair (sin(ωpos), cos(ωpos)), shifting by k gives (sin(ω(pos+k)), cos(ω(pos+k))) = M_k · (sin(ωpos), cos(ωpos)) where M_k is the 2D rotation matrix [[cos(ωk), -sin(ωk)], [sin(ωk), cos(ωk)]]. So PE(pos+k) = R_k · PE(pos) where R_k is a block-diagonal rotation. This means the network can learn to attend to relative positions (e.g. 'patch k to the right') with a single linear projection on Q and K.",
                explanation:
                  "This relative-position property is why sinusoidal PE generalizes to sequences longer than those seen in training — a key advantage over learned PE tables that have a fixed maximum length.",
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
}
