"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RefreshCw } from "lucide-react";
import { MatrixView } from "@/components/cvlm/MatrixView";
import { Heatmap } from "@/components/cvlm/Heatmap";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import { sinusoidalPE } from "@/lib/math/attention";
import { randMatrix, shape } from "@/lib/math/matrix";

const D = 4; // embedding dimension
const DEFAULT_SENTENCE = "A dog is running";

// Tiny vocabulary of V = 10 words. Index = token id.
const VOCAB = [
  "A", "dog", "is", "running", "cat",
  "the", "sleeps", "barks", "fast", "jumps",
] as const;
const V = VOCAB.length;

function tokenizeBySpace(sentence: string): string[] {
  return sentence
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => w.replace(/[^A-Za-z]/g, "")) // strip punctuation for the toy tokenizer
    .filter(Boolean);
}

function lookupId(token: string): number | null {
  const t = token.charAt(0).toUpperCase() + token.slice(1).toLowerCase();
  const idx = VOCAB.indexOf(t as (typeof VOCAB)[number]);
  return idx === -1 ? null : idx;
}

export function Module10_TextRepresentation() {
  const [sentence, setSentence] = useState(DEFAULT_SENTENCE);
  const [committed, setCommitted] = useState(DEFAULT_SENTENCE);
  const [seed, setSeed] = useState(11);

  // Embedding matrix E: V x D, fixed across renders per seed.
  const E = useMemo(() => randMatrix(V, D, -1, 1, seed * 31 + 5), [seed]);

  // Positional encoding matrix: rows = max positions, cols = D.
  // We compute one large enough and slice.
  const PE = useMemo(() => sinusoidalPE(16, D), []);

  const steps = useMemo(() => {
    const rawTokens = tokenizeBySpace(committed);
    const tokens = rawTokens.slice(0, 8); // cap length for readability
    const ids: number[] = [];
    const oov: boolean[] = [];
    for (const t of tokens) {
      const id = lookupId(t);
      if (id === null) {
        oov.push(true);
        ids.push(-1); // placeholder
      } else {
        oov.push(false);
        ids.push(id);
      }
    }
    // Token embeddings: rows of E indexed by id (skip OOV).
    const tokenEmb = tokens.map((_, i) => (ids[i] === -1 ? [0, 0, 0, 0] : E[ids[i]]));
    // Positional embeddings for positions 0..tokens.length-1.
    const peRows = PE.slice(0, tokens.length);
    // Final Z = token embeddings + PE.
    const Z = tokenEmb.map((row, i) => row.map((v, j) => v + peRows[i][j]));
    return { tokens, ids, oov, tokenEmb, peRows, Z };
  }, [committed, E, PE]);

  const [zRows, zCols] = shape(steps.Z);

  return (
    <div>
      <ModuleHeader
        number={10}
        title="Text Representation"
        subtitle="From a raw sentence to a sequence of contextualized vectors. Tokenize, look up embeddings, add sinusoidal positional encoding."
      >
        <MathBlock block>
          {`Z = E[\\text{token\\_ids}] + \\text{PE}, \\quad E \\in \\mathbb{R}^{V \\times d},\\ \\ \\text{PE}(p, 2i) = \\sin(p / 10000^{2i/d})`}
        </MathBlock>
      </ModuleHeader>

      <div className="grid gap-4 lg:grid-cols-[300px_1fr]">
        {/* Controls */}
        <Card className="h-fit">
          <CardHeader><CardTitle className="text-base">Input sentence</CardTitle></CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="space-y-1">
              <Label className="text-xs" htmlFor="sentence-input">Sentence</Label>
              <Input
                id="sentence-input"
                value={sentence}
                onChange={(e) => setSentence(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") setCommitted(sentence); }}
                placeholder="A dog is running"
              />
            </div>
            <Button size="sm" className="w-full" onClick={() => setCommitted(sentence)}>
              Tokenize
            </Button>
            <Button size="sm" variant="ghost" className="w-full" onClick={() => setSeed((s) => s + 1)}>
              <RefreshCw className="h-3.5 w-3.5" /> Regenerate embeddings
            </Button>
            <div className="rounded border bg-muted/30 p-2 text-xs space-y-1">
              <div className="flex justify-between"><span>Vocabulary size V</span><code>{V}</code></div>
              <div className="flex justify-between"><span>Embedding dim d</span><code>{D}</code></div>
              <div className="flex justify-between"><span>Tokens produced</span><code>{steps.tokens.length}</code></div>
              <div className="flex justify-between"><span>E shape</span><DimBadge dims={`${V} x ${D}`} variant="weight" /></div>
              <div className="flex justify-between"><span>Output Z shape</span><DimBadge dims={`${zRows} x ${zCols}`} variant="output" /></div>
            </div>
            <p className="text-[10px] text-muted-foreground">
              Real tokenizers (BPE, WordPiece, SentencePiece) split unknown words into subword units. This toy
              tokenizer just splits on whitespace and looks up the dictionary.
            </p>
          </CardContent>
        </Card>

        <div className="space-y-4">
          {/* Pipeline visual */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Pipeline: text → tokens → ids → embeddings → + PE</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="trace">
                <TabsList className="grid w-full grid-cols-5">
                  <TabsTrigger value="trace">Trace</TabsTrigger>
                  <TabsTrigger value="tokens">Tokens</TabsTrigger>
                  <TabsTrigger value="ids">IDs</TabsTrigger>
                  <TabsTrigger value="embeddings">Embeddings</TabsTrigger>
                  <TabsTrigger value="final">Z = E + PE</TabsTrigger>
                </TabsList>

                {/* Trace */}
                <TabsContent value="trace" className="space-y-3 pt-3">
                  <div className="overflow-x-auto">
                    <div className="flex items-center gap-3 min-w-max">
                      {/* Text */}
                      <PipelineStage label="Text" variant="input">
                        <code className="text-xs">"{committed}"</code>
                      </PipelineStage>
                      <Arrow />
                      {/* Tokens */}
                      <PipelineStage label="Tokens" variant="intermediate">
                        <TokenChips items={steps.tokens.length ? steps.tokens : ["—"]} oov={steps.oov} />
                      </PipelineStage>
                      <Arrow />
                      {/* IDs */}
                      <PipelineStage label="Token IDs" variant="intermediate">
                        <IDChips ids={steps.ids} oov={steps.oov} />
                      </PipelineStage>
                      <Arrow />
                      {/* Token embeddings */}
                      <PipelineStage label="Token Emb" variant="weight">
                        <span className="text-[10px] text-muted-foreground">lookup E[id]</span>
                      </PipelineStage>
                      <Arrow />
                      {/* Final */}
                      <PipelineStage label="Z = E[id] + PE" variant="output">
                        <DimBadge dims={`${zRows} x ${zCols}`} variant="output" />
                      </PipelineStage>
                    </div>
                  </div>
                  {steps.oov.some(Boolean) && (
                    <p className="mt-2 text-xs text-amber-600 dark:text-amber-400">
                      Some tokens are out-of-vocabulary (highlighted). Real tokenizers split these into subwords instead of dropping them.
                    </p>
                  )}
                </TabsContent>

                {/* Tokens */}
                <TabsContent value="tokens" className="pt-3 space-y-2">
                  <p className="text-xs text-muted-foreground">Whitespace split (no BPE for this demo):</p>
                  <TokenChips items={steps.tokens.length ? steps.tokens : ["—"]} oov={steps.oov} large />
                  <p className="text-xs text-muted-foreground">
                    Each token is a string. In a real tokenizer, "running" might be split into "run" + "##ning".
                  </p>
                </TabsContent>

                {/* IDs */}
                <TabsContent value="ids" className="pt-3 space-y-2">
                  <p className="text-xs text-muted-foreground">Dictionary lookup against the V={V} vocabulary:</p>
                  <IDChips ids={steps.ids} oov={steps.oov} large />
                  <div className="mt-2 rounded border bg-muted/20 p-2">
                    <p className="mb-1 text-[10px] uppercase text-muted-foreground">Vocabulary (id → word)</p>
                    <div className="flex flex-wrap gap-1 text-[10px] font-mono">
                      {VOCAB.map((w, i) => (
                        <span key={w} className="rounded border bg-background px-1.5 py-0.5">
                          {i}: {w}
                        </span>
                      ))}
                    </div>
                  </div>
                </TabsContent>

                {/* Embeddings */}
                <TabsContent value="embeddings" className="pt-3 space-y-3">
                  <p className="text-xs text-muted-foreground">
                    The embedding matrix E ∈ ℝ^({V}×{D}) holds one row per word. The token embeddings are rows of E
                    selected by token IDs:
                  </p>
                  <div className="rounded border bg-muted/20 p-3">
                    <p className="mb-1 text-[10px] uppercase text-muted-foreground">Full embedding matrix E (V x d)</p>
                    <MatrixView
                      matrix={E}
                      digits={2}
                      heatmap
                      diverging
                      cellSize="xs"
                      rowLabels={VOCAB.map((w) => `${w}`)}
                      colLabels={["d0", "d1", "d2", "d3"]}
                    />
                  </div>
                  <div className="rounded border bg-muted/20 p-3">
                    <p className="mb-1 text-[10px] uppercase text-muted-foreground">Selected token embeddings (n x d)</p>
                    {steps.tokens.length === 0 ? (
                      <p className="text-xs text-muted-foreground">No tokens.</p>
                    ) : (
                      <MatrixView
                        matrix={steps.tokenEmb}
                        digits={2}
                        heatmap
                        diverging
                        cellSize="sm"
                        rowLabels={steps.tokens}
                        colLabels={["d0", "d1", "d2", "d3"]}
                      />
                    )}
                  </div>
                </TabsContent>

                {/* Final */}
                <TabsContent value="final" className="pt-3 space-y-3">
                  <p className="text-xs text-muted-foreground">
                    Add sinusoidal positional encodings to inject order information:
                  </p>
                  <div className="grid gap-3 md:grid-cols-3">
                    <div>
                      <p className="mb-1 text-[10px] uppercase text-muted-foreground">Token emb</p>
                      {steps.tokens.length ? (
                        <MatrixView matrix={steps.tokenEmb} digits={2} heatmap diverging cellSize="xs" />
                      ) : <p className="text-xs text-muted-foreground">—</p>}
                    </div>
                    <div>
                      <p className="mb-1 text-[10px] uppercase text-muted-foreground">+ PE</p>
                      <MatrixView matrix={steps.peRows} digits={2} heatmap diverging cellSize="xs" />
                    </div>
                    <div>
                      <p className="mb-1 text-[10px] uppercase text-muted-foreground">= Z (final)</p>
                      <MatrixView matrix={steps.Z} digits={2} heatmap diverging cellSize="xs" />
                    </div>
                  </div>
                  <div className="rounded border bg-muted/20 p-3">
                    <p className="mb-2 text-[10px] uppercase text-muted-foreground">Sinusoidal positional encoding (full PE matrix, positions 0..15)</p>
                    <Heatmap
                      matrix={PE}
                      diverging
                      cellSize={18}
                      rowLabels={Array.from({ length: PE.length }, (_, i) => `p${i}`)}
                      colLabels={["d0", "d1", "d2", "d3"]}
                      format={(v) => v.toFixed(1)}
                    />
                    <p className="mt-2 text-[10px] text-muted-foreground">
                      Each row is the position vector. Adjacent rows are smooth in position; even/odd columns use
                      sin/cos at different frequencies.
                    </p>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <ExpandableSection title="Why do we need positional encoding?" variant="why" defaultOpen>
              <p>
                Self-attention is <em>permutation-equivariant</em>: swapping two tokens in the input produces the
                same set of attended features, just reordered. The model has no notion of order unless we inject it.
              </p>
              <p className="mt-2">
                Positional encoding adds a position-dependent bias to each token embedding, so that two identical
                words at different positions end up with different input vectors.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Why sinusoidal? Why not just integer position?" variant="math" defaultOpen>
              <p>
                The sinusoidal form <MathBlock>{`\\sin(p / 10000^{2i/d})`}</MathBlock> gives a unique, bounded vector
                for every position <MathBlock>{`p`}</MathBlock>. Each dimension corresponds to a different frequency,
                so the encoding is a Fourier-like decomposition of position.
              </p>
              <p className="mt-2">
                Three properties matter:
              </p>
              <ol className="list-decimal pl-5 space-y-1">
                <li>Bounded range <MathBlock>{`[-1, 1]`}</MathBlock> — compatible with normalized embeddings.</li>
                <li>Different positions produce different vectors, even for very long sequences.</li>
                <li><MathBlock>{`\\text{PE}(p+k)`}</MathBlock> is a linear function of <MathBlock>{`\\text{PE}(p)`}</MathBlock> — the model can learn relative shifts.</li>
              </ol>
              <p className="mt-2">
                Integer position would be unbounded and would dominate the embedding magnitudes.
              </p>
            </ExpandableSection>

            <ExpandableSection title="Learned vs sinusoidal positional embeddings" variant="how">
              <p>
                The original Transformer ("Attention Is All You Need", 2017) used sinusoidal PE. BERT, GPT-2, and most
                modern transformers instead use <em>learned</em> positional embeddings — a lookup table of shape
                <MathBlock>{`(\\text{max\\_len} \\times d_{\\text{model}})`}</MathBlock> trained alongside the model.
              </p>
              <p className="mt-2">
                Sinusoidal PE generalizes to longer sequences than seen at training time; learned embeddings do not,
                but perform slightly better in-distribution. Recent models (RoPE, ALiBi) encode position implicitly
                through attention score modulation.
              </p>
            </ExpandableSection>

            <ExpandableSection title="From tokens to vectors: the whole pipeline" variant="intuition">
              <p>
                Text is a string — discrete symbols. To do math (matrix multiplies, attention, softmax) we need
                floating-point vectors. The pipeline:
              </p>
              <ol className="list-decimal pl-5 space-y-1">
                <li><strong>Tokenize</strong> — split the string into discrete tokens (words or subwords).</li>
                <li><strong>Map to IDs</strong> — look up each token in a vocabulary to get an integer.</li>
                <li><strong>Embed</strong> — index rows of an embedding matrix E to get a dense vector per token.</li>
                <li><strong>Add position</strong> — combine with PE so the model knows which token came first.</li>
              </ol>
              <p className="mt-2">
                The result <MathBlock>{`Z \\in \\mathbb{R}^{n \\times d}`}</MathBlock> is what a transformer block actually
                operates on. Everything downstream (attention, MLP, layer norm) is matrix algebra over <MathBlock>{`Z`}</MathBlock>.
              </p>
            </ExpandableSection>
          </div>

          <VivaPanel
            questions={[
              {
                level: "Easy",
                question: "Why do we need to convert tokens to embeddings before feeding them to a transformer?",
                hint: "Neural networks operate on what kind of numbers?",
                answer:
                  "Neural networks are composed of matrix multiplications and nonlinearities, which require continuous-valued vectors. Tokens are discrete integers (or strings) — they cannot be multiplied by a weight matrix directly. Embeddings map each token id to a learnable dense vector that the network can process.",
                explanation:
                  "One-hot multiplying by an embedding matrix is mathematically identical to row lookup: E[id] selects the id-th row. Treating it as a matrix multiply lets autograd compute gradients with respect to E.",
              },
              {
                level: "Medium",
                question: "Why is sinusoidal positional encoding a sensible default compared to just adding the integer position to the embedding?",
                hint: "Think about magnitude and uniqueness across long sequences.",
                answer:
                  "Adding raw integer positions would (i) produce unbounded values that dominate the embeddings, and (ii) be the same scalar across all d dimensions, providing no spatial structure. Sinusoidal PE is bounded in [-1, 1], has a unique vector per position, and varies smoothly so nearby positions have similar encodings.",
                explanation:
                  "The use of multiple frequencies means each dimension captures position at a different scale. The model can attend over relative positions because PE(p+k) is a linear function of PE(p), so attention can learn a 'shift by k' operation.",
              },
              {
                level: "Difficult",
                question:
                  "Self-attention is permutation-equivariant. After adding positional encodings to the input, is the model still permutation-equivariant? Why or why not?",
                hint: "Consider what changes when you permute the input.",
                answer:
                  "No — the model is no longer permutation-equivariant once positional encodings are added. Permuting the input also permutes the positional encodings, so two tokens that were identical but at different positions now carry different PE and are not interchangeable. PE breaks the symmetry that bare self-attention would otherwise preserve.",
                explanation:
                  "Strictly speaking, self-attention alone remains permutation-equivariant: permuting tokens permutes the output in the same way. But PE couples each token to its position. Once the input is Z = E[id] + PE, swapping two rows of Z changes which positions are present, and the output changes accordingly. This is the entire point of PE: to make order matter.",
              },
            ]}
          />
        </div>
      </div>
    </div>
  );
}

function PipelineStage({ label, children, variant }: { label: string; children: React.ReactNode; variant: "input" | "intermediate" | "weight" | "output" }) {
  const borders: Record<typeof variant, string> = {
    input: "border-emerald-500/40",
    intermediate: "border-sky-500/40",
    weight: "border-amber-500/40",
    output: "border-violet-500/40",
  };
  return (
    <div className={`flex min-w-[140px] flex-col gap-1 rounded-lg border ${borders[variant]} bg-card/60 p-3`}>
      <span className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</span>
      <div className="min-h-[40px] flex items-center">{children}</div>
    </div>
  );
}

function Arrow() {
  return <div className="text-xl text-muted-foreground">→</div>;
}

function TokenChips({ items, oov, large }: { items: string[]; oov?: boolean[]; large?: boolean }) {
  return (
    <div className="flex flex-wrap gap-1">
      {items.map((t, i) => (
        <span
          key={i}
          className={`rounded border font-mono ${large ? "px-2 py-1 text-sm" : "px-1.5 py-0.5 text-[10px]"} ${
            oov && oov[i]
              ? "border-amber-500/50 bg-amber-500/10 text-amber-700 dark:text-amber-300"
              : "border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300"
          }`}
        >
          {t}
        </span>
      ))}
    </div>
  );
}

function IDChips({ ids, oov, large }: { ids: number[]; oov?: boolean[]; large?: boolean }) {
  return (
    <div className="flex flex-wrap gap-1">
      {ids.length === 0 ? (
        <span className="text-xs text-muted-foreground">—</span>
      ) : (
        ids.map((id, i) => (
          <span
            key={i}
            className={`rounded border font-mono ${large ? "px-2 py-1 text-sm" : "px-1.5 py-0.5 text-[10px]"} ${
              id === -1
                ? "border-amber-500/50 bg-amber-500/10 text-amber-700 dark:text-amber-300"
                : "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
            }`}
          >
            {id === -1 ? "OOV" : id}
          </span>
        ))
      )}
    </div>
  );
}
