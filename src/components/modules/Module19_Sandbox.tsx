"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Calculator, Play, Dice5, RotateCcw, TriangleAlert } from "lucide-react";
import { MatrixView } from "@/components/cvlm/MatrixView";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";
import {
  matmul,
  transpose,
  relu,
  softmaxVec,
  softmaxRows,
  cosineSimilarity,
  norm,
  shape,
  flatten,
  type Matrix,
  type Vector,
} from "@/lib/math/matrix";

type OpId =
  | "matmul"
  | "transpose"
  | "add"
  | "relu"
  | "softmaxRows"
  | "cosineSim"
  | "normMatrix"
  | "softmaxVec"
  | "reluVec"
  | "normVec";

interface OpDef {
  id: OpId;
  label: string;
  arity: "unary" | "binary";
  kind: "matrix" | "vector" | "scalar";
  formula: string;
  formulaLatex: string;
}

const OPS: OpDef[] = [
  { id: "matmul", label: "A × B (matmul)", arity: "binary", kind: "matrix", formula: "C[i,j] = Σ_k A[i,k] · B[k,j]", formulaLatex: "C = AB,\\ C_{ij}=\\sum_k A_{ik} B_{kj}" },
  { id: "transpose", label: "Aᵀ (transpose)", arity: "unary", kind: "matrix", formula: "Aᵀ[j,i] = A[i,j]", formulaLatex: "A^\\top_{ij} = A_{ji}" },
  { id: "add", label: "A + B (element-wise)", arity: "binary", kind: "matrix", formula: "C[i,j] = A[i,j] + B[i,j]", formulaLatex: "C = A + B" },
  { id: "relu", label: "ReLU(A)", arity: "unary", kind: "matrix", formula: "C[i,j] = max(0, A[i,j])", formulaLatex: "\\text{ReLU}(x)=\\max(0,x)" },
  { id: "softmaxRows", label: "softmax(A, rows)", arity: "unary", kind: "matrix", formula: "row-wise softmax", formulaLatex: "\\text{softmax}(x)_i=\\frac{e^{x_i}}{\\sum_j e^{x_j}}" },
  { id: "cosineSim", label: "cosine(flatten A, flatten B)", arity: "binary", kind: "scalar", formula: "(a·b) / (||a|| · ||b||)", formulaLatex: "\\cos(a,b)=\\frac{a\\cdot b}{\\|a\\|\\,\\|b\\|}" },
  { id: "normMatrix", label: "||flatten A||₂ (L2 norm)", arity: "unary", kind: "scalar", formula: "√(Σ aᵢ²)", formulaLatex: "\\|a\\|_2 = \\sqrt{\\sum_i a_i^2}" },
  { id: "softmaxVec", label: "softmax(first row of A)", arity: "unary", kind: "vector", formula: "row 0 of A → softmax", formulaLatex: "\\text{softmax}(x)" },
  { id: "reluVec", label: "ReLU(first row of A)", arity: "unary", kind: "vector", formula: "max(0, ·)", formulaLatex: "\\text{ReLU}(x)" },
  { id: "normVec", label: "||first row of A||₂", arity: "unary", kind: "scalar", formula: "L2 norm of row 0", formulaLatex: "\\|x\\|_2" },
];

const EXAMPLES: { name: string; a: string; b: string; op: OpId }[] = [
  {
    name: "2x3 × 3x2 matmul",
    a: "1,2,3\n4,5,6",
    b: "7,8\n9,10\n11,12",
    op: "matmul",
  },
  {
    name: "Identity softmax (1.0,2.0,3.0)",
    a: "1,2,3",
    b: "",
    op: "softmaxVec",
  },
  {
    name: "ReLU with negatives",
    a: "-1,2,-3\n4,-5,6",
    b: "",
    op: "relu",
  },
  {
    name: "Cosine sim of two rows",
    a: "1,2,3\n4,5,6",
    b: "1,0,1\n2,3,4",
    op: "cosineSim",
  },
  {
    name: "Transpose 2x3 → 3x2",
    a: "1,2,3\n4,5,6",
    b: "",
    op: "transpose",
  },
];

function parseMatrix(text: string): Matrix | null {
  const lines = text.trim().split(/\n+/).map((l) => l.trim()).filter((l) => l.length > 0);
  if (lines.length === 0) return null;
  const rows: number[][] = [];
  for (const line of lines) {
    const parts = line.split(/[,\s]+/).map((p) => p.trim()).filter((p) => p.length > 0);
    if (parts.length === 0) return null;
    const row = parts.map((p) => parseFloat(p));
    if (row.some((v) => Number.isNaN(v))) return null;
    rows.push(row);
  }
  const cols = rows[0].length;
  if (rows.some((r) => r.length !== cols)) return null;
  return rows;
}

interface ComputeResult {
  ok: boolean;
  error?: string;
  matrix?: Matrix;
  vector?: Vector;
  scalar?: number;
}

function compute(op: OpDef, A: Matrix, B: Matrix | null): ComputeResult {
  try {
    switch (op.id) {
      case "matmul": {
        if (!B) return { ok: false, error: "Operation 'matmul' requires both A and B." };
        const [, k1] = shape(A);
        const [k2] = shape(B);
        if (k1 !== k2) {
          return {
            ok: false,
            error: `Dimension mismatch: A has ${k1} columns but B has ${k2} rows. For matmul A·B, A's column count must equal B's row count.`,
          };
        }
        return { ok: true, matrix: matmul(A, B) };
      }
      case "transpose":
        return { ok: true, matrix: transpose(A) };
      case "add": {
        if (!B) return { ok: false, error: "Operation 'add' requires both A and B." };
        const [ra, ca] = shape(A);
        const [rb, cb] = shape(B);
        if (ra !== rb || ca !== cb) {
          return {
            ok: false,
            error: `Dimension mismatch: A is ${ra}×${ca} but B is ${rb}×${cb}. Element-wise add requires identical shapes.`,
          };
        }
        return { ok: true, matrix: A.map((row, i) => row.map((v, j) => v + B[i][j])) };
      }
      case "relu":
        return { ok: true, matrix: relu(A) };
      case "softmaxRows":
        return { ok: true, matrix: softmaxRows(A) };
      case "cosineSim": {
        if (!B) return { ok: false, error: "Operation 'cosine sim' requires both A and B." };
        const a = flatten(A);
        const b = flatten(B);
        if (a.length !== b.length) {
          return {
            ok: false,
            error: `Cosine similarity requires vectors of equal length: flatten(A) has ${a.length} elements, flatten(B) has ${b.length}.`,
          };
        }
        return { ok: true, scalar: cosineSimilarity(a, b) };
      }
      case "normMatrix":
        return { ok: true, scalar: norm(flatten(A)) };
      case "softmaxVec":
        return { ok: true, vector: softmaxVec(A[0]) };
      case "reluVec":
        return { ok: true, vector: A[0].map((v) => Math.max(0, v)) };
      case "normVec":
        return { ok: true, scalar: norm(A[0]) };
      default:
        return { ok: false, error: "Unknown operation." };
    }
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
}

export function Module19_Sandbox() {
  const [textA, setTextA] = useState("1,2,3\n4,5,6");
  const [textB, setTextB] = useState("7,8\n9,10\n11,12");
  const [opId, setOpId] = useState<OpId>("matmul");
  const [trigger, setTrigger] = useState(0);

  const A = useMemo(() => parseMatrix(textA), [textA]);
  const B = useMemo(() => parseMatrix(textB), [textB]);

  const op = OPS.find((o) => o.id === opId)!;

  // Compute on demand (triggered by button or example load)
  const result = useMemo<ComputeResult>(() => {
    if (trigger === 0) return { ok: true };
    if (!A) return { ok: false, error: "Matrix A is empty or invalid. Use commas between entries and newlines between rows, e.g. 1,2,3\\n4,5,6" };
    if (op.arity === "binary" && !B) return { ok: false, error: "Matrix B is empty or invalid. This operation requires two matrices." };
    return compute(op, A, B);
  }, [trigger, op, A, B]);

  const loadExample = (idx: number) => {
    const ex = EXAMPLES[idx];
    setTextA(ex.a);
    setTextB(ex.b);
    setOpId(ex.op);
    setTrigger((t) => t + 1);
  };

  const shapeStr = (M: Matrix | null): string => (M ? `${shape(M)[0]} × ${shape(M)[1]}` : "—");

  return (
    <div>
      <ModuleHeader
        number={19}
        title="Mathematical Sandbox"
        subtitle="A pocket calculator for the matrix operations you have seen throughout the course. Enter matrices, pick an operation, see exact dimensions and output."
      >
        <MathBlock block>
          {`\\text{operations}: \\ AB,\\ A^\\top,\\ A+B,\\ \\text{ReLU}(A),\\ \\text{softmax}(A),\\ \\cos(a,b),\\ \\|a\\|_2`}
        </MathBlock>
      </ModuleHeader>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center justify-between">
              <span>Matrix A</span>
              <DimBadge dims={shapeStr(A)} variant="input" />
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Textarea
              value={textA}
              onChange={(e) => setTextA(e.target.value)}
              rows={4}
              className="font-mono text-xs"
              placeholder={"1,2,3\n4,5,6"}
            />
            <p className="text-[10px] text-muted-foreground">
              Comma- or space-separated values; one row per line.
            </p>
          </CardContent>
        </Card>

        <Card className={op.arity === "unary" ? "opacity-50" : ""}>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center justify-between">
              <span>Matrix B {op.arity === "unary" && <Badge variant="secondary" className="text-[10px]">not used</Badge>}</span>
              <DimBadge dims={shapeStr(B)} variant="input" />
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Textarea
              value={textB}
              onChange={(e) => setTextB(e.target.value)}
              rows={4}
              className="font-mono text-xs"
              placeholder={"7,8\n9,10\n11,12"}
              disabled={op.arity === "unary"}
            />
            <p className="text-[10px] text-muted-foreground">
              Required only for binary operations (matmul, add, cosine sim).
            </p>
          </CardContent>
        </Card>
      </div>

      <Card className="mt-4">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Operation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-2 items-end">
            <div className="space-y-1">
              <Label className="text-xs">Choose operation</Label>
              <Select value={opId} onValueChange={(v) => { setOpId(v as OpId); setTrigger((t) => t + 1); }}>
                <SelectTrigger className="w-[280px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {OPS.map((o) => (
                    <SelectItem key={o.id} value={o.id}>
                      {o.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button size="sm" onClick={() => setTrigger((t) => t + 1)}>
              <Play className="h-4 w-4" /> Compute
            </Button>
            <Button size="sm" variant="outline" onClick={() => { setTextA("1,2,3\n4,5,6"); setTextB("7,8\n9,10\n11,12"); setOpId("matmul"); setTrigger((t) => t + 1); }}>
              <RotateCcw className="h-4 w-4" /> Reset
            </Button>
            <Button size="sm" variant="outline" onClick={() => loadExample(Math.floor(Math.random() * EXAMPLES.length))}>
              <Dice5 className="h-4 w-4" /> Random example
            </Button>
          </div>

          <div className="rounded-md border bg-muted/30 p-2 text-xs">
            <span className="text-muted-foreground">Formula: </span>
            <MathBlock>{op.formulaLatex}</MathBlock>
            <span className="ml-2 text-muted-foreground">({op.formula})</span>
          </div>

          <div>
            <Label className="text-xs">Load example</Label>
            <div className="mt-1 flex flex-wrap gap-2">
              {EXAMPLES.map((ex, i) => (
                <Button key={i} size="sm" variant="outline" onClick={() => loadExample(i)}>
                  <Calculator className="h-3.5 w-3.5" /> {ex.name}
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="mt-4">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Calculator className="h-4 w-4" /> Result
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {!result.ok && result.error ? (
            <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm">
              <TriangleAlert className="h-4 w-4 mt-0.5 text-destructive shrink-0" />
              <div>
                <p className="font-medium text-destructive">Operation failed</p>
                <p className="text-xs mt-1 text-muted-foreground">{result.error}</p>
              </div>
            </div>
          ) : trigger === 0 ? (
            <p className="text-sm text-muted-foreground py-4 text-center">
              Pick an operation and press <strong>Compute</strong> to see the output.
            </p>
          ) : (
            <ResultView op={op} result={result} />
          )}
        </CardContent>
      </Card>

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <ExpandableSection title="Why does matmul require the inner dimensions to match?" variant="why" defaultOpen>
          <p>
            For <MathBlock>{`C = AB`}</MathBlock> with <MathBlock>{`A \\in \\mathbb{R}^{m \\times k}`}</MathBlock> and{" "}
            <MathBlock>{`B \\in \\mathbb{R}^{k \\times n}`}</MathBlock>, every entry of <MathBlock>{`C`}</MathBlock> is a
            dot product of a row of <MathBlock>{`A`}</MathBlock> (length <MathBlock>{`k`}</MathBlock>) with a column
            of <MathBlock>{`B`}</MathBlock> (length <MathBlock>{`k`}</MathBlock>). If those lengths disagree, the dot
            product is undefined.
          </p>
          <p className="mt-2">
            This is the <em>inner dimension</em> rule: <code>(m, k) · (k, n) → (m, n)</code>.
          </p>
        </ExpandableSection>

        <ExpandableSection title="Why is softmax row-wise for attention weights?" variant="how">
          <p>
            Attention weight <MathBlock>{`A_{ij}`}</MathBlock> says how much token <MathBlock>{`i`}</MathBlock>{" "}
            attends to token <MathBlock>{`j`}</MathBlock>. For each query token <MathBlock>{`i`}</MathBlock>, the
            weights across all <MathBlock>{`j`}</MathBlock> must sum to 1 (a probability distribution). Hence we
            softmax <em>along each row</em>.
          </p>
          <p className="mt-2">
            Softmaxing along columns would normalize the wrong axis and break the interpretation.
          </p>
        </ExpandableSection>
      </div>

      <div className="mt-6">
        <VivaPanel
          questions={[
            {
              level: "Easy",
              question: "Given A ∈ ℝ^(2×3) and B ∈ ℝ^(3×4), what is the shape of A·B? Can you compute B·A?",
              hint: "Apply the inner-dimension rule both ways.",
              answer:
                "A·B is (2×4). B·A is undefined because B has 4 columns but A has only 2 rows; the inner dimensions 4 and 2 do not match.",
              explanation:
                "Matrix multiplication is not commutative in general — and not even always defined both ways. Always check the inner dimensions: (m,k)·(k,n)→(m,n).",
            },
            {
              level: "Medium",
              question:
                "Cosine similarity returns a value in [-1, 1]. What does each extreme mean, and what is the geometric interpretation?",
              hint: "Think of vectors as arrows from the origin.",
              answer:
                "cos(a,b) = (a·b)/(||a|| ||b||). +1 means the vectors point in the same direction (parallel), 0 means orthogonal (no shared component), -1 means opposite directions.",
              explanation:
                "Cosine ignores magnitude and measures only the angle. That is why it is the standard similarity in CLIP-style contrastive learning: scale-invariant comparison of embeddings.",
            },
          ]}
        />
      </div>
    </div>
  );
}

function ResultView({ op, result }: { op: OpDef; result: ComputeResult }) {
  if (!result.ok) return null;
  if (result.matrix) {
    const [r, c] = shape(result.matrix);
    return (
      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="text-muted-foreground">Output shape:</span>
          <DimBadge dims={`${r} × ${c}`} variant="output" />
          <Badge variant="outline" className="text-[10px]">{op.kind}</Badge>
        </div>
        <MatrixView matrix={result.matrix} heatmap digits={3} cellSize="sm" />
      </div>
    );
  }
  if (result.vector) {
    return (
      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="text-muted-foreground">Output shape:</span>
          <DimBadge dims={`${result.vector.length}`} variant="output" />
          <Badge variant="outline" className="text-[10px]">vector</Badge>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {result.vector.map((v, i) => (
            <div key={i} className="rounded-md border bg-card px-2 py-1 font-mono text-xs">
              <span className="text-muted-foreground">[{i}]</span> {v.toFixed(4)}
            </div>
          ))}
        </div>
      </div>
    );
  }
  if (result.scalar !== undefined) {
    return (
      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="text-muted-foreground">Output type:</span>
          <DimBadge dims="scalar" variant="output" />
        </div>
        <div className="rounded-md border-2 border-primary/30 bg-primary/5 p-4 text-center">
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Value</p>
          <p className="mt-1 text-2xl font-mono">{result.scalar.toFixed(6)}</p>
        </div>
      </div>
    );
  }
  return null;
}
