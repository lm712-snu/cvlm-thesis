"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Plus, Trash2, RotateCcw, TriangleAlert, CheckCircle2 } from "lucide-react";
import { DimBadge } from "@/components/cvlm/DimBadge";
import { MathBlock } from "@/components/cvlm/MathBlock";
import { ExpandableSection } from "@/components/cvlm/ExpandableSection";
import { ModuleHeader } from "@/components/cvlm/ModuleHeader";
import { VivaPanel } from "@/components/cvlm/VivaCard";

type OpType =
  | "Linear"
  | "Conv2d"
  | "MultiheadAttention"
  | "Softmax"
  | "LayerNorm"
  | "ReLU"
  | "Flatten"
  | "Reshape";

interface ParamSpec {
  name: string;
  label: string;
  default: number;
}

const OP_PARAMS: Record<OpType, ParamSpec[]> = {
  Linear: [
    { name: "in_features", label: "in_features", default: 64 },
    { name: "out_features", label: "out_features", default: 128 },
  ],
  Conv2d: [
    { name: "in_channels", label: "in_channels", default: 3 },
    { name: "out_channels", label: "out_channels", default: 16 },
    { name: "kernel", label: "kernel size", default: 3 },
    { name: "stride", label: "stride", default: 1 },
    { name: "padding", label: "padding", default: 1 },
  ],
  MultiheadAttention: [
    { name: "d_model", label: "d_model", default: 64 },
    { name: "n_heads", label: "n_heads", default: 8 },
  ],
  Softmax: [],
  LayerNorm: [{ name: "normalized_shape", label: "last dim", default: 64 }],
  ReLU: [],
  Flatten: [],
  Reshape: [],
};

interface OpInstance {
  id: number;
  type: OpType;
  params: Record<string, number>;
  reshapeTarget?: string; // comma-separated for Reshape
}

interface OpResult {
  outShape: number[];
  error?: string;
  formula: string;
}

function applyOp(op: OpInstance, inShape: number[]): OpResult {
  switch (op.type) {
    case "Linear": {
      const inF = op.params.in_features ?? 0;
      const outF = op.params.out_features ?? 0;
      const last = inShape[inShape.length - 1] ?? 0;
      const formula = `[*, ${inF}] → [*, ${outF}]`;
      if (inF !== last) {
        return { outShape: [], error: `Linear expects in_features=${inF} but previous output's last dim is ${last}.`, formula };
      }
      return { outShape: [...inShape.slice(0, -1), outF], formula };
    }
    case "Conv2d": {
      const C_in = op.params.in_channels ?? 0;
      const C_out = op.params.out_channels ?? 0;
      const k = op.params.kernel ?? 1;
      const s = op.params.stride ?? 1;
      const p = op.params.padding ?? 0;
      const formula = `H' = ⌊(H + 2p − k)/s⌋ + 1, with k=${k}, s=${s}, p=${p}`;
      if (inShape.length !== 4) {
        return { outShape: [], error: `Conv2d requires 4D input [B, C, H, W], got [${inShape.join(", ")}].`, formula };
      }
      const [B, C, H, W] = inShape;
      if (C !== C_in) {
        return { outShape: [], error: `Conv2d expects in_channels=${C_in} but previous output has C=${C}.`, formula };
      }
      if (k > H + 2 * p || k > W + 2 * p) {
        return { outShape: [], error: `Kernel ${k} larger than padded input (${H + 2 * p}×${W + 2 * p}).`, formula };
      }
      const H_out = Math.floor((H + 2 * p - k) / s) + 1;
      const W_out = Math.floor((W + 2 * p - k) / s) + 1;
      return { outShape: [B, C_out, H_out, W_out], formula };
    }
    case "MultiheadAttention": {
      const d_model = op.params.d_model ?? 0;
      const n_heads = op.params.n_heads ?? 1;
      const formula = `[B, N, d_model] → [B, N, d_model], d_k = d_model/n_heads`;
      if (inShape.length !== 3) {
        return { outShape: [], error: `MultiheadAttention requires 3D input [B, N, d_model], got [${inShape.join(", ")}].`, formula };
      }
      const d = inShape[2];
      if (d !== d_model) {
        return { outShape: [], error: `MultiheadAttention expects d_model=${d_model} but previous output's last dim is ${d}.`, formula };
      }
      if (d_model % n_heads !== 0) {
        return { outShape: [], error: `d_model=${d_model} must be divisible by n_heads=${n_heads}.`, formula };
      }
      return { outShape: [...inShape], formula };
    }
    case "Softmax":
      return { outShape: [...inShape], formula: `softmax over last axis: shape unchanged` };
    case "LayerNorm": {
      const ns = op.params.normalized_shape ?? 0;
      const last = inShape[inShape.length - 1] ?? 0;
      const formula = `LayerNorm over last ${ns}`;
      if (ns !== last) {
        return { outShape: [], error: `LayerNorm expects normalized_shape=${ns} but last dim is ${last}.`, formula };
      }
      return { outShape: [...inShape], formula };
    }
    case "ReLU":
      return { outShape: [...inShape], formula: `ReLU: shape unchanged` };
    case "Flatten": {
      if (inShape.length < 2) {
        return { outShape: [...inShape], formula: `already flat` };
      }
      const B = inShape[0];
      const rest = inShape.slice(1).reduce((a, b) => a * b, 1);
      return { outShape: [B, rest], formula: `[B, d1, d2, ...] → [B, d1·d2·...]` };
    }
    case "Reshape": {
      const target = (op.reshapeTarget ?? "")
        .split(",")
        .map((s) => parseInt(s.trim(), 10))
        .filter((v) => !Number.isNaN(v));
      const formula = `reshape to [${target.join(", ")}]`;
      const total = inShape.reduce((a, b) => a * b, 1);
      const targetTotal = target.reduce((a, b) => a * b, 1);
      if (target.length === 0) {
        return { outShape: [], error: `Reshape requires a target shape (e.g. "1, 16, 64").`, formula };
      }
      if (total !== targetTotal) {
        return { outShape: [], error: `Reshape: input has ${total} elements but target [${target.join(", ")}] has ${targetTotal}.`, formula };
      }
      return { outShape: target, formula };
    }
    default:
      return { outShape: [...inShape], formula: `passthrough` };
  }
}

const PRESETS: { name: string; shape: number[] }[] = [
  { name: "Image (1, 3, 224, 224)", shape: [1, 3, 224, 224] },
  { name: "Image (1, 3, 32, 32)", shape: [1, 3, 32, 32] },
  { name: "Sequence (1, 16, 64)", shape: [1, 16, 64] },
  { name: "Flatten (1, 196608)", shape: [1, 196608] },
];

let opIdCounter = 1;

export function Module20_DimensionTracker() {
  const [startShape, setStartShape] = useState<string>("1, 3, 32, 32");
  const [ops, setOps] = useState<OpInstance[]>([
    { id: opIdCounter++, type: "Conv2d", params: { in_channels: 3, out_channels: 16, kernel: 3, stride: 1, padding: 1 } },
    { id: opIdCounter++, type: "ReLU", params: {} },
    { id: opIdCounter++, type: "Flatten", params: {} },
    { id: opIdCounter++, type: "Linear", params: { in_features: 16384, out_features: 10 } },
  ]);

  const parsedStart = useMemo(() => {
    const arr = startShape
      .split(",")
      .map((s) => parseInt(s.trim(), 10))
      .filter((v) => !Number.isNaN(v));
    return arr.length > 0 ? arr : null;
  }, [startShape]);

  // Walk through ops, computing shape after each.
  const trace = useMemo(() => {
    if (!parsedStart) return [];
    let current = parsedStart;
    const results: { op: OpInstance; inShape: number[]; outShape: number[]; error?: string; formula: string }[] = [];
    for (const op of ops) {
      const inShape = current;
      const r = applyOp(op, inShape);
      results.push({ op, inShape, outShape: r.outShape, error: r.error, formula: r.formula });
      if (r.error) {
        // Stop propagating after an error; downstream ops cannot be computed.
        break;
      }
      current = r.outShape;
    }
    return results;
  }, [parsedStart, ops]);

  const addOp = (type: OpType) => {
    const specs = OP_PARAMS[type];
    const params: Record<string, number> = {};
    for (const s of specs) params[s.name] = s.default;
    setOps([...ops, { id: opIdCounter++, type, params, reshapeTarget: type === "Reshape" ? "1, 4, 16" : undefined }]);
  };

  const removeOp = (id: number) => setOps(ops.filter((o) => o.id !== id));

  const updateParam = (id: number, name: string, value: number) =>
    setOps(ops.map((o) => (o.id === id ? { ...o, params: { ...o.params, [name]: value } } : o)));

  const updateReshape = (id: number, value: string) =>
    setOps(ops.map((o) => (o.id === id ? { ...o, reshapeTarget: value } : o)));

  const changeType = (id: number, type: OpType) => {
    const specs = OP_PARAMS[type];
    const params: Record<string, number> = {};
    for (const s of specs) params[s.name] = s.default;
    setOps(ops.map((o) => (o.id === id ? { ...o, type, params, reshapeTarget: type === "Reshape" ? "1, 4, 16" : undefined } : o)));
  };

  const reset = () => {
    setOps([]);
    setStartShape("1, 3, 32, 32");
  };

  const finalShape = trace.length > 0 && !trace[trace.length - 1].error ? trace[trace.length - 1].outShape : null;
  const hasError = trace.some((t) => t.error);

  return (
    <div>
      <ModuleHeader
        number={20}
        title="Dimension Tracker"
        subtitle="Stack operations on a starting tensor and watch the dimensions propagate. The tool computes output shapes automatically and flags any mismatch in red."
      >
        <MathBlock block>
          {`\\text{Conv2d}:\\ H' = \\left\\lfloor \\frac{H + 2p - k}{s} \\right\\rfloor + 1,\\quad W' = \\left\\lfloor \\frac{W + 2p - k}{s} \\right\\rfloor + 1`}
        </MathBlock>
      </ModuleHeader>

      <Card className="mb-4">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Starting tensor shape</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-2 items-end">
            <div className="space-y-1">
              <Label className="text-xs">Shape (comma-separated)</Label>
              <Input
                value={startShape}
                onChange={(e) => setStartShape(e.target.value)}
                className="font-mono w-[180px]"
                placeholder="1, 3, 32, 32"
              />
            </div>
            <div className="flex flex-wrap gap-2">
              {PRESETS.map((p) => (
                <Button key={p.name} size="sm" variant="outline" onClick={() => setStartShape(p.shape.join(", "))}>
                  {p.name}
                </Button>
              ))}
            </div>
          </div>
          {parsedStart && (
            <div className="flex items-center gap-2 text-xs">
              <span className="text-muted-foreground">Parsed:</span>
              <DimBadge dims={`[${parsedStart.join(", ")}]`} variant="input" />
              <span className="text-muted-foreground">total elements:</span>
              <code className="font-mono">{parsedStart.reduce((a, b) => a * b, 1)}</code>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="space-y-2">
        {ops.length === 0 && (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">
              No operations yet. Click an operation below to add it to the pipeline.
            </CardContent>
          </Card>
        )}
        {trace.map(({ op, inShape, outShape, error, formula }, i) => (
          <div key={op.id}>
            <div className={`rounded-lg border-2 ${error ? "border-destructive/60 bg-destructive/5" : "border-emerald-500/40 bg-emerald-500/5"}`}>
              <div className="flex flex-wrap items-center gap-3 p-3">
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border-2 text-xs font-mono bg-card">
                  {i + 1}
                </span>
                <div className="min-w-[160px]">
                  <Select value={op.type} onValueChange={(v) => changeType(op.id, v as OpType)}>
                    <SelectTrigger className="w-[220px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {(Object.keys(OP_PARAMS) as OpType[]).map((t) => (
                        <SelectItem key={t} value={t}>
                          {t}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Op-specific parameter inputs */}
                <div className="flex flex-wrap items-center gap-2">
                  {OP_PARAMS[op.type].map((spec) => (
                    <div key={spec.name} className="flex items-center gap-1">
                      <Label className="text-[10px] text-muted-foreground">{spec.label}</Label>
                      <Input
                        type="number"
                        value={op.params[spec.name] ?? 0}
                        onChange={(e) => updateParam(op.id, spec.name, parseInt(e.target.value, 10) || 0)}
                        className="font-mono h-8 w-[80px]"
                      />
                    </div>
                  ))}
                  {op.type === "Reshape" && (
                    <div className="flex items-center gap-1">
                      <Label className="text-[10px] text-muted-foreground">target</Label>
                      <Input
                        value={op.reshapeTarget ?? ""}
                        onChange={(e) => updateReshape(op.id, e.target.value)}
                        className="font-mono h-8 w-[120px]"
                        placeholder="1, 4, 16"
                      />
                    </div>
                  )}
                </div>

                <div className="ml-auto flex items-center gap-2">
                  {error ? (
                    <Badge variant="destructive" className="text-[10px]">
                      <TriangleAlert className="h-3 w-3 mr-1" /> mismatch
                    </Badge>
                  ) : (
                    <Badge variant="secondary" className="text-[10px] bg-emerald-500/20 text-emerald-700 dark:text-emerald-300">
                      <CheckCircle2 className="h-3 w-3 mr-1" /> ok
                    </Badge>
                  )}
                  <Button size="icon" variant="ghost" onClick={() => removeOp(op.id)}>
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              {/* Shape flow row */}
              <div className="border-t bg-card/40 px-3 py-2 flex flex-wrap items-center gap-2 text-xs">
                <span className="text-muted-foreground uppercase tracking-wider text-[10px]">in</span>
                <DimBadge dims={`[${inShape.join(", ")}]`} variant="input" />
                <span className="text-muted-foreground">→</span>
                {error ? (
                  <DimBadge dims="—" variant="label" />
                ) : (
                  <DimBadge dims={`[${outShape.join(", ")}]`} variant="output" />
                )}
                <span className="text-muted-foreground ml-2 text-[10px]">formula: <code>{formula}</code></span>
              </div>

              {error && (
                <div className="border-t border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive flex items-start gap-2">
                  <TriangleAlert className="h-4 w-4 mt-0.5 shrink-0" />
                  <span>{error}</span>
                </div>
              )}
            </div>
            {i < trace.length - 1 && (
              <div className="flex justify-center py-0.5">
                <div className="h-4 w-0.5 bg-border" />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Add operation bar */}
      <Card className="mt-4">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Add operation</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2 items-center">
          {(Object.keys(OP_PARAMS) as OpType[]).map((t) => (
            <Button key={t} size="sm" variant="outline" onClick={() => addOp(t)}>
              <Plus className="h-3.5 w-3.5" /> {t}
            </Button>
          ))}
          <div className="ml-auto">
            <Button size="sm" variant="ghost" onClick={reset}>
              <RotateCcw className="h-4 w-4" /> Reset
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Summary */}
      <Card className={`mt-4 ${hasError ? "border-destructive/40" : "border-emerald-500/40"}`}>
        <CardContent className="py-4 flex flex-wrap items-center gap-3">
          <span className="text-xs uppercase tracking-wider text-muted-foreground">Final shape:</span>
          {finalShape ? (
            <>
              <DimBadge dims={`[${finalShape.join(", ")}]`} variant="output" />
              <span className="text-xs text-muted-foreground">
                total elements: <code className="font-mono">{finalShape.reduce((a, b) => a * b, 1)}</code>
              </span>
            </>
          ) : (
            <Badge variant="destructive">pipeline blocked — fix the mismatch above</Badge>
          )}
        </CardContent>
      </Card>

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <ExpandableSection title="Why the Conv2d output formula?" variant="math" defaultOpen>
          <p>
            A kernel of size <MathBlock>{`k \\times k`}</MathBlock> slides over the input with stride{" "}
            <MathBlock>{`s`}</MathBlock> and padding <MathBlock>{`p`}</MathBlock>. After padding the
            input is <MathBlock>{`H + 2p`}</MathBlock> tall. Each step moves the window by{" "}
            <MathBlock>{`s`}</MathBlock> pixels, so the number of valid positions is{" "}
            <MathBlock>{`\\lfloor (H + 2p - k) / s \\rfloor + 1`}</MathBlock>.
          </p>
          <p className="mt-2">
            Common case: <MathBlock>{`k = 3, s = 1, p = 1`}</MathBlock> preserves spatial size (H&prime; = H).
            With <MathBlock>{`k = 2, s = 2, p = 0`}</MathBlock> the output is exactly half the input.
          </p>
        </ExpandableSection>

        <ExpandableSection title="Why must d_model be divisible by n_heads?" variant="why">
          <p>
            In multi-head attention, <MathBlock>{`d_{model}`}</MathBlock> is split evenly across{" "}
            <MathBlock>{`h`}</MathBlock> heads, so each head operates on{" "}
            <MathBlock>{`d_k = d_{model} / h`}</MathBlock>. If <MathBlock>{`d_{model}`}</MathBlock> is not
            divisible by <MathBlock>{`h`}</MathBlock>, the split is non-integer and the architecture is
            ill-defined.
          </p>
          <p className="mt-2">
            Standard choices: <code>d_model=512, h=8</code> (so <code>d_k=64</code>), or{" "}
            <code>d_model=768, h=12</code> (so <code>d_k=64</code>).
          </p>
        </ExpandableSection>
      </div>

      <div className="mt-6">
        <VivaPanel
          questions={[
            {
              level: "Easy",
              question: "An image of shape [1, 3, 32, 32] passes through Conv2d(in=3, out=16, k=3, s=1, p=1) then MaxPool(2,2). What is the output shape?",
              hint: "Apply the conv formula, then halve both spatial dims.",
              answer: "[1, 16, 32, 32] after the conv (k=3, p=1 preserves size), then [1, 16, 16, 16] after 2×2 max pool with stride 2.",
              explanation: "Conv2d changes channels and (with padding) keeps H, W unchanged. Pooling then halves the spatial dims. Always track the [B, C, H, W] axes separately.",
            },
            {
              level: "Difficult",
              question: "You build a tiny ViT: image [1, 3, 224, 224] → patchify (P=16) → Linear(P²C, d=384) → 12 transformer layers (d_model=384, h=6) → Linear(384, 1000). Trace every shape and identify any constraint that must hold.",
              hint: "Patchify produces N patches of size P²C.",
              answer: "Patchify: N = (224/16)² = 196 patches, each flattened to P²C = 16²·3 = 768. Linear: [1, 196, 768] → [1, 196, 384]. Transformer layers preserve shape: [1, 196, 384]. Final Linear: [1, 196, 1000] or pooled to [1, 1000]. Constraint: 768 (patch dim) must match Linear's in_features, and 384 must be divisible by n_heads=6 (384/6=64, ok).",
              explanation: "Two constraints: (1) the patch-flatten dimension P²C must equal the Linear in_features; (2) d_model must be divisible by n_heads. Both must be checked before training.",
            },
          ]}
        />
      </div>
    </div>
  );
}
