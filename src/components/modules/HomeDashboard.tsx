"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Sigma, ArrowRight, BookOpen, Calculator, Layers, GitBranch, FlaskConical } from "lucide-react";
import { useAppStore } from "@/store/app-store";
import { MODULES } from "@/data/modules";
import { MathBlock } from "@/components/cvlm/MathBlock";

const HIGHLIGHTS = [
  {
    icon: Layers,
    title: "Image → Tensor",
    eq: "X \\in \\mathbb{R}^{H \\times W \\times 3}",
    desc: "See how a pixel becomes a 3D tensor.",
    moduleId: "image-repr",
  },
  {
    icon: Sigma,
    title: "Convolution",
    eq: "Y_{ij} = \\sum_{m,n} X_{i+m,j+n} \\cdot K_{mn}",
    desc: "Sliding kernels, feature maps, stride.",
    moduleId: "convolution",
  },
  {
    icon: GitBranch,
    title: "Self-Attention",
    eq: "\\text{Attn}(X) = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V",
    desc: "Q, K, V — the heart of transformers.",
    moduleId: "self-attention",
  },
  {
    icon: Calculator,
    title: "Softmax & Next Token",
    eq: "P(t_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}}",
    desc: "From logits to probabilities to words.",
    moduleId: "next-token",
  },
  {
    icon: FlaskConical,
    title: "Cross-Modal Attention",
    eq: "\\text{softmax}\\!\\left(\\frac{Q_{\\text{text}} K_{\\text{img}}^\\top}{\\sqrt{d_k}}\\right) V_{\\text{img}}",
    desc: "How text queries attend to image patches.",
    moduleId: "cross-attention",
  },
  {
    icon: BookOpen,
    title: "End-to-End Pipeline",
    eq: "P(y_{1:T} | X) = \\prod_{t=1}^T P(y_t \\mid y_{<t}, X)",
    desc: "Follow the full image → language story.",
    moduleId: "pipeline",
  },
];

export function HomeDashboard() {
  const setActiveModule = useAppStore((s) => s.setActiveModule);

  return (
    <div className="space-y-6">
      <Card className="overflow-hidden border-2 border-primary/20">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Sigma className="h-6 w-6 text-primary" />
            <CardTitle className="text-2xl md:text-3xl">Mathematics of Computer Vision Language Models</CardTitle>
          </div>
          <CardDescription className="text-base">
            An interactive educational tool that visualizes, calculates, and explains the mathematics between
            an input image and a language-model output.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <p>
            This is not a chatbot. Every page exposes the underlying <strong>matrices, tensors, and operations</strong>{" "}
            (convolution, attention, softmax, cross-entropy) explicitly — no hidden{" "}
            <code className="rounded bg-muted px-1">torch.nn.MultiheadAttention</code> calls in the educational path.
          </p>
          <p className="text-muted-foreground">
            Built for an undergraduate Mathematics + AI/ML thesis. Use the sidebar to navigate through 18 modules, a
            mathematical sandbox, dimension tracker, viva mode, and a final thesis demo.
          </p>
          <div className="flex flex-wrap gap-2 pt-2">
            <Button onClick={() => setActiveModule("image-repr")}>
              Start from Module 1 <ArrowRight className="ml-1 h-4 w-4" />
            </Button>
            <Button variant="outline" onClick={() => setActiveModule("self-attention")}>
              Jump to Self-Attention
            </Button>
            <Button variant="outline" onClick={() => setActiveModule("thesis-demo")}>
              Thesis Demo
            </Button>
          </div>
        </CardContent>
      </Card>

      <div>
        <h2 className="mb-3 text-lg font-semibold">Highlights</h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {HIGHLIGHTS.map((h) => (
            <Card
              key={h.moduleId}
              className="cursor-pointer transition-colors hover:border-primary/40 hover:bg-accent/30"
              onClick={() => setActiveModule(h.moduleId)}
            >
              <CardHeader className="pb-2">
                <div className="flex items-center gap-2">
                  <h.icon className="h-4 w-4 text-primary" />
                  <CardTitle className="text-base">{h.title}</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="rounded bg-muted/50 p-2 text-center">
                  <MathBlock>{h.eq}</MathBlock>
                </div>
                <p className="text-xs text-muted-foreground">{h.desc}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">All Modules</CardTitle>
          <CardDescription>22 interactive modules across 7 categories.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-1 sm:grid-cols-2 lg:grid-cols-3">
            {MODULES.map((m) => (
              <button
                key={m.id}
                onClick={() => setActiveModule(m.id)}
                className="flex items-start gap-2 rounded p-2 text-left hover:bg-accent/40"
              >
                <span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded bg-muted font-mono text-[10px] font-bold">
                  {m.number}
                </span>
                <span className="flex-1">
                  <span className="block text-sm font-medium leading-tight">{m.shortTitle}</span>
                  <span className="block text-[10px] text-muted-foreground">{m.description}</span>
                </span>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card className="border-amber-500/30 bg-amber-500/5">
        <CardContent className="pt-4 text-xs text-amber-900 dark:text-amber-200">
          <strong>Academic honesty note.</strong> This is a <em>simplified educational representation</em>. Real
          production VLMs (BLIP-2, LLaVA, etc.) use additional normalization, modality-specific components, larger
          dimensions, and optimization tricks that are out of scope here. Each module flags where simplifications are
          made.
        </CardContent>
      </Card>
    </div>
  );
}
