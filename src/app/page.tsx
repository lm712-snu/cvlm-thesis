"use client";

import type { ComponentType } from "react";
import { TopBar } from "@/components/cvlm/Shell";
import { useAppStore } from "@/store/app-store";
import { HomeDashboard } from "@/components/modules/HomeDashboard";
import { Module1_ImageRepresentation } from "@/components/modules/Module1_ImageRepresentation";
import { Module2_Convolution } from "@/components/modules/Module2_Convolution";
import { Module3_CNN } from "@/components/modules/Module3_CNN";
import { Module4_Patches } from "@/components/modules/Module4_Patches";
import { Module5_PatchEmbedding } from "@/components/modules/Module5_PatchEmbedding";
import { Module6_Positional } from "@/components/modules/Module6_Positional";
import { Module7_ViT } from "@/components/modules/Module7_ViT";
import { Module8_SelfAttention } from "@/components/modules/Module8_SelfAttention";
import { Module9_MultiHead } from "@/components/modules/Module9_MultiHead";
import { Module10_TextRepresentation } from "@/components/modules/Module10_Text";
import { Module11_Alignment } from "@/components/modules/Module11_Alignment";
import { Module12_Contrastive } from "@/components/modules/Module12_Contrastive";
import { Module13_Architecture } from "@/components/modules/Module13_Architecture";
import { Module14_Projector } from "@/components/modules/Module14_Projector";
import { Module15_CrossAttention } from "@/components/modules/Module15_CrossAttention";
import { Module16_NextToken } from "@/components/modules/Module16_NextToken";
import { Module17_Autoregressive } from "@/components/modules/Module17_Autoregressive";
import { Module18_Pipeline } from "@/components/modules/Module18_Pipeline";
import { Module19_Sandbox } from "@/components/modules/Module19_Sandbox";
import { Module20_DimensionTracker } from "@/components/modules/Module20_DimensionTracker";
import { Module21_ThesisDemo } from "@/components/modules/Module21_ThesisDemo";
import { Module22_VivaMode } from "@/components/modules/Module22_VivaMode";

const MODULE_MAP: Record<string, ComponentType> = {
  home: HomeDashboard,
  "image-repr": Module1_ImageRepresentation,
  convolution: Module2_Convolution,
  cnn: Module3_CNN,
  patches: Module4_Patches,
  "patch-embed": Module5_PatchEmbedding,
  positional: Module6_Positional,
  vit: Module7_ViT,
  "self-attention": Module8_SelfAttention,
  multihead: Module9_MultiHead,
  "text-repr": Module10_TextRepresentation,
  alignment: Module11_Alignment,
  contrastive: Module12_Contrastive,
  architecture: Module13_Architecture,
  projector: Module14_Projector,
  "cross-attention": Module15_CrossAttention,
  "next-token": Module16_NextToken,
  autoregressive: Module17_Autoregressive,
  pipeline: Module18_Pipeline,
  sandbox: Module19_Sandbox,
  "dimension-tracker": Module20_DimensionTracker,
  "thesis-demo": Module21_ThesisDemo,
  viva: Module22_VivaMode,
};

export default function Home() {
  const activeModule = useAppStore((s) => s.activeModule);
  const Active = MODULE_MAP[activeModule] ?? HomeDashboard;

  return (
    <div className="min-h-screen bg-background">
      <TopBar />
      <main className="mx-auto max-w-7xl px-4 py-6 md:px-6">
        <Active />
      </main>
      <footer className="mt-auto border-t px-4 py-3 text-center text-[10px] text-muted-foreground">
        <p>
          Mathematics of Computer Vision Language Models — Undergraduate Thesis Tool ·
          Educational implementation; math computed client-side ·
          Deploy on GitHub Pages via <code className="font-mono">next build &amp;&amp; next export</code>
        </p>
      </footer>
    </div>
  );
}
