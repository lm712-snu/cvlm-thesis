"use client";

import { useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface SectionProps {
  title: string;
  icon?: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
  variant?: "math" | "why" | "how" | "numerical" | "visual" | "viva" | "intuition";
  className?: string;
}

const VARIANTS: Record<NonNullable<SectionProps["variant"]>, { label: string; ring: string; chip: string }> = {
  math: { label: "Mathematics", ring: "border-sky-500/40", chip: "bg-sky-500/10 text-sky-700 dark:text-sky-300" },
  why: { label: "Why?", ring: "border-rose-500/40", chip: "bg-rose-500/10 text-rose-700 dark:text-rose-300" },
  how: { label: "How?", ring: "border-emerald-500/40", chip: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300" },
  numerical: { label: "Numerical Example", ring: "border-amber-500/40", chip: "bg-amber-500/10 text-amber-700 dark:text-amber-300" },
  visual: { label: "Visual Interpretation", ring: "border-violet-500/40", chip: "bg-violet-500/10 text-violet-700 dark:text-violet-300" },
  viva: { label: "Viva Question", ring: "border-fuchsia-500/40", chip: "bg-fuchsia-500/10 text-fuchsia-700 dark:text-fuchsia-300" },
  intuition: { label: "Intuition", ring: "border-cyan-500/40", chip: "bg-cyan-500/10 text-cyan-700 dark:text-cyan-300" },
};

/** Expandable section used throughout modules: "Mathematics", "Why?", "Numerical Example", etc. */
export function ExpandableSection({
  title,
  icon,
  children,
  defaultOpen = false,
  variant = "math",
  className,
}: SectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  const v = VARIANTS[variant];
  return (
    <div className={cn("rounded-lg border bg-card/50", v.ring, className)}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-2 p-3 text-left hover:bg-accent/30 transition-colors"
        aria-expanded={open}
      >
        <div className="flex items-center gap-2">
          {icon}
          <span className={cn("text-[10px] uppercase tracking-wider px-2 py-0.5 rounded font-semibold", v.chip)}>
            {v.label}
          </span>
          <span className="font-medium text-sm">{title}</span>
        </div>
        <ChevronDown className={cn("h-4 w-4 text-muted-foreground transition-transform", open && "rotate-180")} />
      </button>
      {open && <div className="px-3 pb-3 pt-1 text-sm leading-relaxed">{children}</div>}
    </div>
  );
}
