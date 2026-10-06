"use client";

import { cn } from "@/lib/utils";

interface DimBadgeProps {
  dims: string;
  /** Color hint for the role of this tensor. */
  variant?: "input" | "weight" | "output" | "intermediate" | "label";
  className?: string;
}

const VARIANTS: Record<NonNullable<DimBadgeProps["variant"]>, string> = {
  input: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/30",
  weight: "bg-amber-500/15 text-amber-700 dark:text-amber-300 border-amber-500/30",
  output: "bg-violet-500/15 text-violet-700 dark:text-violet-300 border-violet-500/30",
  intermediate: "bg-sky-500/15 text-sky-700 dark:text-sky-300 border-sky-500/30",
  label: "bg-rose-500/15 text-rose-700 dark:text-rose-300 border-rose-500/30",
};

/** Small pill that shows tensor dimensions like "196 × 768" or "(B, n, d)". */
export function DimBadge({ dims, variant = "intermediate", className }: DimBadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 rounded-md border text-xs font-mono font-medium",
        VARIANTS[variant],
        className
      )}
    >
      {dims}
    </span>
  );
}
