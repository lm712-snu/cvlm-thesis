"use client";

import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface ModuleHeaderProps {
  number: number;
  title: string;
  subtitle?: string;
  children?: ReactNode; // equation summary
}

/** Consistent header for each module: shows module number, title, and a one-line equation summary. */
export function ModuleHeader({ number, title, subtitle, children }: ModuleHeaderProps) {
  return (
    <div className="mb-6 space-y-2">
      <div className="flex items-center gap-2 text-xs uppercase tracking-widest text-muted-foreground">
        <span className="font-mono">Module {String(number).padStart(2, "0")}</span>
      </div>
      <h1 className="text-2xl md:text-3xl font-bold tracking-tight">{title}</h1>
      {subtitle && <p className="text-muted-foreground text-sm md:text-base">{subtitle}</p>}
      {children && <div className={cn("mt-3 rounded-lg border bg-muted/30 p-3 text-sm")}>{children}</div>}
    </div>
  );
}
