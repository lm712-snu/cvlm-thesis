"use client";

import { ChevronLeft, ChevronRight, Home } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAppStore } from "@/store/app-store";
import { MODULES } from "@/data/modules";

/**
 * Prev / Next navigation between modules.
 * Renders at the bottom of every module page so the user can read through
 * the thesis sequentially without returning to the sidebar each time.
 *
 * Wraps around the entire MODULES catalog (Modules 1..22) in catalog order.
 */
export function ModuleNav() {
  const activeModule = useAppStore((s) => s.activeModule);
  const setActiveModule = useAppStore((s) => s.setActiveModule);

  // Don't show on the home dashboard.
  if (activeModule === "home") return null;

  const idx = MODULES.findIndex((m) => m.id === activeModule);
  if (idx === -1) return null;

  const prev = idx > 0 ? MODULES[idx - 1] : null;
  const next = idx < MODULES.length - 1 ? MODULES[idx + 1] : null;

  return (
    <nav
      aria-label="Module navigation"
      className="mt-10 flex items-center justify-between gap-3 border-t pt-4"
    >
      <div className="flex-1">
        {prev ? (
          <Button
            variant="outline"
            size="sm"
            onClick={() => setActiveModule(prev.id)}
            className="group max-w-[45%] flex flex-col items-start gap-0.5 h-auto py-2 text-left"
          >
            <span className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-muted-foreground group-hover:text-foreground">
              <ChevronLeft className="h-3 w-3" /> Module {String(prev.number).padStart(2, "0")}
            </span>
            <span className="font-medium text-sm normal-case">{prev.shortTitle}</span>
          </Button>
        ) : (
          <Button variant="ghost" size="sm" onClick={() => setActiveModule("home")}>
            <Home className="h-4 w-4 mr-1" /> Home
          </Button>
        )}
      </div>

      <div className="flex-1 flex justify-end">
        {next ? (
          <Button
            variant="default"
            size="sm"
            onClick={() => setActiveModule(next.id)}
            className="group max-w-[45%] flex flex-col items-end gap-0.5 h-auto py-2 text-right"
          >
            <span className="flex items-center gap-1 text-[10px] uppercase tracking-wider opacity-80 group-hover:opacity-100">
              Module {String(next.number).padStart(2, "0")} <ChevronRight className="h-3 w-3" />
            </span>
            <span className="font-medium text-sm">{next.shortTitle}</span>
          </Button>
        ) : (
          <Button variant="ghost" size="sm" onClick={() => setActiveModule("home")}>
            Finish <Home className="h-4 w-4 ml-1" />
          </Button>
        )}
      </div>
    </nav>
  );
}
