"use client";

import { useEffect, useMemo, useState } from "react";
import { Menu, X, Sun, Moon, Sigma, Home } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store/app-store";
import { MODULES, CATEGORY_LABELS, type ModuleMeta } from "@/data/modules";

interface SidebarProps {
  open: boolean;
  onClose: () => void;
}

export function Sidebar({ open, onClose }: SidebarProps) {
  const activeModule = useAppStore((s) => s.activeModule);
  const setActiveModule = useAppStore((s) => s.setActiveModule);

  const grouped = useMemo(() => {
    const map: Record<ModuleMeta["category"], ModuleMeta[]> = {
      image: [], vision: [], attention: [], language: [], alignment: [], pipeline: [], tools: [],
    };
    for (const m of MODULES) map[m.category].push(m);
    return map;
  }, []);

  return (
    <>
      {/* Backdrop overlay when sidebar is open (any screen size) */}
      {open && (
        <div
          className="fixed inset-0 z-30 bg-black/40 backdrop-blur-sm"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 w-72 transform border-r bg-sidebar/95 backdrop-blur transition-transform",
          open ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="flex h-14 items-center justify-between border-b px-4">
          <div className="flex items-center gap-2">
            <Sigma className="h-5 w-5 text-primary" />
            <div>
              <p className="text-sm font-bold leading-none">Math of CVLMs</p>
              <p className="text-[10px] text-muted-foreground">Undergraduate Thesis Tool</p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        <nav className="flex h-[calc(100vh-3.5rem)] flex-col overflow-y-auto p-2">
          <div className="mb-3">
            <button
              onClick={() => {
                setActiveModule("home");
                onClose();
              }}
              className={cn(
                "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors",
                activeModule === "home"
                  ? "bg-primary text-primary-foreground"
                  : "hover:bg-accent text-foreground/80 hover:text-foreground"
              )}
            >
              <Home className="h-4 w-4" />
              <span className="font-medium">Home</span>
            </button>
          </div>
          {(Object.keys(grouped) as ModuleMeta["category"][]).map((cat) => {
            const items = grouped[cat];
            if (items.length === 0) return null;
            return (
              <div key={cat} className="mb-3">
                <p className="px-2 py-1 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                  {CATEGORY_LABELS[cat]}
                </p>
                <div className="space-y-0.5">
                  {items.map((m) => (
                    <button
                      key={m.id}
                      onClick={() => {
                        setActiveModule(m.id);
                        onClose();
                      }}
                      className={cn(
                        "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors",
                        activeModule === m.id
                          ? "bg-primary text-primary-foreground"
                          : "hover:bg-accent text-foreground/80 hover:text-foreground"
                      )}
                    >
                      <span className={cn(
                        "inline-flex h-5 w-5 shrink-0 items-center justify-center rounded text-[10px] font-mono font-bold",
                        activeModule === m.id ? "bg-primary-foreground/20" : "bg-muted"
                      )}>
                        {m.number}
                      </span>
                      <span className="font-medium">{m.shortTitle}</span>
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
        </nav>
      </aside>
    </>
  );
}

export function TopBar() {
  const [menuOpen, setMenuOpen] = useState(false);
  const theme = useAppStore((s) => s.theme);
  const toggleTheme = useAppStore((s) => s.toggleTheme);
  const setTheme = useAppStore((s) => s.setTheme);
  const activeModule = useAppStore((s) => s.activeModule);
  const active = MODULES.find((m) => m.id === activeModule);

  useEffect(() => {
    try {
      const saved = localStorage.getItem("cvlm-theme") as "light" | "dark" | null;
      if (saved) setTheme(saved);
      else if (window.matchMedia?.("(prefers-color-scheme: dark)").matches) setTheme("dark");
    } catch {
      /* ignore */
    }
  }, [setTheme]);

  return (
    <>
      <header className="sticky top-0 z-30 flex h-14 items-center justify-between border-b bg-background/95 px-4 backdrop-blur">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={() => setMenuOpen(true)} title="Open menu">
            <Menu className="h-4 w-4" />
          </Button>
          <div>
            {active ? (
              <>
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                  Module {String(active.number).padStart(2, "0")}
                </p>
                <p className="text-sm font-semibold leading-none">{active.shortTitle}</p>
              </>
            ) : (
              <>
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Dashboard</p>
                <p className="text-sm font-semibold leading-none">Math of CVLMs</p>
              </>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon" onClick={toggleTheme} title="Toggle theme">
            {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>
        </div>
      </header>

      <Sidebar open={menuOpen} onClose={() => setMenuOpen(false)} />
    </>
  );
}
