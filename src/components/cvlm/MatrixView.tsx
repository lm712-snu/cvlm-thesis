"use client";

import { cn } from "@/lib/utils";

interface MatrixViewProps {
  matrix: number[][];
  /** Highlight a single cell (i, j). */
  highlight?: [number, number];
  /** Heatmap coloring: scale each cell by its value relative to the matrix min/max. */
  heatmap?: boolean;
  /** Diverging colormap (red-white-blue) for values centered around 0. */
  diverging?: boolean;
  /** Number of decimal places. */
  digits?: number;
  /** Show row/col labels. */
  rowLabels?: string[];
  colLabels?: string[];
  className?: string;
  /** Clickable cell callback. */
  onCellClick?: (i: number, j: number, v: number) => void;
  /** Font size for cells. */
  cellSize?: "xs" | "sm" | "md";
}

/** Render a 2D matrix as an HTML table with optional heatmap coloring. */
export function MatrixView({
  matrix,
  highlight,
  heatmap = false,
  diverging = false,
  digits = 3,
  rowLabels,
  colLabels,
  className,
  onCellClick,
  cellSize = "sm",
}: MatrixViewProps) {
  if (!matrix || matrix.length === 0) return null;

  // Compute min/max across the matrix for heatmap scaling.
  let mn = Infinity;
  let mx = -Infinity;
  for (const row of matrix) for (const v of row) {
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  const range = mx - mn || 1;

  const cellPad =
    cellSize === "xs" ? "px-1 py-0.5 text-[10px]" : cellSize === "sm" ? "px-1.5 py-1 text-xs" : "px-2 py-1 text-sm";

  return (
    <div className={cn("inline-block overflow-x-auto", className)}>
      <table className="border-collapse">
        <tbody>
          {colLabels && (
            <tr>
              {rowLabels && <td className="px-1.5 py-1" />}
              {colLabels.map((l, j) => (
                <td key={j} className="px-1.5 py-1 text-[10px] text-muted-foreground text-center font-mono">
                  {l}
                </td>
              ))}
            </tr>
          )}
          {matrix.map((row, i) => (
            <tr key={i}>
              {rowLabels && (
                <td className="px-1.5 py-1 text-[10px] text-muted-foreground text-right font-mono whitespace-nowrap">
                  {rowLabels[i]}
                </td>
              )}
              {row.map((v, j) => {
                const isHi = highlight && highlight[0] === i && highlight[1] === j;
                let bg = "";
                if (heatmap) {
                  if (diverging) {
                    // map v in [-1, 1] (or actual range) to a red-white-blue color
                    const t = (v - mn) / range; // 0..1
                    const centered = mn < 0 && mx > 0 ? (v / Math.max(Math.abs(mn), Math.abs(mx)) + 1) / 2 : t;
                    const r = Math.round(255 * (1 - centered));
                    const b = Math.round(255 * centered);
                    bg = `rgb(${r}, 240, ${b})`;
                  } else {
                    const t = (v - mn) / range;
                    const c = Math.round(255 * (1 - t));
                    bg = `rgb(${c}, ${c}, 255)`;
                  }
                }
                return (
                  <td
                    key={j}
                    onClick={onCellClick ? () => onCellClick(i, j, v) : undefined}
                    className={cn(
                      "border border-border/60 text-center font-mono tabular-nums transition-colors",
                      cellPad,
                      onCellClick && "cursor-pointer hover:ring-2 hover:ring-primary/40",
                      isHi && "ring-2 ring-primary outline-none",
                      heatmap && "text-black"
                    )}
                    style={heatmap ? { backgroundColor: bg } : undefined}
                    title={`[${i},${j}] = ${v.toFixed(6)}`}
                  >
                    {Math.abs(v) < 1e-10 ? "0.000" : v.toFixed(digits)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
