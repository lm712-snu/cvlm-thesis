"use client";

import { cn } from "@/lib/utils";

interface HeatmapProps {
  matrix: number[][];
  /** Diverging (red-blue) vs sequential (blue). */
  diverging?: boolean;
  /** Cell label rendering function. */
  format?: (v: number) => string;
  /** Highlight cell. */
  highlight?: [number, number];
  onCellClick?: (i: number, j: number, v: number) => void;
  rowLabels?: string[];
  colLabels?: string[];
  className?: string;
  /** Min/max for fixed color scale (otherwise auto). */
  min?: number;
  max?: number;
  cellSize?: number;
}

/** A heatmap of an arbitrary matrix. Useful for attention weights, positional encodings, etc. */
export function Heatmap({
  matrix,
  diverging = false,
  format,
  highlight,
  onCellClick,
  rowLabels,
  colLabels,
  className,
  min,
  max,
  cellSize = 28,
}: HeatmapProps) {
  if (!matrix || matrix.length === 0) return null;
  let mn = min ?? Infinity;
  let mx = max ?? -Infinity;
  if (min === undefined || max === undefined) {
    for (const row of matrix) for (const v of row) {
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  const range = mx - mn || 1;

  function color(v: number): string {
    const t = (v - mn) / range;
    if (diverging) {
      // red (negative) -> white (zero) -> blue (positive)
      const norm = mn < 0 && mx > 0 ? v / Math.max(Math.abs(mn), Math.abs(mx)) : 2 * t - 1;
      if (norm >= 0) {
        const b = Math.round(255 * norm);
        return `rgb(${255 - b}, ${255 - b}, 255)`;
      } else {
        const r = Math.round(255 * -norm);
        return `rgb(255, ${255 - r}, ${255 - r})`;
      }
    } else {
      const c = Math.round(255 * (1 - t));
      return `rgb(${c}, ${c}, 255)`;
    }
  }

  return (
    <div className={cn("inline-block overflow-x-auto", className)}>
      <div className="inline-grid" style={{ gridTemplateColumns: `auto repeat(${matrix[0].length}, ${cellSize}px)` }}>
        {colLabels && (
          <>
            <div />
            {colLabels.map((l, j) => (
              <div key={j} className="text-[9px] text-muted-foreground text-center pb-0.5 font-mono truncate" style={{ width: cellSize }}>
                {l}
              </div>
            ))}
          </>
        )}
        {matrix.map((row, i) => (
          <div key={i} className="contents">
            {rowLabels && (
              <div className="text-[9px] text-muted-foreground text-right pr-1 font-mono truncate" style={{ width: 40 }}>
                {rowLabels[i]}
              </div>
            )}
            {row.map((v, j) => {
              const isHi = highlight && highlight[0] === i && highlight[1] === j;
              return (
                <div
                  key={j}
                  onClick={onCellClick ? () => onCellClick(i, j, v) : undefined}
                  className={cn(
                    "border border-background/40 flex items-center justify-center text-[9px] font-mono text-black/80 transition-transform",
                    onCellClick && "cursor-pointer hover:scale-110 hover:z-10 hover:ring-2 hover:ring-primary",
                    isHi && "ring-2 ring-primary scale-110 z-10"
                  )}
                  style={{ width: cellSize, height: cellSize, backgroundColor: color(v) }}
                  title={`[${i},${j}] = ${v.toFixed(6)}`}
                >
                  {format ? format(v) : ""}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
