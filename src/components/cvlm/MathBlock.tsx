"use client";

import { Component, type ReactNode } from "react";
import { BlockMath, InlineMath } from "react-katex";
import "katex/dist/katex.min.css";

interface MathProps {
  children: string;
  block?: boolean;
  className?: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
}

/** Error boundary so that a single bad LaTeX string does not break the whole page. */
class MathErrorBoundary extends Component<{ children: ReactNode; fallback: ReactNode }, ErrorBoundaryState> {
  constructor(props: { children: ReactNode; fallback: ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }
  render() {
    if (this.state.hasError) return this.props.fallback;
    return this.props.children;
  }
}

/** Renders LaTeX math via KaTeX. Use `block` for display-mode equations. */
export function MathBlock({ children, block = false, className }: MathProps) {
  const fallback = <code className="text-xs">{children}</code>;
  return (
    <MathErrorBoundary fallback={fallback}>
      {block ? (
        <div className={`my-2 overflow-x-auto ${className ?? ""}`}>
          <BlockMath math={children} />
        </div>
      ) : (
        <span className={className}>
          <InlineMath math={children} />
        </span>
      )}
    </MathErrorBoundary>
  );
}
