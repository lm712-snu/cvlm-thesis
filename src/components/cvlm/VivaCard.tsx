"use client";

import { useState, type ReactNode } from "react";
import { Lightbulb, Eye, EyeOff } from "lucide-react";
import { cn } from "@/lib/utils";

export interface VivaQuestion {
  level: "Easy" | "Medium" | "Difficult";
  question: string;
  hint: string;
  answer: string;
  explanation: string;
}

interface VivaCardProps {
  question: VivaQuestion;
}

const LEVEL_STYLES = {
  Easy: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border-emerald-500/30",
  Medium: "bg-amber-500/10 text-amber-700 dark:text-amber-300 border-amber-500/30",
  Difficult: "bg-rose-500/10 text-rose-700 dark:text-rose-300 border-rose-500/30",
};

/** A single viva question with hint + answer reveal. */
export function VivaCard({ question }: VivaCardProps) {
  const [showHint, setShowHint] = useState(false);
  const [showAnswer, setShowAnswer] = useState(false);

  return (
    <div className={cn("rounded-lg border p-4 space-y-3", LEVEL_STYLES[question.level])}>
      <div className="flex items-center gap-2">
        <Lightbulb className="h-4 w-4" />
        <span className="text-[10px] font-bold uppercase tracking-wider">{question.level}</span>
      </div>
      <p className="font-medium text-foreground">{question.question}</p>

      <div className="space-y-2">
        <button
          onClick={() => setShowHint((s) => !s)}
          className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          <Eye className="h-3 w-3" /> {showHint ? "Hide hint" : "Show hint"}
        </button>
        {showHint && (
          <p className="text-sm italic text-muted-foreground pl-4 border-l-2 border-border">{question.hint}</p>
        )}
      </div>

      <div className="space-y-2">
        <button
          onClick={() => setShowAnswer((s) => !s)}
          className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          {showAnswer ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
          {showAnswer ? "Hide answer" : "Reveal answer"}
        </button>
        {showAnswer && (
          <div className="space-y-2 pl-4 border-l-2 border-border">
            <p className="text-sm font-medium text-foreground">{question.answer}</p>
            <p className="text-sm text-muted-foreground">{question.explanation}</p>
          </div>
        )}
      </div>
    </div>
  );
}

/** A list of viva questions, grouped by level. */
export function VivaPanel({ questions }: { questions: VivaQuestion[] }) {
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
        <Lightbulb className="h-4 w-4" /> Viva Mode — Practice Questions
      </h3>
      <div className="grid gap-3 md:grid-cols-3">
        {questions.map((q, i) => (
          <VivaCard key={i} question={q} />
        ))}
      </div>
    </div>
  );
}
