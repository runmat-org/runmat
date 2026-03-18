"use client";

import { useMemo } from "react";
import { cn } from "@/lib/utils";
import { getWindowCoefficients, type WindowType } from "./FFTEngine";

const WINDOW_OPTIONS: { value: WindowType; label: string }[] = [
  { value: "rectangular", label: "Rectangular" },
  { value: "hann", label: "Hann" },
  { value: "hamming", label: "Hamming" },
  { value: "blackman-harris", label: "Blackman-Harris" },
];

const PREVIEW_N = 64;
const PREVIEW_W = 80;
const PREVIEW_H = 28;

function WindowPreview({ windowType }: { windowType: WindowType }) {
  const points = useMemo(() => {
    const coeffs = getWindowCoefficients(PREVIEW_N, windowType);
    return coeffs
      .map((c, i) => {
        const x = (i / (PREVIEW_N - 1)) * PREVIEW_W;
        const y = PREVIEW_H - c * (PREVIEW_H - 2);
        return `${x},${y}`;
      })
      .join(" ");
  }, [windowType]);

  return (
    <svg
      width={PREVIEW_W}
      height={PREVIEW_H}
      viewBox={`0 0 ${PREVIEW_W} ${PREVIEW_H}`}
      className="shrink-0"
      aria-hidden
    >
      <polyline
        points={points}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-primary"
      />
    </svg>
  );
}

export interface WindowSelectorProps {
  value: WindowType;
  onChange: (w: WindowType) => void;
  className?: string;
}

export function WindowSelector({ value, onChange, className }: WindowSelectorProps) {
  return (
    <div className={cn("rounded-lg border border-border bg-card p-3 space-y-3", className)}>
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-foreground">Window function</h2>
        <WindowPreview windowType={value} />
      </div>
      <div className="flex flex-wrap gap-2">
        {WINDOW_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            className={cn(
              "rounded-full border px-3.5 py-1.5 text-sm font-medium transition-colors cursor-pointer",
              opt.value === value
                ? "border-primary bg-primary/15 text-primary shadow-sm shadow-primary/10"
                : "border-border/80 bg-muted/50 text-muted-foreground hover:border-primary/50 hover:text-foreground hover:bg-muted/80"
            )}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}
