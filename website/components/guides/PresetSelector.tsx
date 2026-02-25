"use client";

import { useCallback } from "react";
import { cn } from "@/lib/utils";
import type { MatrixValues } from "./MatrixInput";

export const PRESETS: { name: string; matrix: MatrixValues; title: string }[] = [
  { name: "Identity", matrix: { a: 1, b: 0, c: 0, d: 1 }, title: "Eigenvalues at (1,0). Grounding." },
  { name: "Rotation", matrix: { a: 0, b: -1, c: 1, d: 0 }, title: "Pure imaginary eigenvalues; marginally stable." },
  { name: "Shear", matrix: { a: 1, b: 1, c: 0, d: 1 }, title: "Repeated eigenvalue at (1,0). Defective matrix." },
  { name: "Unstable", matrix: { a: 1, b: 2, c: 0, d: 3 }, title: "Both eigenvalues positive real. Unstable." },
  { name: "Oscillatory", matrix: { a: 0, b: 1, c: -2, d: -0.1 }, title: "Complex pair in left-half plane. Damped oscillation." },
  { name: "Saddle", matrix: { a: -1, b: 0, c: 0, d: 2 }, title: "One stable, one unstable. Saddle point." },
];

export interface PresetSelectorProps {
  onSelect: (matrix: MatrixValues) => void;
  className?: string;
}

export function PresetSelector({ onSelect, className }: PresetSelectorProps) {
  const handleClick = useCallback(
    (matrix: MatrixValues) => () => {
      onSelect(matrix);
    },
    [onSelect]
  );

  return (
    <div className={cn("flex flex-wrap gap-2", className)}>
      {PRESETS.map(({ name, matrix, title }) => (
        <button
          key={name}
          type="button"
          onClick={handleClick(matrix)}
          title={title}
          className="rounded-full border border-[#8b5cf6]/50 px-3 py-1.5 text-xs font-medium text-[#a78bfa] transition-colors hover:border-[#8b5cf6]/80 hover:bg-[#8b5cf6]/10 hover:text-[#c4b5fd]"
        >
          {name}
        </button>
      ))}
    </div>
  );
}
