"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import { formatComplex } from "./EigenSolver";
import type { EigenResult } from "./EigenSolver";

function round2(x: number): number {
  return Math.round(x * 100) / 100;
}

export interface EigenvalueLabelsProps {
  result: EigenResult;
  className?: string;
}

export function EigenvalueLabels({ result, className }: EigenvalueLabelsProps) {
  const [showDetails, setShowDetails] = useState(false);
  const { lambda1, lambda2, trace, determinant, discriminant } = result;

  const line1 = `λ₁ = ${formatComplex(lambda1.re, lambda1.im)}`;
  const line2 = `λ₂ = ${formatComplex(lambda2.re, lambda2.im)}`;

  return (
    <div className={cn("space-y-2", className)}>
      <p className="text-sm font-mono text-foreground">
        {line1}
        <br />
        {line2}
      </p>
      <button
        type="button"
        onClick={() => setShowDetails((s) => !s)}
        className="text-xs text-muted-foreground hover:text-foreground underline"
      >
        {showDetails ? "Hide" : "Show"} trace, det, discriminant
      </button>
      {showDetails && (
        <p className="text-xs font-mono text-muted-foreground">
          trace = {round2(trace)}, det = {round2(determinant)}, disc = {round2(discriminant)}
        </p>
      )}
    </div>
  );
}
