"use client";

import { useCallback, useMemo, useState } from "react";
import Link from "next/link";
import { MatrixInput } from "./MatrixInput";
import { PresetSelector } from "./PresetSelector";
import { ComplexPlaneMotion as ComplexPlane } from "./ComplexPlaneMotion";
import { StabilityBadge } from "./StabilityBadge";
import { EigenvalueLabels } from "./EigenvalueLabels";
import { RunInRunMatButton } from "./RunInRunMatButton";
import { solveEig } from "./EigenSolver";
import type { MatrixValues } from "./MatrixInput";

const EIG_SOURCE_URL =
  "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/eig.rs";
const EIG_DOCS_PATH = "/docs/reference/builtins/eig";

const DEFAULT_MATRIX: MatrixValues = { a: 1, b: 0, c: 0, d: 1 };

export function EigenvalueExplorerClient() {
  const [matrix, setMatrix] = useState<MatrixValues>(DEFAULT_MATRIX);

  const result = useMemo(
    () => solveEig(matrix.a, matrix.b, matrix.c, matrix.d),
    [matrix.a, matrix.b, matrix.c, matrix.d]
  );

  const isComplexPair = result.lambda1.im !== 0 || result.lambda2.im !== 0;

  const handlePreset = useCallback((preset: MatrixValues) => {
    setMatrix(preset);
  }, []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
      <div className="space-y-4">
        <div className="rounded-lg border border-border bg-card p-4">
          <h2 className="text-sm font-medium text-foreground mb-3">2×2 matrix</h2>
          <MatrixInput value={matrix} onChange={setMatrix} />
        </div>
        <PresetSelector onSelect={handlePreset} />
        <div className="flex flex-col gap-3 pt-2">
          <RunInRunMatButton />
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm">
            <Link
              href={EIG_SOURCE_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground underline"
            >
              View eig source on GitHub
            </Link>
            <Link
              href={EIG_DOCS_PATH}
              className="text-muted-foreground hover:text-foreground underline"
            >
              eig in Docs
            </Link>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <ComplexPlane
          lambda1={result.lambda1}
          lambda2={result.lambda2}
          isComplexPair={isComplexPair}
        />
        <div className="flex flex-wrap items-center gap-3">
          <EigenvalueLabels result={result} />
          <StabilityBadge status={result.stability} />
        </div>
      </div>
    </div>
  );
}
