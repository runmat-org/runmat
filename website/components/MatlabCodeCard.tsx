"use client";

import { cn } from "@/lib/utils";
import Prism from "prismjs";
import "prismjs/components/prism-matlab";
import { useMemo } from "react";

const sampleCode = `
x = 0:0.001:4*pi;           % 0 to 4π in steps of 0.001
y = sin(x) .* exp(-x/10);   % regular MATLAB language math
`.trim();

interface MatlabCodeCardProps {
  code?: string;
  className?: string;
}

export function MatlabCodeCard({ code = sampleCode, className }: MatlabCodeCardProps) {
  const highlighted = useMemo(() => Prism.highlight(code, Prism.languages.matlab, "matlab"), [code]);

  return (
    <div className={cn("w-full max-w-3xl rounded-3xl border border-white/10 bg-gradient-to-b from-[#131a2a] to-[#060a12] p-4 shadow-2xl ", className)}>
      <div className="flex items-center gap-2 rounded-2xl bg-[#0d1422] px-4 py-3">
        <div className="flex gap-2">
          <span className="h-3.5 w-3.5 rounded-full bg-[#ff5f56]" />
          <span className="h-3.5 w-3.5 rounded-full bg-[#ffbd2e]" />
          <span className="h-3.5 w-3.5 rounded-full bg-[#27c93f]" />
        </div>
      </div>
      <pre
        className="mt-3 overflow-x-auto rounded-2xl bg-[#0d1422] p-1 text-left font-mono text-lg leading-8 text-white language-example"
        tabIndex={0}
        suppressHydrationWarning
      >
        <code
          className="language-matlab"
          dangerouslySetInnerHTML={{ __html: highlighted }}
          suppressHydrationWarning
        />
      </pre>
    </div>
  );
}

export default MatlabCodeCard;


