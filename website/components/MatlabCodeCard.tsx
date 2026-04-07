"use client";

import { cn } from "@/lib/utils";
import Prism from "prismjs";
import "prismjs/components/prism-matlab";
import { useMemo } from "react";
import { TryInBrowserButton } from "@/components/TryInBrowserButton";
import { openWorkspace } from "@/lib/desktop";

const sampleCode = `
x = 0:0.001:4*pi;           % 0 to 4π in steps of 0.001
y = sin(x) .* exp(-x/10);   % regular MATLAB language math
`.trim();

interface MatlabCodeCardProps {
  code?: string;
  className?: string;
}

export function MatlabCodeCard({ code = sampleCode, className }: MatlabCodeCardProps) {
  const highlightedLines = useMemo(() => {
    const html = Prism.highlight(code, Prism.languages.matlab, "matlab");
    return html.split("\n");
  }, [code]);

  const openInBrowser = () =>
    openWorkspace([{ path: "/example.m", content: code }], { targetPath: "/sandbox", metadata: { source: "example-page" } });

  return (
    <div
      className={cn(
        "matlab-code-card-hero w-full max-w-3xl overflow-hidden rounded-xl cursor-pointer",
        "border border-border bg-[var(--editor-background)]",
        "shadow-sm hover:shadow-md transition-shadow",
        className
      )}
      role="link"
      tabIndex={0}
      onClick={openInBrowser}
      onKeyDown={event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openInBrowser();
        }
      }}
    >
      {/* macOS-style title bar with traffic lights + URL bar */}
      <div className="flex items-center gap-3 px-4 py-2.5 border-b border-border bg-[#F7F8FB] dark:bg-[#141820]">
        <div className="flex gap-2 shrink-0">
          <span className="h-3 w-3 rounded-full bg-[#ff5f56]" />
          <span className="h-3 w-3 rounded-full bg-[#ffbd2e]" />
          <span className="h-3 w-3 rounded-full bg-[#27c93f]" />
        </div>
        {/* URL bar */}
        <div className="flex-1 flex justify-center">
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-black/[0.04] dark:bg-white/[0.06] text-[12px] text-muted-foreground select-none">
            <svg width="10" height="10" viewBox="0 0 16 16" fill="none" className="shrink-0 opacity-50">
              <path d="M8 1a4.5 4.5 0 00-4.5 4.5V7H3a1 1 0 00-1 1v6a1 1 0 001 1h10a1 1 0 001-1V8a1 1 0 00-1-1h-.5V5.5A4.5 4.5 0 008 1zm-3 4.5a3 3 0 116 0V7H5V5.5z" fill="currentColor" />
            </svg>
            <span>runmat.com/sandbox</span>
          </div>
        </div>
        {/* Spacer to balance the traffic lights */}
        <div className="w-[52px] shrink-0" />
      </div>

      {/* Editor tab strip */}
      <div className="flex items-center border-b border-border bg-[#F3F4F6] dark:bg-[#12151D]">
        <div className="flex items-center gap-1.5 px-3 py-1.5 text-[12px] bg-[var(--editor-background)] border-r border-border">
          <span className="text-foreground dark:text-[#CDD4E2] font-medium select-none">example.m</span>
        </div>
      </div>

      {/* Editor body with line numbers */}
      <div className="relative group">
        <div
          className="absolute top-2 right-2 z-10"
          onClick={event => event.stopPropagation()}
          onKeyDown={event => event.stopPropagation()}
        >
          <TryInBrowserButton
            code={code}
            size="sm"
            className="bg-[var(--editor-background)]/80 backdrop-blur-sm border-border"
            source="matlab-code-card"
          />
        </div>
        <div
          className="overflow-x-auto py-3"
          style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace" }}
          tabIndex={0}
          onClick={event => event.stopPropagation()}
          onKeyDown={event => event.stopPropagation()}
        >
          {highlightedLines.map((lineHtml, i) => (
            <div key={i} className="flex text-[13px] leading-[21px] hover:bg-white/[0.02]">
              <span
                className="shrink-0 select-none text-right w-10 pr-4 dark:text-[#737C90] text-[#9099AA]"
                aria-hidden="true"
              >
                {i + 1}
              </span>
              <span
                className="flex-1 pr-4 language-matlab"
                dangerouslySetInnerHTML={{ __html: lineHtml }}
                suppressHydrationWarning
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default MatlabCodeCard;
