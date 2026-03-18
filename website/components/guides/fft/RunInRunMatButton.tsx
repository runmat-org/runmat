"use client";

import { useCallback } from "react";
import { openWorkspace } from "@/lib/desktop";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { SignalComponent, WindowType } from "./FFTEngine";

export interface RunInRunMatButtonProps {
  components: SignalComponent[];
  sampleRate: number;
  windowType: WindowType;
  className?: string;
}

function buildMatlabSnippet(
  components: SignalComponent[],
  sampleRate: number,
  windowType: WindowType
): string {
  const lines: string[] = [];
  lines.push(`fs = ${sampleRate};`);
  lines.push(`t = 0:1/fs:1-1/fs;`);

  const signalParts = components.map((c) => {
    const ampStr = c.amplitude === 1 ? "" : `${c.amplitude}*`;
    const phaseStr = c.phase === 0 ? "" : ` + ${c.phase.toFixed(2)}`;
    return `${ampStr}sin(2*pi*${c.frequency}*t${phaseStr})`;
  });

  if (signalParts.length === 1) {
    lines.push(`x = ${signalParts[0]};`);
  } else {
    lines.push(`x = ${signalParts.join(" + ...\n    ")};`);
  }

  if (windowType !== "rectangular") {
    const winMap: Record<string, string> = {
      hann: "hann",
      hamming: "hamming",
      "blackman-harris": "blackmanharris",
    };
    const winFn = winMap[windowType] ?? "hann";
    lines.push(`w = ${winFn}(length(x))';`);
    lines.push(`x = x .* w;`);
  }

  lines.push(`Y = fft(x);`);
  lines.push(`f = (0:length(Y)-1) * fs / length(Y);`);
  lines.push(`plot(f(1:end/2), abs(Y(1:end/2)));`);
  lines.push(`xlabel('Frequency (Hz)');`);
  lines.push(`ylabel('Magnitude');`);

  return lines.join("\n");
}

export function RunInRunMatButton({
  components,
  sampleRate,
  windowType,
  className,
}: RunInRunMatButtonProps) {
  const handleClick = useCallback(() => {
    const code = buildMatlabSnippet(components, sampleRate, windowType);
    openWorkspace(
      [{ path: "/fft_demo.m", content: code }],
      {
        targetPath: "/sandbox",
        metadata: { source: "fft-blog" },
        newTab: true,
      }
    );
  }, [components, sampleRate, windowType]);

  return (
    <Button
      variant="outline"
      size="lg"
      className={cn(
        "inline-flex items-center gap-2 font-semibold border-primary bg-primary/10 text-primary hover:bg-primary/20 hover:border-primary shadow-sm shadow-primary/10 transition-all",
        className
      )}
      data-ph-capture-attribute-destination="sandbox"
      data-ph-capture-attribute-source="fft-blog"
      data-ph-capture-attribute-cta="run-in-browser"
      onClick={handleClick}
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="size-4"
        aria-hidden
      >
        <path d="M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z" />
      </svg>
      Run in RunMat
    </Button>
  );
}
