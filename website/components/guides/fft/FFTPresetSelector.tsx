"use client";

import { useCallback, useMemo } from "react";
import { cn } from "@/lib/utils";
import { generateSignal } from "./FFTEngine";
import type { SignalComponent } from "./FFTEngine";

export interface FFTPreset {
  id: string;
  name: string;
  subtitle: string;
  components: SignalComponent[];
  sampleRate?: number;
}

export const FFT_PRESETS: FFTPreset[] = [
  {
    id: "pure-tone",
    name: "Pure tone",
    subtitle: "One clean peak at 50 Hz",
    components: [{ frequency: 50, amplitude: 1, phase: 0 }],
  },
  {
    id: "two-tone",
    name: "Two-tone",
    subtitle: "Peaks at 50 Hz & 120 Hz",
    components: [
      { frequency: 50, amplitude: 1, phase: 0 },
      { frequency: 120, amplitude: 0.5, phase: 0 },
    ],
  },
  {
    id: "chord",
    name: "Chord",
    subtitle: "Harmonic series: 100, 200, 300 Hz",
    components: [
      { frequency: 100, amplitude: 1, phase: 0 },
      { frequency: 200, amplitude: 0.7, phase: 0 },
      { frequency: 300, amplitude: 0.5, phase: 0 },
    ],
  },
  {
    id: "square-wave",
    name: "Square wave",
    subtitle: "Odd harmonics summed",
    components: [
      { frequency: 50, amplitude: 1, phase: 0 },
      { frequency: 150, amplitude: 0.33, phase: 0 },
      { frequency: 250, amplitude: 0.2, phase: 0 },
      { frequency: 350, amplitude: 0.14, phase: 0 },
    ],
  },
  {
    id: "chirp",
    name: "Chirp",
    subtitle: "Energy spread across the spectrum",
    components: [
      { frequency: 30, amplitude: 0.5, phase: 0 },
      { frequency: 90, amplitude: 0.5, phase: 1.5 },
      { frequency: 200, amplitude: 0.5, phase: 3.0 },
      { frequency: 400, amplitude: 0.5, phase: 4.5 },
    ],
  },
  {
    id: "noisy",
    name: "Noisy signal",
    subtitle: "100 Hz buried in interference",
    components: [
      { frequency: 100, amplitude: 1, phase: 0 },
      { frequency: 37, amplitude: 0.3, phase: 0.8 },
      { frequency: 173, amplitude: 0.25, phase: 2.1 },
      { frequency: 311, amplitude: 0.15, phase: 4.7 },
    ],
  },
];

const SPARKLINE_SAMPLES = 64;
const SPARKLINE_RATE = 512;

function PresetSparkline({ components }: { components: SignalComponent[] }) {
  const path = useMemo(() => {
    const sig = generateSignal(components, SPARKLINE_RATE, SPARKLINE_SAMPLES);
    let min = Infinity;
    let max = -Infinity;
    for (const v of sig) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min || 1;
    const w = 48;
    const h = 20;
    return sig
      .map((v, i) => {
        const x = (i / (SPARKLINE_SAMPLES - 1)) * w;
        const y = h - ((v - min) / range) * (h - 2) - 1;
        return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }, [components]);

  return (
    <svg width={48} height={20} viewBox="0 0 48 20" className="shrink-0 opacity-60 group-hover:opacity-100 transition-opacity" aria-hidden>
      <path d={path} fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export interface FFTPresetSelectorProps {
  activePresetId: string | null;
  onSelect: (presetId: string, components: SignalComponent[], sampleRate?: number) => void;
  className?: string;
}

export function FFTPresetSelector({ activePresetId, onSelect, className }: FFTPresetSelectorProps) {
  const handleClick = useCallback(
    (preset: FFTPreset) => () => {
      onSelect(preset.id, preset.components, preset.sampleRate);
    },
    [onSelect]
  );

  return (
    <div className={cn("grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2", className)}>
      {FFT_PRESETS.map((preset) => {
        const isActive = preset.id === activePresetId;
        return (
          <button
            key={preset.id}
            type="button"
            onClick={handleClick(preset)}
            className={cn(
              "group relative flex flex-col items-start gap-1.5 rounded-lg border p-3 text-left transition-all cursor-pointer shadow-sm hover:shadow-md",
              isActive
                ? "border-primary bg-primary/15 ring-2 ring-primary/40 shadow-primary/10"
                : "border-border/80 bg-card hover:border-primary/60 hover:bg-primary/5"
            )}
          >
            <div className="flex w-full items-center justify-between gap-2">
              <span className={cn(
                "text-sm font-semibold leading-tight",
                isActive ? "text-primary" : "text-foreground"
              )}>
                {preset.name}
              </span>
              <PresetSparkline components={preset.components} />
            </div>
            <span className="text-[10px] leading-tight text-muted-foreground">
              {preset.subtitle}
            </span>
          </button>
        );
      })}
    </div>
  );
}
