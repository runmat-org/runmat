"use client";

import { useCallback, useMemo, useState } from "react";
import Link from "next/link";
import { generateSignal, applyWindow, fft, magnitude, nextPow2 } from "./FFTEngine";
import type { SignalComponent, WindowType } from "./FFTEngine";
import { SignalBuilder } from "./SignalBuilder";
import { WindowSelector } from "./WindowSelector";
import { FFTPresetSelector } from "./FFTPresetSelector";
import { TimeDomainChart } from "./TimeDomainChart";
import { SpectrumChart } from "./SpectrumChart";
import { RunInRunMatButton } from "./RunInRunMatButton";

const FFT_SOURCE_URL =
  "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/fft/forward.rs";
const FFT_DOCS_PATH = "/docs/reference/builtins/fft";

const DEFAULT_COMPONENTS: SignalComponent[] = [
  { frequency: 50, amplitude: 1, phase: 0 },
  { frequency: 120, amplitude: 0.5, phase: 0 },
];
const DEFAULT_SAMPLE_RATE = 1024;
const DEFAULT_WINDOW: WindowType = "rectangular";

export function FFTVisualizerClient() {
  const [components, setComponents] = useState<SignalComponent[]>(DEFAULT_COMPONENTS);
  const [sampleRate, setSampleRate] = useState(DEFAULT_SAMPLE_RATE);
  const [windowType, setWindowType] = useState<WindowType>(DEFAULT_WINDOW);
  const [activePresetId, setActivePresetId] = useState<string | null>("two-tone");

  const signal = useMemo(
    () => generateSignal(components, sampleRate, sampleRate),
    [components, sampleRate]
  );

  const windowedSignal = useMemo(
    () => applyWindow(signal, windowType),
    [signal, windowType]
  );

  const spectrum = useMemo(() => {
    const n = nextPow2(windowedSignal.length);
    const re = new Array<number>(n).fill(0);
    const im = new Array<number>(n).fill(0);
    for (let i = 0; i < windowedSignal.length; i++) re[i] = windowedSignal[i];
    const result = fft(re, im);
    return magnitude(result.re, result.im);
  }, [windowedSignal]);

  const handlePreset = useCallback(
    (presetId: string, presetComponents: SignalComponent[], presetSampleRate?: number) => {
      setActivePresetId(presetId);
      setComponents(presetComponents);
      if (presetSampleRate) setSampleRate(presetSampleRate);
    },
    []
  );

  const handleComponentsChange = useCallback((next: SignalComponent[]) => {
    setActivePresetId(null);
    setComponents(next);
  }, []);

  const handleSampleRateChange = useCallback((rate: number) => {
    setActivePresetId(null);
    setSampleRate(rate);
  }, []);

  const componentCount = components.length;
  const peakFreqs = components.map((c) => `${c.frequency} Hz`).join(", ");

  return (
    <div className="rounded-xl border border-border/60 bg-muted/30 p-5 sm:p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary" aria-hidden><path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-4"/></svg>
            Interactive FFT Explorer
          </h2>
          <p className="text-sm text-muted-foreground mt-0.5">Adjust signal parameters and window functions to see the spectrum update in real time.</p>
        </div>
        <span className="hidden sm:inline-flex shrink-0 items-center gap-1.5 rounded-full border border-primary/40 bg-primary/10 px-2.5 py-1 text-xs font-medium text-primary">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60 opacity-75" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
          </span>
          Interactive
        </span>
      </div>

      {/* Presets */}
      <section>
        <div className="mb-2 flex items-baseline justify-between">
          <h3 className="text-sm font-medium text-foreground">Try a preset</h3>
          <span className="text-xs text-muted-foreground">or build your own signal below</span>
        </div>
        <FFTPresetSelector
          activePresetId={activePresetId}
          onSelect={handlePreset}
        />
      </section>

      {/* Charts — the hero */}
      <section className="space-y-2">
        <TimeDomainChart signal={windowedSignal} sampleRate={sampleRate} />
        <p className="px-1 text-xs text-muted-foreground">
          {componentCount === 1
            ? `The signal contains a single sinusoid at ${peakFreqs}. The FFT below reveals it as one peak.`
            : `The signal contains ${componentCount} sinusoids at ${peakFreqs}. The FFT below separates them into distinct peaks.`}
        </p>
        <SpectrumChart spectrum={spectrum} sampleRate={sampleRate} />
      </section>

      {/* Controls — supporting role */}
      <section className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-4 items-start">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <SignalBuilder
            components={components}
            sampleRate={sampleRate}
            onComponentsChange={handleComponentsChange}
            onSampleRateChange={handleSampleRateChange}
          />
          <div className="space-y-4">
            <WindowSelector value={windowType} onChange={setWindowType} />
            <div className="flex flex-col gap-3">
              <RunInRunMatButton
                components={components}
                sampleRate={sampleRate}
                windowType={windowType}
              />
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm">
                <Link
                  href={FFT_SOURCE_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted-foreground hover:text-foreground underline"
                >
                  View fft source on GitHub
                </Link>
                <Link
                  href={FFT_DOCS_PATH}
                  className="text-muted-foreground hover:text-foreground underline"
                >
                  fft in Docs
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
