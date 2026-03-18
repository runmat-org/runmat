"use client";

import { useCallback, useId } from "react";
import { cn } from "@/lib/utils";
import type { SignalComponent } from "./FFTEngine";

const FREQ_MIN = 1;
const FREQ_MAX = 500;
const FREQ_STEP = 1;
const AMP_MIN = 0;
const AMP_MAX = 1;
const AMP_STEP = 0.01;
const PHASE_MIN = 0;
const PHASE_MAX = 2 * Math.PI;
const PHASE_STEP = 0.01;
const MAX_COMPONENTS = 4;

const SAMPLE_RATE_OPTIONS = [256, 512, 1024, 2048, 4096] as const;

interface SliderRowProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  unit?: string;
  idPrefix: string;
}

function SliderRow({ label, value, min, max, step, onChange, unit, idPrefix }: SliderRowProps) {
  const inputId = `${idPrefix}-${label}`;
  const display = step >= 1 ? Math.round(value) : value.toFixed(2);

  return (
    <div className="flex items-center gap-3">
      <label htmlFor={inputId} className="w-12 shrink-0 text-xs text-muted-foreground">
        {label}
      </label>
      <input
        type="range"
        id={inputId}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="fft-slider flex-1"
        aria-label={`${label} slider`}
      />
      <span className="w-16 shrink-0 text-right text-xs font-mono text-foreground">
        {display}
        {unit ? ` ${unit}` : ""}
      </span>
    </div>
  );
}

export interface SignalBuilderProps {
  components: SignalComponent[];
  sampleRate: number;
  onComponentsChange: (components: SignalComponent[]) => void;
  onSampleRateChange: (rate: number) => void;
  className?: string;
}

export function SignalBuilder({
  components,
  sampleRate,
  onComponentsChange,
  onSampleRateChange,
  className,
}: SignalBuilderProps) {
  const idPrefix = useId().replace(/:/g, "-");

  const updateComponent = useCallback(
    (idx: number, key: keyof SignalComponent, val: number) => {
      const next = components.map((c, i) => (i === idx ? { ...c, [key]: val } : c));
      onComponentsChange(next);
    },
    [components, onComponentsChange]
  );

  const addComponent = useCallback(() => {
    if (components.length >= MAX_COMPONENTS) return;
    onComponentsChange([...components, { frequency: 100, amplitude: 0.5, phase: 0 }]);
  }, [components, onComponentsChange]);

  const removeComponent = useCallback(
    (idx: number) => {
      if (components.length <= 1) return;
      onComponentsChange(components.filter((_, i) => i !== idx));
    },
    [components, onComponentsChange]
  );

  return (
    <div className={cn("space-y-4", className)}>
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-foreground">Signal components</h2>
        {components.length < MAX_COMPONENTS && (
          <button
            type="button"
            onClick={addComponent}
            className="rounded-full border border-primary/60 bg-primary/10 px-3 py-1.5 text-sm font-medium text-primary transition-colors hover:bg-primary/20 cursor-pointer"
          >
            + Add
          </button>
        )}
      </div>

      {components.map((comp, idx) => (
        <div
          key={idx}
          className="rounded-lg border border-border bg-card p-3 space-y-2"
        >
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-muted-foreground">
              Component {idx + 1}
            </span>
            {components.length > 1 && (
              <button
                type="button"
                onClick={() => removeComponent(idx)}
                className="text-xs text-muted-foreground hover:text-destructive transition-colors"
                aria-label={`Remove component ${idx + 1}`}
              >
                Remove
              </button>
            )}
          </div>
          <SliderRow
            label="Freq"
            value={comp.frequency}
            min={FREQ_MIN}
            max={FREQ_MAX}
            step={FREQ_STEP}
            onChange={(v) => updateComponent(idx, "frequency", v)}
            unit="Hz"
            idPrefix={`${idPrefix}-c${idx}`}
          />
          <SliderRow
            label="Amp"
            value={comp.amplitude}
            min={AMP_MIN}
            max={AMP_MAX}
            step={AMP_STEP}
            onChange={(v) => updateComponent(idx, "amplitude", v)}
            idPrefix={`${idPrefix}-c${idx}`}
          />
          <SliderRow
            label="Phase"
            value={comp.phase}
            min={PHASE_MIN}
            max={PHASE_MAX}
            step={PHASE_STEP}
            onChange={(v) => updateComponent(idx, "phase", v)}
            unit="rad"
            idPrefix={`${idPrefix}-c${idx}`}
          />
        </div>
      ))}

      <div className="rounded-lg border border-border bg-card p-3">
        <label className="text-xs font-medium text-muted-foreground">
          Sample rate / FFT size
        </label>
        <div className="mt-2 flex flex-wrap gap-2">
          {SAMPLE_RATE_OPTIONS.map((rate) => (
            <button
              key={rate}
              type="button"
              onClick={() => onSampleRateChange(rate)}
              className={cn(
                "rounded-full border px-3.5 py-1.5 text-sm font-medium transition-colors cursor-pointer",
                rate === sampleRate
                  ? "border-primary bg-primary/15 text-primary shadow-sm shadow-primary/10"
                  : "border-border/80 bg-muted/50 text-muted-foreground hover:border-primary/50 hover:text-foreground hover:bg-muted/80"
              )}
            >
              {rate}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
