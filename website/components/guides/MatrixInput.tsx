"use client";

import { useCallback, useId } from "react";
import { cn } from "@/lib/utils";

export interface MatrixValues {
  a: number;
  b: number;
  c: number;
  d: number;
}

const SLIDER_MIN = -10;
const SLIDER_MAX = 10;
const SLIDER_STEP = 0.1;

interface CellProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  idPrefix: string;
}

function Cell({ label, value, onChange, idPrefix }: CellProps) {
  const inputId = `${idPrefix}-${label}`;
  const num = Number.isFinite(value) ? value : 0;
  const clamped = Math.max(SLIDER_MIN, Math.min(SLIDER_MAX, num));

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const v = parseFloat(e.target.value);
      if (!Number.isNaN(v)) onChange(v);
    },
    [onChange]
  );

  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange(parseFloat(e.target.value));
    },
    [onChange]
  );

  return (
    <div className="space-y-1">
      <label htmlFor={inputId} className="text-xs text-muted-foreground">
        {label}
      </label>
      <input
        id={inputId}
        type="number"
        step={SLIDER_STEP}
        min={SLIDER_MIN}
        max={SLIDER_MAX}
        value={clamped}
        onChange={handleInputChange}
        className="w-full rounded border border-input bg-background px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
      />
      <input
        type="range"
        min={SLIDER_MIN}
        max={SLIDER_MAX}
        step={SLIDER_STEP}
        value={clamped}
        onChange={handleSliderChange}
        className="w-full accent-primary"
        aria-label={`Slider for ${label}`}
      />
    </div>
  );
}

export interface MatrixInputProps {
  value: MatrixValues;
  onChange: (v: MatrixValues) => void;
  className?: string;
}

export function MatrixInput({ value, onChange, className }: MatrixInputProps) {
  const idPrefix = useId().replace(/:/g, "-");

  const update = useCallback(
    (key: keyof MatrixValues) => (v: number) => {
      onChange({ ...value, [key]: v });
    },
    [value, onChange]
  );

  return (
    <div className={cn("space-y-4", className)}>
      <div className="grid grid-cols-2 gap-3">
        <Cell label="a" value={value.a} onChange={update("a")} idPrefix={idPrefix} />
        <Cell label="b" value={value.b} onChange={update("b")} idPrefix={idPrefix} />
        <Cell label="c" value={value.c} onChange={update("c")} idPrefix={idPrefix} />
        <Cell label="d" value={value.d} onChange={update("d")} idPrefix={idPrefix} />
      </div>
    </div>
  );
}
