"use client";

import { cn } from "@/lib/utils";
import type { BenchmarkShowcaseSlide } from "@/lib/marketing-benchmarks";

interface BenchmarkTickerProps {
  slides: BenchmarkShowcaseSlide[];
  activeIndex: number;
  onSelect: (index: number) => void;
}

function getVisibleIndexes(length: number, active: number): number[] {
  if (length >= 3) {
    return [-1, 0, 1].map((offset) => (active + offset + length) % length);
  }
  return Array.from({ length }, (_, i) => i);
}

export default function BenchmarkTicker({ slides, activeIndex, onSelect }: BenchmarkTickerProps) {
  if (!slides.length) {
    return null;
  }

  const visible = getVisibleIndexes(slides.length, activeIndex).map((index) => ({
    slide: slides[index],
    index,
    isActive: index === activeIndex,
  }));

  return (
    <div className="flex flex-col space-y-2 w-full">
      {visible.map(({ slide, index, isActive }) => (
        <button
          key={`${slide.caseId}-${index}`}
          type="button"
          onClick={() => onSelect(index)}
          className={cn(
            "w-full rounded-2xl px-4 py-3 text-left transition",
            isActive ? "bg-white/10 shadow-lg" : "bg-transparent"
          )}
        >
          <p
            className={cn(
              "text-2xl font-semibold leading-tight",
              isActive ? "gradient-brand" : "text-white/60"
            )}
          >
            {slide.heroLabel}
          </p>
        </button>
      ))}
    </div>
  );
}


