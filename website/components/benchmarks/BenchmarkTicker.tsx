"use client";

import { motion } from "motion/react";
import { cn } from "@/lib/utils";
import type { BenchmarkShowcaseSlide } from "@/lib/marketing-benchmarks";

interface BenchmarkTickerProps {
  slides: BenchmarkShowcaseSlide[];
  activeIndex: number;
  onSelect: (index: number) => void;
}

function getVisibleIndexes(length: number, active: number): number[] {
  if (length >= 3) {
    return [1, 0, -1].map((offset) => (active + offset + length) % length);
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
    <div className="relative flex w-full flex-col space-y-3 overflow-hidden">
      <div className="pointer-events-none absolute inset-x-0 top-1/2 z-0 -translate-y-1/2 px-1">
        <div className="h-14 rounded-2xl bg-white/10 blur-[1px]" />
      </div>
      {visible.map(({ slide, index, isActive }) => (
        <motion.div
          key={`${slide.caseId}-${index}`}
          layout
          transition={{ type: "spring", stiffness: 400, damping: 40 }}
          className="relative z-10"
        >
          <button
            type="button"
            onClick={() => onSelect(index)}
            className={cn(
              "w-full rounded-2xl px-6 py-3 text-left transition-all duration-300",
              isActive ? "scale-[1.02]" : "scale-100"
            )}
          >
            <p
              className={cn(
                "font-semibold leading-tight transition-colors duration-200",
                isActive ? "text-2xl gradient-brand" : "text-lg text-white/60"
              )}
            >
              {slide.heroLabel}
            </p>
          </button>
        </motion.div>
      ))}
    </div>
  );
}


