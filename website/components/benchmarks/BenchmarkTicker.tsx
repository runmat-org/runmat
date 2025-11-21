"use client";

import { motion } from "motion/react";
import { cn } from "@/lib/utils";
import type { BenchmarkShowcaseSlide } from "@/lib/marketing-benchmarks";

interface BenchmarkTickerProps {
  slides: BenchmarkShowcaseSlide[];
  activeIndex: number;
  onSelect: (index: number) => void;
}

interface VisibleSlide {
  index: number;
  offset: number;
}

function getVisibleSlides(length: number, active: number): VisibleSlide[] {
  if (length >= 3) {
    return [1, 0, -1].map((offset) => ({
      offset,
      index: (active + offset + length) % length,
    }));
  }

  return Array.from({ length }, (_, index) => {
    let offset = index - active;
    if (offset > length / 2) {
      offset -= length;
    } else if (offset < -length / 2) {
      offset += length;
    }
    return { index, offset };
  });
}

function getDepthStyles(offset: number) {
  if (offset === 0) {
    return { scale: 1, opacity: 1, zIndex: 30 };
  }

  if (offset > 0) {
    // Item rotating toward the back of the wheel
    return { scale: 0.82, opacity: 0.35, zIndex: 5 };
  }

  // Item emerging toward the front
  return { scale: 0.9, opacity: 0.6, zIndex: 15 };
}

export default function BenchmarkTicker({ slides, activeIndex, onSelect }: BenchmarkTickerProps) {
  if (!slides.length) {
    return null;
  }

  const visible = getVisibleSlides(slides.length, activeIndex).map(({ index, offset }) => ({
    slide: slides[index],
    index,
    offset,
    isActive: offset === 0,
  }));

  return (
    <div className="relative flex w-full flex-col space-y-3 overflow-hidden">
      <div className="pointer-events-none absolute inset-x-0 top-1/2 z-0 -translate-y-1/2 px-1">
        <div className="h-14 rounded-2xl bg-white/10 blur-[1px]" />
      </div>
      {visible.map(({ slide, index, offset, isActive }) => {
        const depth = getDepthStyles(offset);
        return (
        <motion.div
          key={`${slide.caseId}-${index}`}
          layout
            animate={{ scale: depth.scale, opacity: depth.opacity }}
            style={{ zIndex: depth.zIndex }}
            transition={{
              layout: { type: "spring", stiffness: 400, damping: 40 },
              scale: { duration: 0.3, ease: "easeOut" },
              opacity: { duration: 0.3, ease: "easeOut" },
            }}
          className="relative z-10"
        >
          <button
            type="button"
            onClick={() => onSelect(index)}
            className={cn(
              "w-full rounded-2xl px-6 py-3 text-left transition-all duration-300",
              isActive ? "bg-transparent" : "bg-transparent"
            )}
          >
            <p
              className={cn(
                "font-semibold leading-tight transition-colors duration-200",
                isActive ? "text-2xl gradient-brand" : "text-lg text-muted-foreground"
              )}
            >
              {slide.heroLabel}
            </p>
          </button>
        </motion.div>
        );
      })}
    </div>
  );
}


