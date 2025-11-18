"use client";

import type { BenchmarkChartData, BenchmarkShowcaseSlide } from "@/lib/marketing-benchmarks";
import { cn } from "@/lib/utils";

import BenchmarkBarChart from "./charts/BenchmarkBarChart";
import BenchmarkLineChart from "./charts/BenchmarkLineChart";

interface BenchmarkShowcaseCarouselProps {
  slides: BenchmarkShowcaseSlide[];
  activeIndex: number;
  onNavigate: (nextIndex: number) => void;
  onHoverChange?: (hovered: boolean) => void;
}

function renderChart(chart: BenchmarkChartData) {
  if (chart.type === "bar") {
    return <BenchmarkBarChart data={chart} />;
  }
  return <BenchmarkLineChart data={chart} />;
}

export function BenchmarkShowcaseCarousel({
  slides,
  activeIndex,
  onNavigate,
  onHoverChange,
}: BenchmarkShowcaseCarouselProps) {
  const activeSlide = slides[activeIndex];
  const hasMultiple = slides.length > 1;

  return (
    <div
      className="relative w-full"
      onMouseEnter={() => onHoverChange?.(true)}
      onMouseLeave={() => onHoverChange?.(false)}
    >
      <div className="rounded-3xl border border-white/10 bg-gradient-to-b from-[#060910] to-[#020305] p-6 shadow-2xl">
        <div className="mb-5 space-y-2 text-center">
          {activeSlide.description && (
            <p className="text-lg font-semibold text-white/90">{activeSlide.description}</p>
          )}
          {activeSlide.deviceLabel && (
            <p className="text-base text-white/75">{activeSlide.deviceLabel}</p>
          )}
          {activeSlide.headlineRange && (
            <p className="text-lg text-white/90">
              
              <span className="gradient-brand font-semibold">RunMat is{" "}{activeSlide.headlineRange}</span>
            </p>
          )}
        </div>
        <div className="w-full">{renderChart(activeSlide.chart)}</div>
      </div>

      {hasMultiple && (
        <div className="mt-4 flex items-center justify-center gap-2">
          {slides.map((slide, idx) => (
            <button
              key={slide.caseId}
              type="button"
              aria-label={`Go to ${slide.label}`}
              onClick={() => onNavigate(idx)}
              className={cn(
                "h-2 rounded-full transition-all",
                idx === activeIndex ? "w-8 bg-white" : "w-3 bg-white/30 hover:bg-white/60"
              )}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default BenchmarkShowcaseCarousel;


