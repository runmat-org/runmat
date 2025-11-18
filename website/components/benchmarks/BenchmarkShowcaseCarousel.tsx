"use client";

import Link from "next/link";
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
      <Link
        href={activeSlide.link ?? "#"}
        target={activeSlide.link ? "_blank" : undefined}
        rel={activeSlide.link ? "noopener noreferrer" : undefined}
        className="group block rounded-3xl border border-white/10 bg-gradient-to-b from-[#060910] to-[#020305] p-6 shadow-2xl transition-transform duration-400 hover:-translate-y-2 hover:shadow-[0_45px_75px_-15px_rgba(0,0,0,0.8)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white/70"
      >
        <div className="mb-5 space-y-1 text-center">
          {activeSlide.description && (
            <p className="text-base text-white/85">{activeSlide.description}</p>
          )}
          {activeSlide.deviceLabel && (
            <p className="text-base text-white/75">{activeSlide.deviceLabel}</p>
          )}
          {activeSlide.headlineRange && (
            <p className="text-lg font-semibold text-white/90">
              <span className="gradient-brand font-semibold">RunMat is {activeSlide.headlineRange}</span>
            </p>
          )}
        </div>
        <div className="w-full">{renderChart(activeSlide.chart)}</div>
        {activeSlide.link && (
          <div className="mt-4 text-right text-base text-white/40 transition-colors duration-300 group-hover:text-white/80">
            View detailed benchmark →
          </div>
        )}
      </Link>

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


