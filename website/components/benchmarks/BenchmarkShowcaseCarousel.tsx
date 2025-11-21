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
        className="group block focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white/70"
      >
        <div className="relative rounded-3xl border border-white/10 bg-gradient-to-b from-[#0b1427] via-[#070d1a] to-[#030508] p-6 shadow-[0_25px_70px_rgba(0,0,0,0.65)] transition-transform duration-400 hover:-translate-y-2 hover:shadow-[0_45px_90px_rgba(0,0,0,0.8)]">
          <span
            aria-hidden
            className="pointer-events-none absolute inset-x-12 -bottom-8 h-16 rounded-full bg-black/60 opacity-60 blur-3xl transition-opacity duration-300 group-hover:opacity-90"
          />
          <div className="relative z-10">
            <div className="mb-5 space-y-1 text-center">
              {activeSlide.description && (
                <p className="text-base text-white/85">{activeSlide.description}</p>
              )}
              {activeSlide.headlineText && (
                <p className="font-semibold text-white/90 text-[clamp(1rem,2.6vw,1.35rem)]">
                  <span className="gradient-brand font-semibold">
                    {activeSlide.headlineText}
                  </span>
                </p>
              )}
            </div>
            <div className="w-full min-h-[320px] sm:min-h-[360px]">{renderChart(activeSlide.chart)}</div>
            {(activeSlide.deviceLabel || activeSlide.link) && (
              <div className="mt-4 flex flex-col gap-2 text-base sm:flex-row sm:items-center sm:justify-between">
                {activeSlide.deviceLabel && (
                  <p className="text-left text-white/60">Measured on {activeSlide.deviceLabel}</p>
                )}
                {activeSlide.link && (
                  <div className="text-left text-white/40 transition-colors duration-300 group-hover:text-white/80 sm:text-right">
                    View detailed benchmark →
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </Link>

      {hasMultiple && (
        <div className="mt-4 flex items-center justify-center gap-2">
          {slides.map((slide, idx) => (
            <button
              key={`${slide.caseId}-${idx}`}
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


