"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import type { BenchmarkChartData, BenchmarkShowcaseSlide } from "@/lib/marketing-benchmarks";
import { cn } from "@/lib/utils";

const ChartFallback = () => <div className="w-full" style={{ height: 320 }} />;

const BenchmarkBarChart = dynamic(() => import("./charts/BenchmarkBarChart"), {
  ssr: false,
  loading: ChartFallback,
});
const BenchmarkLineChart = dynamic(() => import("./charts/BenchmarkLineChart"), {
  ssr: false,
  loading: ChartFallback,
});

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
        className="group block focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
      >
        <div className="relative rounded-lg border border-border bg-card p-6">
          <div className="relative z-10">
            <div className="mb-5 space-y-1 text-center">
              {activeSlide.description && (
                <p className="text-base text-muted-foreground">{activeSlide.description}</p>
              )}
              {activeSlide.headlineText && (
                <p className="font-semibold text-foreground text-[clamp(1rem,2.6vw,1.35rem)]">
                  <span className="text-[hsl(var(--brand))] font-semibold">
                    {activeSlide.headlineText}
                  </span>
                </p>
              )}
            </div>
            <div className="w-full min-h-[320px] sm:min-h-[360px]">{renderChart(activeSlide.chart)}</div>
            {(activeSlide.deviceLabel || activeSlide.link) && (
              <div className="mt-4 flex flex-col gap-2 text-base sm:flex-row sm:items-center sm:justify-between">
                {activeSlide.deviceLabel && (
                  <p className="text-left text-muted-foreground">Measured on {activeSlide.deviceLabel}</p>
                )}
                {activeSlide.link && (
                  <div className="text-left text-muted-foreground/60 transition-colors duration-300 group-hover:text-foreground sm:text-right">
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
                idx === activeIndex ? "w-8 bg-foreground" : "w-3 bg-foreground/30 hover:bg-foreground/60"
              )}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default BenchmarkShowcaseCarousel;


