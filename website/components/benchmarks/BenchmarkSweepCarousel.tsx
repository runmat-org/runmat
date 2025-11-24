 "use client";

import { useState } from "react";

import FourKImagePipelineSweep from "./FourKImagePipelineSweep";
import MonteCarloSweep from "./MonteCarloSweep";
import ElementwiseMathSweep from "./ElementwiseMathSweep";

const slides = [
  { id: "4k", component: FourKImagePipelineSweep, href: "/benchmarks/4k-image-processing" },
  { id: "monte-carlo", component: MonteCarloSweep, href: "/benchmarks/monte-carlo-analysis" },
  { id: "elementwise", component: ElementwiseMathSweep, href: "/benchmarks/elementwise-math" },
];

export default function BenchmarkSweepCarousel() {
  const [index, setIndex] = useState(0);
  const activeSlide = slides[index];
  const Component = activeSlide.component;

  const goNext = () => setIndex((prev) => (prev + 1) % slides.length);
  const goPrev = () => setIndex((prev) => (prev - 1 + slides.length) % slides.length);

  return (
    <div className="relative flex flex-col items-center gap-4">
      <div className="flex w-full max-w-[48rem] items-center justify-center sm:justify-between gap-2">
        <button
          type="button"
          onClick={goPrev}
          aria-label="Previous benchmark"
          className="hidden sm:inline-flex h-10 w-10 items-center justify-center rounded-full border border-border bg-background/80 text-foreground shadow-sm transition hover:bg-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
        >
          ←
        </button>
        <a className="w-full max-w-[42rem]" href={activeSlide.href}>
          <Component />
        </a>
        <button
          type="button"
          onClick={goNext}
          aria-label="Next benchmark"
          className="hidden sm:inline-flex h-10 w-10 items-center justify-center rounded-full border border-border bg-background/80 text-foreground shadow-sm transition hover:bg-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
        >
          →
        </button>
      </div>
      <div className="flex items-center gap-3">
        {slides.map((slide, idx) => (
          <button
            key={slide.id}
            type="button"
            onClick={() => setIndex(idx)}
            aria-label={`Show ${slide.id} benchmark`}
            className={`h-3 rounded-full border transition-all ${
              idx === index
                ? "w-10 border-white bg-white text-black"
                : "w-4 border-white/60 bg-white/40 hover:bg-white/70"
            }`}
          />
        ))}
      </div>
    </div>
  );
}


