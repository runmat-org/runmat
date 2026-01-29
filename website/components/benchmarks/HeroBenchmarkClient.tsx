"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type { BenchmarkShowcaseSlide } from "@/lib/marketing-benchmarks";

import BenchmarkShowcaseCarousel from "./BenchmarkShowcaseCarousel";
import BenchmarkTicker from "./BenchmarkTicker";
import { HeroTabs } from "@/components/HeroTabs";

interface HeroBenchmarkClientProps {
  slides: BenchmarkShowcaseSlide[];
}

function usePrefersReducedMotion(): boolean {
  const [prefers, setPrefers] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    setPrefers(media.matches);
    const handler = (event: MediaQueryListEvent) => setPrefers(event.matches);
    media.addEventListener("change", handler);
    return () => media.removeEventListener("change", handler);
  }, []);
  return prefers;
}

export default function HeroBenchmarkClient({ slides }: HeroBenchmarkClientProps) {
  const [activeIndex, setActiveIndex] = useState(0);
  const [isHovered, setIsHovered] = useState(false);
  const prefersReducedMotion = usePrefersReducedMotion();
  const intervalRef = useRef<number | null>(null);

  const canAutoplay = !prefersReducedMotion && !isHovered && slides.length > 1;

  const clearAutoplay = useCallback(() => {
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const goToIndex = useCallback(
    (index: number) => {
      if (!slides.length) return;
      const nextIndex = (index + slides.length) % slides.length;
      setActiveIndex(nextIndex);
      // Reset autoplay timer after manual interaction
      if (intervalRef.current !== null) {
        clearAutoplay();
        intervalRef.current = window.setInterval(() => {
          setActiveIndex((prev) => (prev + 1) % slides.length);
        }, 3000);
      }
    },
    [clearAutoplay, slides.length]
  );

  const handleHoverChange = useCallback((hovered: boolean) => {
    setIsHovered(hovered);
  }, []);

  useEffect(() => {
    clearAutoplay();
    if (!canAutoplay) {
      return;
    }
    intervalRef.current = window.setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % slides.length);
    }, 3000);
    return () => clearAutoplay();
  }, [canAutoplay, clearAutoplay, slides.length]);

  if (!slides.length) {
    return null;
  }

  return (
    <section className="w-full py-16 md:py-24 lg:py-24" id="hero">
      <div className="container mx-auto px-4 md:px-6">
        <div className="grid grid-cols-1 gap-10 lg:grid-cols-2 lg:items-center">
          <div className="flex flex-col space-y-6 text-left items-start">
            <div className="mb-2 p-0">
              🚀 Open Source • MIT Licensed • Pre-Release
            </div>
            <h1 className="font-heading text-left leading-tight whitespace-nowrap tracking-tight text-[clamp(2.6rem,4.8vw,4.25rem)] sm:text-[clamp(3rem,4vw,5rem)] lg:text-[clamp(3.25rem,3.6vw,5.25rem)]">
              Fastest runtime for
            </h1>
            <BenchmarkTicker slides={slides} activeIndex={activeIndex} onSelect={goToIndex} />
            <p className="max-w-[42rem] leading-relaxed text-muted-foreground text-base sm:text-lg">
              RunMat automatically fuses operations and intelligently
              routes between CPU/GPU. Write math in MATLAB syntax, and RunMat will run it blazing fast.
            </p>
            <HeroTabs />
          </div>

          <BenchmarkShowcaseCarousel
            slides={slides}
            activeIndex={activeIndex}
            onNavigate={goToIndex}
            onHoverChange={handleHoverChange}
          />
        </div>
      </div>
    </section>
  );
}


