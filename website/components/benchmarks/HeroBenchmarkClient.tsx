"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";

import type { BenchmarkShowcaseSlide } from "@/lib/marketing-benchmarks";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

import BenchmarkShowcaseCarousel from "./BenchmarkShowcaseCarousel";
import BenchmarkTicker from "./BenchmarkTicker";

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
            <Badge variant="secondary" className="rounded-lg px-1 py-1 text-sm">
              🚀 Open Source • MIT Licensed
            </Badge>
            <h1 className="font-heading text-left leading-tight whitespace-nowrap tracking-tight text-[clamp(2.75rem,5vw,4.75rem)] sm:text-[clamp(3.25rem,4.5vw,5.5rem)] lg:text-[clamp(3.5rem,4vw,6rem)]">
              Fastest runtime for
            </h1>
            <BenchmarkTicker slides={slides} activeIndex={activeIndex} onSelect={goToIndex} />
            <p className="max-w-[42rem] leading-relaxed text-muted-foreground sm:text-xl sm:leading-8">
              We speak GPU so you don&apos;t have to. RunMat automatically fuses operations and intelligently
              routes between CPU/GPU.
              <br />
              MATLAB syntax. No kernel code, no rewrites.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto">
              <Button
                size="lg"
                asChild
                className="h-12 px-8 text-lg font-semibold bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-xl border-0 transition-all duration-200"
              >
                <Link href="/download">Download</Link>
              </Button>
              <Button
                variant="outline"
                size="lg"
                asChild
                className="h-12 px-8 text-lg bg-[#0E1B1E] dark:bg-[#0E1B1E] border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100"
              >
                <Link href="/docs/getting-started">Get Started</Link>
              </Button>
              <Button
                variant="outline"
                size="lg"
                asChild
                className="h-12 px-8 text-lg border-0 shadow-[inset_0_0_0_0.25px_rgb(255,255,255)] dark:shadow-[inset_0_0_0_0.25px_rgb(255,255,255)]"
              >
                <Link href="#benchmarks">Benchmarks</Link>
              </Button>
            </div>
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


