"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { BenchmarkShowcaseSlide } from "@/lib/marketing-benchmarks";
import BenchmarkShowcaseCarousel from "./BenchmarkShowcaseCarousel";

interface BenchmarkShowcaseBlockClientProps {
  slides: BenchmarkShowcaseSlide[];
}

function usePrefersReducedMotion(): boolean {
  const [prefers, setPrefers] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined") return;
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    setPrefers(media.matches);
    const handler = (event: MediaQueryListEvent) => setPrefers(event.matches);
    media.addEventListener("change", handler);
    return () => media.removeEventListener("change", handler);
  }, []);
  return prefers;
}

export default function BenchmarkShowcaseBlockClient({ slides }: BenchmarkShowcaseBlockClientProps) {
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
    if (!canAutoplay) return;
    intervalRef.current = window.setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % slides.length);
    }, 3000);
    return () => clearAutoplay();
  }, [canAutoplay, clearAutoplay, slides.length]);

  if (!slides.length) return null;

  return (
    <BenchmarkShowcaseCarousel
      slides={slides}
      activeIndex={activeIndex}
      onNavigate={goToIndex}
      onHoverChange={handleHoverChange}
    />
  );
}
