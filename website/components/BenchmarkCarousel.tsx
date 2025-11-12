"use client";

import { useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface BenchmarkCarouselProps {
  svgs: string[];
}

export default function BenchmarkCarousel({ svgs }: BenchmarkCarouselProps) {
  const [currentIndex, setCurrentIndex] = useState(0);

  const goToPrevious = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setCurrentIndex((prev) => (prev === 0 ? svgs.length - 1 : prev - 1));
  };

  const goToNext = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setCurrentIndex((prev) => (prev === svgs.length - 1 ? 0 : prev + 1));
  };

  return (
    <div className="max-w-5xl mx-auto relative w-full">
      <div className="relative overflow-hidden rounded-lg w-full">
        <div className="flex transition-transform duration-500 ease-in-out" style={{ transform: `translateX(-${currentIndex * 100}%)` }}>
          {svgs.map((svg, index) => (
            <div key={index} className="min-w-full flex-shrink-0 flex justify-center items-center">
              {index === 0 ? (
                <a href="https://github.com/runmat-org/runmat/tree/main/benchmarks/4k-image-processing" target="_blank" rel="noopener noreferrer" className="cursor-pointer block flex justify-center items-center min-w-[280px] sm:min-w-[400px]">
                  <div className="[&>svg]:max-w-full [&>svg]:h-auto [&>svg]:w-auto [&>svg]:mx-auto" dangerouslySetInnerHTML={{ __html: svg }} />
                </a>
              ) : (
                <div className="flex justify-center items-center min-w-[280px] sm:min-w-[400px]">
                  <div className="[&>svg]:max-w-full [&>svg]:h-auto [&>svg]:w-auto [&>svg]:mx-auto" dangerouslySetInnerHTML={{ __html: svg }} />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
      
      {/* Navigation Buttons */}
      <Button
        variant="outline"
        size="icon"
        onClick={goToPrevious}
        type="button"
        className={cn(
          "absolute left-24 top-1/2 -translate-y-1/2 z-20 bg-background/80 backdrop-blur-sm hover:bg-background",
          "shadow-lg border-2"
        )}
        aria-label="Previous slide"
      >
        <ChevronLeft className="h-6 w-6" />
      </Button>
      
      <Button
        variant="outline"
        size="icon"
        onClick={goToNext}
        type="button"
        className={cn(
          "absolute right-24 top-1/2 -translate-y-1/2 z-20 bg-background/80 backdrop-blur-sm hover:bg-background",
          "shadow-lg border-2"
        )}
        aria-label="Next slide"
      >
        <ChevronRight className="h-6 w-6" />
      </Button>
    </div>
  );
}

