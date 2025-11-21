import { Suspense } from "react";

import { getBenchmarkShowcaseSlides } from "@/lib/marketing-benchmarks";

import HeroBenchmarkClient from "./HeroBenchmarkClient";

async function HeroBenchmarkShowcaseInner() {
  const slides = await getBenchmarkShowcaseSlides();
  if (!slides.length) {
    return null;
  }
  return <HeroBenchmarkClient slides={slides} />;
}

export default function HeroBenchmarkShowcase() {
  return (
    <Suspense fallback={null}>
      <HeroBenchmarkShowcaseInner />
    </Suspense>
  );
}


