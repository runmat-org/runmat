import { Suspense } from "react";
import { getBenchmarkShowcaseSlides } from "@/lib/marketing-benchmarks";
import BenchmarkShowcaseBlockClient from "./BenchmarkShowcaseBlockClient";

async function BenchmarkShowcaseBlockInner() {
  const slides = await getBenchmarkShowcaseSlides();
  if (!slides.length) return null;
  return <BenchmarkShowcaseBlockClient slides={slides} />;
}

export default function BenchmarkShowcaseBlock() {
  return (
    <Suspense fallback={null}>
      <BenchmarkShowcaseBlockInner />
    </Suspense>
  );
}
