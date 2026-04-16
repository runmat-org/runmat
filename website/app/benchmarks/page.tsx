import { Metadata } from "next";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ContentCard } from "@/components/content-card";
import { getAllBenchmarks } from "@/lib/benchmarks";

export const metadata: Metadata = {
  title: "RunMat Benchmarks - Performance Comparisons",
  description: "Reproducible, cross-language benchmarks comparing RunMat against common alternatives for representative workloads.",
  openGraph: {
    title: "RunMat Benchmarks - Performance Comparisons",
    description: "Reproducible, cross-language benchmarks comparing RunMat against common alternatives.",
    type: "website",
    url: "https://runmat.com/benchmarks",
  },
  alternates: { canonical: "https://runmat.com/benchmarks" },
};

export default function BenchmarksPage() {
  const benchmarks = getAllBenchmarks();

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-16 md:px-6 md:py-24">
        <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center">
          <h1 className="font-heading text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
            Benchmarks
          </h1>
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {benchmarks.map((benchmark, i) => (
            <ContentCard
              key={encodeURIComponent(benchmark.slug)}
              href={`/benchmarks/${encodeURIComponent(benchmark.slug)}`}
              title={benchmark.title}
              excerpt={benchmark.summary}
              ctaLabel="View Benchmark"
              index={i}
            />
          ))}
        </div>

        {/* Call to Action */}
        <div className="mt-16 max-w-3xl mx-auto">
          <div className="rounded-lg border border-border bg-card px-6 py-8 sm:px-8 sm:py-10 text-center space-y-4">
            <h3 className="text-lg sm:text-xl font-semibold text-foreground">
              Reproduce the benchmarks
            </h3>
            <p className="text-foreground text-sm max-w-md mx-auto">
              Full source code and instructions are in the benchmarks directory of the RunMat repo.
            </p>
            <div className="flex flex-col sm:flex-row gap-3 justify-center pt-1">
              <Button variant="outline" size="lg" asChild className="h-11 px-7 text-sm rounded-none">
                <Link href="https://github.com/runmat-org/runmat/tree/main/benchmarks" target="_blank">
                  View on GitHub
                </Link>
              </Button>
              <Button variant="ghost" size="lg" asChild className="h-11 px-7 text-sm text-muted-foreground hover:text-foreground">
                <Link href="/download">
                  Download RunMat
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
