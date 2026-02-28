import { Metadata } from "next";
import { Card, CardContent } from "@/components/ui/card";
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
        <div className="mx-auto max-w-[58rem] text-center">
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl">
            RunMat Benchmarks
          </h1>
          <p className="mt-6 text-base text-muted-foreground sm:text-lg">
            Reproducible, cross-language benchmarks comparing RunMat against common alternatives for representative workloads
          </p>
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {benchmarks.map((benchmark) => (
            <ContentCard
              key={encodeURIComponent(benchmark.slug)}
              href={`/benchmarks/${encodeURIComponent(benchmark.slug)}`}
              title={benchmark.title}
              image={benchmark.imageUrl}
              imageAlt={benchmark.title}
              excerpt={benchmark.summary}
              ctaLabel="View Benchmark"
            />
          ))}
        </div>

        {/* Call to Action */}
        <div className="mt-16 text-center">
          <Card className="inline-block">
            <CardContent className="p-6">
              <h3 className="text-2xl font-semibold mb-2 sm:text-3xl">
                Reproduce the Benchmarks
              </h3>
              <p className="text-muted-foreground mb-4">
                See the benchmarks directory in the RunMat repo for full source code and instructions
              </p>
              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <Button variant="outline" size="sm" asChild>
                  <Link href="https://github.com/runmat-org/runmat/tree/main/benchmarks" target="_blank">
                    View on GitHub
                  </Link>
                </Button>
                <Button variant="outline" size="sm" asChild>
                  <Link href="/download">
                    Download RunMat
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
