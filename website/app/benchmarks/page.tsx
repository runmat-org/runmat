import { Metadata } from "next";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
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
            <Link key={encodeURIComponent(benchmark.slug)} href={`/benchmarks/${encodeURIComponent(benchmark.slug)}`} className="block">
              <Card className="group overflow-hidden transition-all hover:shadow-lg cursor-pointer h-full flex flex-col">
                <CardContent className="p-6 flex flex-col h-full">
                  {/* Thumbnail */}
                  <div
                    className={`w-full h-48 rounded-lg mb-4 transition-all ${
                      benchmark.imageUrl
                        ? 'bg-muted/10'
                        : 'bg-gradient-to-br from-purple-500 via-purple-600 to-blue-600 group-hover:from-purple-600 group-hover:via-purple-700 group-hover:to-blue-700'
                    }`}
                    style={
                      benchmark.imageUrl
                        ? {
                            backgroundImage: `url(${benchmark.imageUrl})`,
                            backgroundSize: 'contain',
                            backgroundPosition: 'center',
                            backgroundRepeat: 'no-repeat',
                          }
                        : undefined
                    }
                  />

                  {/* Title */}
                  <h3 className="text-2xl font-semibold leading-tight sm:text-3xl mb-4">
                    {benchmark.title}
                  </h3>
                    
                  {/* Description */}
                  <p className="text-sm text-muted-foreground mb-4 flex-grow line-clamp-3">
                    {benchmark.summary}
                  </p>
                    
                  {/* View Benchmark Link */}
                  <div className="flex items-center text-sm text-foreground group-hover:text-primary transition-colors">
                    View Benchmark
                    <svg 
                      className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" 
                      fill="none" 
                      stroke="currentColor" 
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                </CardContent>
              </Card>
            </Link>
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
