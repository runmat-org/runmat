import type { Metadata } from "next";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

import { OSInstallCommand } from "@/components/OSInstallCommand";
import Link from "next/link";

export const metadata: Metadata = {
  title: "RunMat - Fast, Free, Modern MATLAB Runtime",
  description:
    "Run MATLAB code fast and free. RunMat implements full language grammar and core semantics with a slim, portable core, V8-inspired tiered execution, and a package-first standard library.",
  keywords: [
    "run matlab online",
    "free matlab runtime",
    "matlab alternative",
    "octave alternative",
    "run matlab code",
    "matlab replacement",
    "gnu octave vs",
    "high performance matlab",
    "matlab jit",
    "jupyter matlab",
    "matlab jupyter kernel",
    "jupyter matlab integration",
    "matlab plotting",
    "matlab vs octave",
    "matlab vs python",
    "matlab vs julia",
    "matlab vs scilab",
    "matlab blas lapack",
  ],
  alternates: { canonical: "/" },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  openGraph: {
    title: "RunMat - Free, Modern MATLAB Runtime",
    description:
      "Run MATLAB code for free with a modern, high-performance runtime. Jupyter kernel, BLAS/LAPACK, beautiful plots, JIT compilation, and open source.",
    url: "/",
    siteName: "RunMat",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "RunMat - Free, Modern MATLAB Runtime",
    description:
      "Run MATLAB code for free with a fast, open-source alternative. Jupyter kernel, BLAS/LAPACK, beautiful plots, and JIT compilation.",
  },
};

export default function HomePage() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* SEO-optimized opening content */}
      <div className="sr-only">
        <h1>RunMat - Fast, Free, Modern MATLAB Runtime</h1>
        <p>
          Run MATLAB code for free with RunMat: high-performance, open-source runtime with
          Jupyter kernel, BLAS/LAPACK, beautiful plotting, and JIT compilation. Replacement for
          MATLAB and GNU Octave with dramatically faster performance.
        </p>
      </div>

      {/* Hero Section */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="flex flex-col items-center space-y-6 text-center">
          <Badge variant="secondary" className="rounded-lg px-3 py-1 text-sm">
            🚀 Open Source • MIT Licensed • Free Forever
          </Badge>
          <h1 className="font-heading text-3xl sm:text-5xl md:text-6xl lg:text-7xl text-center">
              <span className="gradient-brand">The Fast, Free, Modern<br />MATLAB Runtime</span>
          </h1>
          <p className="max-w-[42rem] leading-relaxed text-muted-foreground sm:text-xl sm:leading-8">
              A free, high-performance runtime for MATLAB code with full language grammar and core semantics.
              No license fees, no lock-in — just a blazing-fast, slim, modern engine.
          </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <Button size="lg" asChild className="h-12 px-8 text-lg font-semibold bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-xl border-0 transition-all duration-200 hover:scale-105 hover:shadow-2xl">
                <Link href="/download">
                  Download RunMat
                </Link>
              </Button>
              <Button variant="outline" size="lg" asChild className="h-12 px-8">
                <Link href="/docs/getting-started">Get Started</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="w-full py-16 md:py-24 lg:py-32 bg-muted/50">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-heading text-3xl leading-[1.1] sm:text-4xl md:text-5xl lg:text-6xl">
              Why Use RunMat?
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-muted-foreground sm:text-xl sm:leading-8">
              Run your existing MATLAB/Octave code faster, and for free.
            </p>
        </div>
          <div className="mx-auto grid justify-center gap-4 sm:grid-cols-2 lg:grid-cols-4 md:max-w-[80rem]">
            <Card>
              <CardHeader>
                <div className="mb-2 text-3xl">✅</div>
                <CardTitle>Full Language Semantics</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Parser + core semantics for arrays and indexing (<code>end</code>, colon, masks), multiple returns,
                  cells/structs, <code>classdef</code> OOP, and more. See the <Link className="underline" href="/docs/language-coverage">language coverage</Link> with Octave comparison.
                </CardDescription>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <div className="mb-2 text-3xl">⚡</div>
                <CardTitle>150×–180× Faster</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Benchmarks on Apple M2 Max show triple‑digit speedups vs GNU Octave across startup, matrix ops,
                  math functions, and control flow. Read the <Link className="underline" href="/blog/introducing-runmat">results</Link>.
                </CardDescription>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <div className="mb-2 text-3xl">📦</div>
                <CardTitle>Slim Core + Packages</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  A minimal, blazing‑fast core with canonical builtins. Breadth comes from packages (native Rust or
                  source MATLAB). See the <Link className="underline" href="/docs/package-manager">package manager (draft)</Link>.
                </CardDescription>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <div className="mb-2 text-3xl">🧱</div>
                <CardTitle>Portable & Lightweight</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Typed HIR lowers to Cranelift IR for small static binaries and predictable performance across
                  Linux/macOS/Windows. Great for laptops, clusters, and CI.
                </CardDescription>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Code Example Section */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
          <h2 className="font-heading text-3xl leading-[1.1] sm:text-3xl md:text-6xl">
              Same Language, Better Engine
          </h2>
          <p className="max-w-[42rem] leading-relaxed text-muted-foreground sm:text-xl sm:leading-8">
              Boots in 5ms, runs 150x-180x faster than GNU Octave, GPU optimizes by default, and is free to use forever.
          </p>
          </div>

          <div className="mx-auto max-w-7xl">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
              {/* Terminal/Shell on the left */}
              <div className="order-1 lg:order-1">
                <Card className="bg-gray-900 dark:bg-gray-900 bg-gray-100 border-gray-700 dark:border-gray-700 border-gray-300 shadow-2xl py-0">
                  {/* Terminal Header */}
                  <div className="flex items-center justify-between px-4 py-3 bg-gray-800 dark:bg-gray-800 bg-gray-200 rounded-t-lg border-b border-gray-700 dark:border-gray-700 border-b-gray-300">
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    </div>
                    <div className="w-12"></div>
                  </div>

                  {/* Terminal Content */}
                  <CardContent className="px-6 pt-4 pb-6 font-mono text-sm">
                    <div className="space-y-2 text-white dark:text-white text-gray-900">
                      {/* Initial command */}
                      <div className="flex">
                        <span className="text-gray-500 dark:text-gray-500 text-gray-600 dark:text-gray-500 dark:text-gray-500 text-gray-600 text-gray-600">$</span>
                        <span className="ml-2">runmat</span>
                      </div>

                      {/* RunMat header */}
                      <div className="text-cyan-400 dark:text-cyan-400 text-cyan-600">RunMat v0.1.0 by Dystr (https://dystr.com)</div>
                      <div className="text-cyan-400 dark:text-cyan-400 text-cyan-600">Fast, free, modern MATLAB runtime with JIT compilation and GC</div>
                      <div className="text-cyan-400 dark:text-cyan-400 text-cyan-600">Type &apos;help&apos; for help, &apos;exit&apos; to quit</div>
                      <div className="h-4"></div>

                      {/* Interactive session */}
                      <div className="space-y-5">
                        {/* Generate data section */}
                        <div className="space-y-1">
                          <div className="text-cyan-300 dark:text-cyan-300 text-cyan-700 mb-2">% Generate data</div>
                          <div>
                            <div className="flex">
                              <span className="text-blue-400 dark:text-blue-400 text-blue-600">runmat&gt;</span>
                              <span className="ml-2">x = linspace(0, 4 * pi, 1000);</span>
                            </div>
                            <div className="text-gray-500 dark:text-gray-500 text-gray-600">ans = Num(1000x1 vector)</div>
                          </div>
                          <div>
                            <div className="flex">
                              <span className="text-blue-400 dark:text-blue-400 text-blue-600">runmat&gt;</span>
                              <span className="ml-2">y = sin(x) .* exp(-x/10);</span>
                            </div>
                            <div className="text-gray-500 dark:text-gray-500 text-gray-600">ans = Num(1000x1 vector)</div>
                          </div>
                        </div>

                        {/* Create plot section */}
                        <div className="space-y-1">
                          <div className="text-cyan-300 dark:text-cyan-300 text-cyan-700 mb-2">% Create beautiful plot</div>
                          <div>
                            <div className="flex">
                              <span className="text-blue-400 dark:text-blue-400 text-blue-600">runmat&gt;</span>
                              <span className="ml-2">plot(x, y);</span>
                            </div>
                            <div className="text-gray-500 dark:text-gray-500 text-gray-600">[Interactive plot window opened]</div>
                          </div>
                        </div>

                        {/* Matrix operations section */}
                        <div className="space-y-1">
                          <div className="text-cyan-300 dark:text-cyan-300 text-cyan-700 mb-2">% Matrix operations</div>
                          <div>
                            <div className="flex">
                              <span className="text-blue-400 dark:text-blue-400 text-blue-600">runmat&gt;</span>
                              <span className="ml-2">A = randn(1000, 1000);</span>
                            </div>
                            <div className="text-gray-500 dark:text-gray-500 text-gray-600">ans = Matrix(1000x1000 double)</div>
                          </div>
                          <div>
                            <div className="flex">
                              <span className="text-blue-400 dark:text-blue-400 text-blue-600">runmat&gt;</span>
                              <span className="ml-2">B = A * A&apos;;</span>
                            </div>
                            <div className="text-gray-500 dark:text-gray-500 text-gray-600">ans = Matrix(1000x1000 double)</div>
                          </div>
                          <div>
                            <div className="flex">
                              <span className="text-blue-400 dark:text-blue-400 text-blue-600">runmat&gt;</span>
                              <span className="ml-2">eigenvals = eig(B);</span>
                            </div>
                            <div className="text-gray-500 dark:text-gray-500 text-gray-600">ans = Num(1000x1 complex vector)</div>
                          </div>
                        </div>

                        {/* Cursor */}
                        <div className="flex mt-3">
                          <span className="text-blue-400 dark:text-blue-400 text-blue-600">runmat&gt;</span>
                          <span className="ml-2 animate-pulse">▊</span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* RunMat Plot Window */}
              <div className="order-2 lg:order-2">
                <div className="bg-gray-700 rounded-lg shadow-2xl overflow-hidden">
                  {/* macOS Window Header */}
                  <div className="bg-gray-600 px-4 py-3 flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    </div>
                    <div className="text-gray-200 text-sm font-medium">RunMat - Interactive Visualization</div>
                    <div className="w-16"></div>
                  </div>

                  {/* Window Content Area */}
                  <div className="overflow-hidden">
                    <img
                      src="/plot-example.jpg"
                      alt="RunMat interactive plot showing a damped sine wave with real-time visualization controls"
                      className="w-full h-auto"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Language Semantics Section */}
      <section className="w-full py-16 md:py-24 lg:py-32 bg-muted/50">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-6 text-center mb-8">
            <h2 className="font-heading text-3xl leading-[1.1] sm:text-4xl md:text-5xl">Full Language Semantics</h2>
            <p className="max-w-[42rem] leading-relaxed text-muted-foreground sm:text-xl sm:leading-8">
              RunMat implements the full grammar and core semantics of the MATLAB language, rather than a subset. See
              <Link className="underline ml-1" href="/docs/language-coverage">language coverage</Link>.
            </p>
          </div>
          <div className="overflow-x-auto mx-auto max-w-[48rem]">
            <table className="w-full text-base md:text-lg">
              <thead>
                <tr className="text-left border-b border-border">
                  <th className="py-3 px-4">Feature Category</th>
                  <th className="py-3 px-4 text-center">RunMat</th>
                  <th className="py-3 px-4 text-center">Octave</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-border/60"><td className="py-3 px-4">Grammar & parser (full surface)</td><td className="py-3 px-4 text-center text-green-600">✅</td><td className="py-3 px-4 text-center">✅</td></tr>
                <tr className="border-b border-border/60"><td className="py-3 px-4">Arrays & indexing (end, colon, masks, N‑D)</td><td className="py-3 px-4 text-center text-green-600">✅</td><td className="py-3 px-4 text-center">✅</td></tr>
                <tr className="border-b border-border/60"><td className="py-3 px-4">Multiple returns, varargin/varargout, nargin/nargout</td><td className="py-3 px-4 text-center text-green-600">✅</td><td className="py-3 px-4 text-center">✅</td></tr>
                <tr className="border-b border-border/60"><td className="py-3 px-4">classdef OOP + operator overloading</td><td className="py-3 px-4 text-center text-green-600">✅</td><td className="py-3 px-4 text-center text-red-500">❌</td></tr>
                <tr className="border-b border-border/60"><td className="py-3 px-4">Events/handles (addlistener/notify/isvalid/delete)</td><td className="py-3 px-4 text-center text-green-600">✅</td><td className="py-3 px-4 text-center text-red-500">❌</td></tr>
                <tr className="border-b border-border/60"><td className="py-3 px-4">Imports precedence & static access (Class.*)</td><td className="py-3 px-4 text-center text-green-600">✅</td><td className="py-3 px-4 text-center text-red-500">❌</td></tr>
                <tr className="border-b border-border/60"><td className="py-3 px-4">Metaclass operator ?Class</td><td className="py-3 px-4 text-center text-green-600">✅</td><td className="py-3 px-4 text-center text-red-500">❌</td></tr>
                <tr className="border-b border-border/60"><td className="py-3 px-4">String arrays (double‑quoted)</td><td className="py-3 px-4 text-center text-green-600">✅</td><td className="py-3 px-4 text-center text-red-500">❌</td></tr>
                <tr><td className="py-3 px-4">Standardized MException identifiers</td><td className="py-3 px-4 text-center text-green-600">✅</td><td className="py-3 px-4 text-center text-red-500">❌</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Benchmarks Section */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-6 text-center mb-8">
            <h2 className="font-heading text-3xl leading-[1.1] sm:text-4xl md:text-5xl">Blazing Fast Performance</h2>
            <p className="max-w-[42rem] leading-relaxed text-muted-foreground sm:text-xl sm:leading-8">
              Summary from our public benchmarks (Apple M2 Max, 32GB). Reproduce with <code>benchmarks</code> in the repo.
            </p>
          </div>
          <div className="overflow-x-auto mx-auto max-w-[48rem]">
            <table className="w-full text-base md:text-lg">
              <thead>
                <tr className="text-left border-b border-border">
                  <th className="py-3 px-4">Benchmark</th>
                  <th className="py-3 px-4 text-right">GNU Octave avg (s)</th>
                  <th className="py-3 px-4 text-right">RunMat interp (s)</th>
                  <th className="py-3 px-4 text-right">RunMat JIT (s)</th>
                  <th className="py-3 px-4 text-right">Speedup</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-border/60"><td className="py-3 px-4">Startup Time</td><td className="py-3 px-4 text-right">0.9147</td><td className="py-3 px-4 text-right">0.0050</td><td className="py-3 px-4 text-right">0.0053</td><td className="py-3 px-4 text-right"><span className="text-green-600 font-semibold">171.5×–182.9×</span></td></tr>
                <tr className="border-b border-border/60"><td className="py-3 px-4">Matrix Operations</td><td className="py-3 px-4 text-right">0.8220</td><td className="py-3 px-4 text-right">0.0050</td><td className="py-3 px-4 text-right">0.0050</td><td className="py-3 px-4 text-right"><span className="text-green-600 font-semibold">164.4×</span></td></tr>
                <tr className="border-b border-border/60"><td className="py-3 px-4">Mathematical Functions</td><td className="py-3 px-4 text-right">0.8677</td><td className="py-3 px-4 text-right">0.0057</td><td className="py-3 px-4 text-right">0.0053</td><td className="py-3 px-4 text-right"><span className="text-green-600 font-semibold">153.1×–162.7×</span></td></tr>
                <tr><td className="py-3 px-4">Control Flow</td><td className="py-3 px-4 text-right">0.8757</td><td className="py-3 px-4 text-right">0.0057</td><td className="py-3 px-4 text-right">0.0057</td><td className="py-3 px-4 text-right"><span className="text-green-600 font-semibold">154.5×</span></td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Slim Core + Packages Section */}
      <section className="w-full py-16 md:py-24 lg:py-32 bg-muted/50">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto grid justify-center gap-6 md:grid-cols-2 md:max-w-[64rem]">
            <Card>
              <CardHeader>
                <CardTitle className="text-xl">Slim Core + Packages</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-sm">
                  The runtime ships a minimal, blazing‑fast core with canonical builtins (e.g., sin/cos/sum; printf‑style
                  formatting). Broader or domain‑specific behavior lives in packages: native (Rust) for speed, or source (MATLAB)
                  for portability. Docs are generated from runtime metadata.
                </CardDescription>
                <div className="mt-4 text-sm"><Link className="underline" href="/docs/package-manager">Read the Package Manager design draft →</Link></div>
                <div className="mt-1 text-sm"><Link className="underline" href="/docs/design-philosophy">Read the Design Philosophy →</Link></div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-xl">Portable & Lightweight</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-sm">
                  Typed HIR lowers to Cranelift IR, yielding small static binaries and predictable performance across
                  Linux/macOS/Windows. Great for laptops, clusters, CI — and for accelerators/LSPs that plug in cleanly.
                </CardDescription>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Quick Install Section */}
      <section className="w-full py-16 md:py-24 lg:py-32 bg-muted/30">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center">
          <h2 className="font-heading text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
            Install in Seconds
          </h2>
          <p className="max-w-[42rem] leading-relaxed text-muted-foreground sm:text-xl sm:leading-8">
            Copy and paste one command to get started with RunMat
          </p>
          
          <OSInstallCommand className="w-full max-w-4xl" />
          
            <div className="flex flex-col sm:flex-row gap-4">
              <Button size="lg" asChild className="h-12 px-8 text-lg font-semibold bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-xl border-0 transition-all duration-200 hover:scale-105 hover:shadow-2xl">
                <Link href="/download">More Install Options</Link>
            </Button>
              <Button variant="outline" size="lg" asChild className="h-12 px-8">
              <Link href="/docs/getting-started">Get Started</Link>
            </Button>
          </div>
        </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center">
          <h2 className="font-heading text-3xl leading-[1.1] sm:text-3xl md:text-6xl">
            Ready to Experience the Future?
          </h2>
          <p className="max-w-[42rem] leading-relaxed text-muted-foreground sm:text-xl sm:leading-8">
            Join researchers and engineers who&rsquo;ve made the switch to RunMat
          </p>
          <div className="flex gap-4">
              <Button size="lg" asChild className="h-12 px-8 text-lg font-semibold bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-xl border-0 transition-all duration-200 hover:scale-105 hover:shadow-2xl">
              <Link href="/blog/introducing-runmat">Read the Story</Link>
            </Button>
              <Button variant="outline" size="lg" asChild className="h-12 px-8">
              <Link href="https://github.com/runmat-org/runmat">View on GitHub</Link>
            </Button>
          </div>
        </div>
        </div>
      </section>
    </div>
  );
}