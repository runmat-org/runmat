import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

import { OSInstallCommand } from "@/components/OSInstallCommand";
import MatlabCodeCard from "@/components/MatlabCodeCard";

import HeroBenchmarkShowcase from "@/components/benchmarks/HeroBenchmarkShowcase";
import BenchmarkSweepCarousel from "@/components/benchmarks/BenchmarkSweepCarousel";

import { FusionGraphic } from "../content/svgs/fusion-graphic";

export const metadata: Metadata = {
  title: "RunMat - The Fastest Runtime for Your Math",
  description:
    "RunMat fuses back-to-back ops into fewer GPU steps and intelligently manages memory. MATLAB-syntax; 200+ built-ins. No kernel code, no rewrites.",
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
    title: "RunMat - The Fastest Runtime for Your Math",
    description:
      "RunMat fuses back-to-back ops into fewer GPU steps and keeps arrays on device. MATLAB syntax. No kernel code, no rewrites.",
    url: "/",
    siteName: "RunMat",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "RunMat - The Blazing-Fast Runtime for Math",
    description:
      "RunMat fuses back-to-back ops into fewer GPU steps and keeps arrays on device.MATLAB syntax. No kernel code, no rewrites.",
  },
};

export default function HomePage() {
  return (
    <div className="flex flex-col min-h-screen home-page">
      {/* SEO-optimized opening content */}
      <div className="sr-only">
        <h1>RunMat - The Fastest Runtime for Your Math</h1>
       
       
        <p>
        RunMat fuses back-to-back ops into fewer GPU steps and keeps arrays on device.
          MATLAB syntax. No kernel code, no rewrites.
        </p>
      </div>

      <HeroBenchmarkShowcase />

      {/* Code Example Section */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
          <h2 className="font-heading text-3xl leading-[1.1] sm:text-4xl md:text-5xl">
              Syntax you already know.
          </h2>
          <p className="max-w-[42rem] leading-relaxed text-base text-muted-foreground sm:text-lg sm:leading-8">
              Write in MATLAB, and RunMat runs your computation automatically across CPU and GPUs for maximum speed. No CUDA, no kernel code.
          </p>
          </div>

          <div className="flex justify-center w-full">
            <MatlabCodeCard />
          </div>
          <div className="mt-8 text-center">
            <Link className="underline" href="/docs/language-coverage">Language Guide</Link>
            <span className="hidden sm:inline text-blue-500"> • </span>
            <Link className="underline" href="/docs/matlab-function-reference">Built-in Function Reference</Link>
          </div>
        </div>
      </section>

      {/* GPU Fusion Architecture Section */}
      <section className="w-full py-16 md:py-24 lg:py-32 bg-muted/50">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-heading text-3xl leading-[1.1] sm:text-4xl md:text-5xl">
              Why it&apos;s Fast: GPU Fusion & Residency
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-muted-foreground sm:text-lg sm:leading-7">
              RunMat fuses sequential operations into fewer computational steps and keeps arrays on device between steps (&quot;residency&quot;). That means less memory traffic and fewer GPU program launches, so your scripts finish sooner.
            </p>
           
           
            <div className="flex justify-center w-full">
              <div className="max-w-5xl [&>svg]:max-w-full [&>svg]:h-auto">
                <FusionGraphic />
              </div>
          </div>
           

            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center flex-wrap text-center">
              <Link 
                href="/docs/accelerate/fusion-intro" 
                className="text-sm hover:text-foreground text-muted-foreground transition-colors underline"
              >
                How fusion works
              </Link>
              <span className="hidden sm:inline text-blue-500">•</span>
              <Link 
                href="/docs/fusion-guide" 
                className="text-sm hover:text-foreground text-muted-foreground transition-colors underline"
              >
                Fusion guide
              </Link>
            </div>
          </div>
        </div>
      </section>


      {/* Benchmarks Section */}
      <section id="benchmarks" className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">

          {/* Benchmarks Summary Table */}
          <div className="mt-12">
            <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-6 text-center mb-8">
              <h3 className="font-heading text-2xl leading-[1.1] sm:text-3xl md:text-4xl">
                Real Workloads, Reproducible Results
              </h3>
              <p className="max-w-[42rem] leading-relaxed text-muted-foreground text-base">
                Benchmarked on an <span className="font-semibold">Apple M2 Max, 32GB</span>.
                Times are wall-clock <span className="font-semibold">milliseconds</span> for each configuration.
              </p>
            </div>

            {/* Benchmark Sweeps Carousel */}
            <BenchmarkSweepCarousel />
            <div className="mx-auto max-w-[40rem] text-sm text-muted-foreground mt-4 space-y-3 text-center">
              <p>
                <span className="font-semibold">4K image pipeline:</span> per-image mean/std, normalization, gain/bias, gamma, and MSE.
              </p>
              <p>
                <span className="font-semibold">Monte Carlo:</span> geometric Brownian motion with terminal PnL and risk stats.
              </p>
              <p>
                <span className="font-semibold">Elementwise math:</span> long chain of sin, exp, cos, and tanh operations on big 1D arrays.
              </p>
              <p className="text-sm">
                Each number is the median of <span className="font-semibold">3 runs</span>. Full scripts live in the{" "}
                <a
                  href="https://github.com/runmat-org/runmat/tree/main/benchmarks"
                  className="underline"
                  target="_blank"
                  rel="noreferrer"
                >
                  benchmarks
                </a>{" "}
                folder.
              </p>
            </div>

          </div>
        </div>
      </section>


      {/* Features Section */}
      <section className="w-full py-16 md:py-24 lg:py-18 bg-muted/50">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-heading text-3xl leading-[1.1] sm:text-4xl md:text-5xl lg:text-6xl">
              Why Use RunMat?
            </h2>
            <p className="max-w-[46rem] text-muted-foreground">
              Comparing options? Read our{" "}
              <Link href="/blog/matlab-alternatives-runmat-vs-octave-julia-python" className="underline">
                MATLAB alternatives guide
              </Link>{" "}
              and why{" "}
              <Link href="/blog/in-defense-of-matlab-whiteboard-style-code" className="underline">
                MATLAB-style syntax still matters
              </Link>
              .
            </p>
          </div>
          <div className="mx-auto grid justify-center gap-4 sm:grid-cols-2 lg:grid-cols-3 md:max-w-[80rem]">
            <Card>
              <CardHeader className="items-center text-center">
                <div className="mb-3 text-3xl">⚡</div>
                <CardTitle>Faster For Math by Design</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center">
                  From Fusion to Residency and VM OpCodes designed for executing math fast, RunMat is optimized for math, not general purpose programming.
                </CardDescription>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="items-center text-center">
                <div className="mb-3 text-3xl">📦</div>
                <CardTitle>Cross-Platform</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center">
                  Single binary with consistent performance on macOS/Windows/Linux and headless servers. Use Mac Metal GPUs, NVDIA/AMD GPUs, or ARM Vulkan GPUs.
                </CardDescription>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="items-center text-center">
                <div className="mb-3 text-3xl">🧱</div>
                <CardTitle>Portable + Lightweight</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center">
                  Same code, everywhere: static binaries with consistent performance on macOS/Windows/Linux and headless servers. GPU portability via Metal, DirectX 12, and Vulkan—no CUDA lock-in. Great for laptops, clusters, and CI.

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
              Free and Open Source
          </h2>
          <p className="max-w-[42rem] leading-relaxed text-muted-foreground sm:text-xl sm:leading-8">
              Copy and paste the command below to get started with RunMat.
          </p>
          
          <OSInstallCommand className="w-full max-w-4xl" />
          
            <div className="flex flex-col sm:flex-row gap-4">
              <Button size="lg" asChild className="h-12 px-8 text-lg font-semibold bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-xl border-0 transition-all duration-200">
                <Link href="/download">More Install Options</Link>
            </Button>
              <Button variant="outline" size="lg" asChild className="h-12 px-8 text-lg">
                <Link href="/docs/getting-started">Get Started</Link>
            </Button>
          </div>
        </div>
        </div>
      </section>

    </div>
  );
}