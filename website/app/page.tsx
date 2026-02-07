import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SiGithub } from "react-icons/si";

import MatlabCodeCard from "@/components/MatlabCodeCard";

import HeroBenchmarkShowcase from "@/components/benchmarks/HeroBenchmarkShowcase";
import BenchmarkSweepCarousel from "@/components/benchmarks/BenchmarkSweepCarousel";

import { FusionGraphic } from "../content/svgs/fusion-graphic";

const jsonLd = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "Organization",
      "@id": "https://runmat.org/#organization",
      "name": "RunMat",
      "alternateName": ["RunMat by Dystr", "Dystr"],
      "legalName": "Dystr Inc.",
      "url": "https://runmat.org",
      "logo": {
        "@type": "ImageObject",
        "url": "https://runmat.org/runmat-logo.svg",
        "caption": "RunMat"
      },
      "description": "RunMat is a high-performance, open-source runtime for math that lets you run MATLAB-syntax code in the browser, on the desktop, or from the CLI, while getting GPU-speed execution.",
      "sameAs": [
        "https://github.com/runmat-org/runmat",
        "https://x.com/runmat_org",
        "https://dystr.com"
      ],
      "knowsAbout": [
        "Scientific Computing",
        "High Performance Computing",
        "MATLAB",
        "WebGPU",
        "Compiler Design"
      ],
      "contactPoint": {
        "@type": "ContactPoint",
        "contactType": "customer support",
        "email": "team@runmat.com"
      }
    },
    {
      "@type": "WebSite",
      "@id": "https://runmat.org/#website",
      "url": "https://runmat.org",
      "name": "RunMat",
      "description": "The Fastest Runtime for Your Math. RunMat fuses back-to-back ops into fewer GPU steps and intelligently manages memory.",
      "publisher": { "@id": "https://runmat.org/#organization" },
      "image": "https://web.runmatstatic.com/runmat-sandbox-dark.png",
      "potentialAction": {
        "@type": "SearchAction",
        "target": {
          "@type": "EntryPoint",
          "urlTemplate": "https://runmat.org/search?q={search_term_string}"
        },
        "query-input": "required name=search_term_string"
      }
    },
    {
      "@type": "SoftwareApplication",
      "@id": "https://runmat.org/#software",
      "name": "RunMat",
      "description": "RunMat is a high-performance, open-source runtime for math that lets you run MATLAB-syntax code in the browser, on the desktop, or from the CLI, while getting GPU-speed execution.",
      "license": "https://opensource.org/licenses/MIT",
      "applicationCategory": "ScientificApplication",
      "applicationSubCategory": "Numerical Analysis & Simulation",
      "operatingSystem": ["Windows", "macOS", "Linux", "Browser"],
      "softwareVersion": "Beta",
      "featureList": [
        "JIT-accelerated MATLAB-style syntax",
        "RunMat Desktop: Full IDE experience with code editor, file explorer, and live plotting in-browser",
        "Automatic GPU Fusion & Memory Management",
        "Cross-platform binary (Metal, Vulkan, DX12) and CLI support",
        "Interactive 2D and 3D plotting with GPU acceleration",
        "Real-time type and shape tracking with dimension error detection",
        "Execution tracing and diagnostic logging"
      ],
      "offers": {
        "@type": "Offer",
        "price": "0",
        "priceCurrency": "USD",
        "availability": "https://schema.org/InStock"
      },
      "author": { "@id": "https://runmat.org/#organization" },
      "publisher": { "@id": "https://runmat.org/#organization" },
      "downloadUrl": "https://runmat.org/download",
      "mainEntityOfPage": { "@id": "https://runmat.org/#website" },
      "screenshot": {
        "@type": "ImageObject",
        "url": "https://web.runmatstatic.com/runmat-sandbox-dark.png",
        "caption": "RunMat Desktop and Browser Sandbox"
      }
    }
  ]
};

export const metadata: Metadata = {
  title: "RunMat: Free Runtime for MATLAB Code (Browser & Desktop)",
  description:
    "Execute .m files instantly with automatic GPU acceleration. An open-source runtime compatible with standard MATLAB code. No license or installation required.",
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
    "matlab 3d plot",
    "matlab plotting online",
    "interactive matlab plots",
    "matlab debugging",
    "matlab dimension error",
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
    title: "RunMat: Free Runtime for MATLAB Code (Browser & Desktop)",
    description:
      "Execute .m files instantly with automatic GPU acceleration. An open-source runtime compatible with standard MATLAB code. No license or installation required.",
    url: "/",
    siteName: "RunMat",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "RunMat: Free Runtime for MATLAB Code (Browser & Desktop)",
    description:
      "Execute .m files instantly with automatic GPU acceleration. An open-source runtime compatible with standard MATLAB code. No license or installation required.",
  },
};

export default function HomePage() {
  return (
    <div className="flex flex-col min-h-screen home-page">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(jsonLd).replace(/<\//g, "<\\/"),
        }}
      />
      {/* SEO-optimized opening content */}
      <div className="sr-only">
        <h1>RunMat: Free Runtime for MATLAB Code (Browser & Desktop)</h1>
        <p>
          Execute .m files instantly with automatic GPU acceleration. An open-source runtime compatible with standard MATLAB code. No license or installation required.
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

      {/* Developer Experience Section */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-heading text-3xl leading-[1.1] sm:text-4xl md:text-5xl">
              See your results. Catch your mistakes.
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-muted-foreground sm:text-lg">
              Rich plotting and live diagnostics, built into the same environment as your code. No separate tools. No waiting until runtime to find errors.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
            <div className="rounded-xl border border-border overflow-hidden bg-muted/40">
              <video
                className="w-full h-auto"
                autoPlay
                muted
                loop
                playsInline
                aria-label="RunMat 3D interactive plotting demo"
              >
                <source
                  src="https://web.runmatstatic.com/video/3d-interactive-plotting-runmat.mp4"
                  type="video/mp4"
                />
              </video>
              <div className="p-4">
                <h3 className="font-semibold text-lg">Interactive 3D Plotting</h3>
                <p className="text-muted-foreground text-sm mt-1">
                  Explore your results as crisp, interactive 3D surfaces. Rotate, zoom in, and inspect your data from any angle.
                </p>
              </div>
            </div>

            <div className="rounded-xl border border-border overflow-hidden bg-muted/40">
              <video
                className="w-full h-auto"
                autoPlay
                muted
                loop
                playsInline
                aria-label="RunMat shape tracking and type system demo"
              >
                <source
                  src="https://web.runmatstatic.com/video/runmat-shape-tracking.mp4"
                  type="video/mp4"
                />
              </video>
              <div className="p-4">
                <h3 className="font-semibold text-lg">Catch Errors Before You Run</h3>
                <p className="text-muted-foreground text-sm mt-1">
                  Hover to see matrix dimensions. Red underlines warn you about dimension
                  mismatches before execution.
                </p>
              </div>
            </div>
          </div>

        </div>
      </section>

      {/* GPU Fusion Architecture Section */}
      <section className="w-full py-16 md:py-24 lg:py-32 bg-muted/50">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-heading text-3xl leading-[1.1] sm:text-4xl md:text-5xl">
              Why it&apos;s fast: GPU fusion &amp; residency
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
                Real workloads, reproducible results
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
              One runtime. Every platform.
            </h2>
            <p className="max-w-[46rem] text-muted-foreground">
              The same math engine runs in your browser, on your desktop, and from the command line — with no dependencies and no license. Comparing options? Read our{" "}
              <Link href="/blog/free-matlab-alternatives" className="underline">
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
                <CardTitle>Browser, Desktop, or CLI</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center">
                  Start in RunMat instantly with zero install. Move to the desktop app for local file access. Run headless from the CLI on servers and CI pipelines. Same runtime, same results, everywhere.
                </CardDescription>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="items-center text-center">
                <div className="mb-3 text-3xl">📦</div>
                <CardTitle>Any GPU. No Lock-In.</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center">
                  Metal on Mac, Vulkan on Linux and ARM, DirectX 12 on Windows. No CUDA dependency, no vendor lock-in. Your code runs on whatever hardware your team already has.
                </CardDescription>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="items-center text-center">
                <div className="mb-3 text-3xl">🧱</div>
                <CardTitle>Open Source and Free</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center">
                  The RunMat runtime is MIT-licensed and open source. No per-seat fees, no license servers, no vendor audits. Collaboration, cloud storage, and team features are coming soon.
                </CardDescription>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Open Source Section */}
      <section className="w-full py-16 md:py-24 lg:py-32 bg-muted/30">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto max-w-3xl text-center">
            <h2 className="font-heading text-3xl leading-[1.1] sm:text-3xl md:text-5xl mb-8">
              Free and open source
            </h2>
            <Card className="border border-border/60 bg-[#0E1421] shadow-lg">
              <CardContent className="py-8 space-y-4">
                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full border border-border/60 bg-background/30 text-sm text-gray-400">
                  <SiGithub className="h-7 w-7" />
                </div>
                <p className="text-gray-300 text-lg">
                  The RunMat runtime is MIT licensed and on{" "}
                  <Link href="https://github.com/runmat-org/runmat" className="underline text-blue-300 hover:text-blue-200" target="_blank" rel="noreferrer">
                    GitHub
                  </Link>
                  . The JIT, the fusion engine, the GPU planner — all open source and actively maintained.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <Card className="mx-auto max-w-3xl border border-purple-500/30 bg-gradient-to-r from-purple-500/10 to-blue-500/10 shadow-lg">
            <CardContent className="py-8 space-y-4 text-center">
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">
                Run MATLAB code online — no install, no license
              </h2>
              <p className="text-muted-foreground text-lg">
                Start running math immediately in your browser.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Button
                  size="lg"
                  asChild
                  className="h-12 px-8 text-base font-semibold bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-xl border-0 transition-all duration-200"
                >
                  <Link
                    href="/sandbox"
                    data-ph-capture-attribute-destination="sandbox"
                    data-ph-capture-attribute-source="home-bottom-cta"
                    data-ph-capture-attribute-cta="launch-sandbox"
                  >
                    Launch the Sandbox
                  </Link>
                </Button>
                <Button variant="outline" size="lg" asChild className="h-12 px-8 text-base">
                  <Link href="/download">Other download options</Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

    </div>
  );
}