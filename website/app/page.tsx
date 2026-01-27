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
        "Cross-platform binary (Metal, Vulkan, DX12) and CLI support"
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
                  <Link
                    href="https://runmat.org/sandbox"
                    className="underline"
                    target="_blank"
                    rel="noreferrer"
                  >
                    Try instantly in the browser with no install,
                  </Link>{" "}
                  or download the CLI for local scripts. Same code runs on macOS, Windows, Linux, and headless servers. GPU portability via Metal, DirectX 12, and Vulkan—no CUDA lock-in.
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