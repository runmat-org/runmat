import type { Metadata } from "next";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { SiGithub } from "react-icons/si";
import { Users, GitBranch, Camera, Lock, Shield, Eye, ClipboardCheck, Cpu, Monitor, HardDrive } from "lucide-react";

import dynamic from "next/dynamic";
import Hero from "@/components/Hero";
import LazyVideo from "@/components/LazyVideo";
import { SandboxCta } from "@/components/SandboxCta";

const MatlabCodeCard = dynamic(() => import("@/components/MatlabCodeCard"), {
  loading: () => <div className="w-full max-w-3xl h-[120px] rounded-lg bg-muted/40 animate-pulse" />,
});

const BenchmarkShowcaseBlock = dynamic(
  () => import("@/components/benchmarks/BenchmarkShowcaseBlock"),
  { loading: () => <div className="w-full h-[300px] rounded-lg bg-muted/40 animate-pulse" /> },
);

const jsonLd = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "Organization",
      "@id": "https://runmat.com/#organization",
      "name": "RunMat",
      "alternateName": ["RunMat by Dystr", "Dystr"],
      "legalName": "Dystr Inc.",
      "url": "https://runmat.com",
      "logo": {
        "@type": "ImageObject",
        "url": "https://runmat.com/runmat-logo.svg",
        "caption": "RunMat"
      },
      "description": "RunMat is a high-performance, open-source runtime for math that lets you run MATLAB-syntax code in the browser, on the desktop, or from the CLI, while getting GPU-speed execution.",
      "sameAs": [
        "https://github.com/runmat-org/runmat",
        "https://x.com/runmat_com",
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
      "@id": "https://runmat.com/#website",
      "url": "https://runmat.com",
      "name": "RunMat",
      "description": "The Fastest Runtime for Your Math. RunMat fuses back-to-back ops into fewer GPU steps and intelligently manages memory.",
      "publisher": { "@id": "https://runmat.com/#organization" },
      "image": "https://web.runmatstatic.com/runmat-sandbox-dark.png",
      "potentialAction": {
        "@type": "SearchAction",
        "target": {
          "@type": "EntryPoint",
          "urlTemplate": "https://runmat.com/search?q={search_term_string}"
        },
        "query-input": "required name=search_term_string"
      }
    },
    {
      "@type": "SoftwareApplication",
      "@id": "https://runmat.com/#software",
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
      "author": { "@id": "https://runmat.com/#organization" },
      "publisher": { "@id": "https://runmat.com/#organization" },
      "downloadUrl": "https://runmat.com/download",
      "mainEntityOfPage": { "@id": "https://runmat.com/#website" },
      "screenshot": {
        "@type": "ImageObject",
        "url": "https://web.runmatstatic.com/runmat-sandbox-dark.png",
        "caption": "RunMat Desktop and Browser Sandbox"
      }
    },
    {
      "@type": "VideoObject",
      "@id": "https://runmat.com/#hero-video",
      "name": "RunMat wave interference simulation",
      "description": "GPU-accelerated wave interference simulation rendered in real time using RunMat's surf() function.",
      "thumbnailUrl": "https://web.runmatstatic.com/video/posters/runmat-wave-simulation.webp",
      "contentUrl": "https://web.runmatstatic.com/video/runmat-wave-simulation.mp4",
      "uploadDate": "2026-03-03T00:00:00Z",
      "duration": "PT10S"
    }
  ]
};

export const metadata: Metadata = {
  title: "RunMat: Free Runtime for MATLAB Code (Browser & Desktop)",
  description:
    "Execute .m files instantly with automatic GPU acceleration. An open-source runtime built on MATLAB semantics. No license or installation required.",
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
      "Execute .m files instantly with automatic GPU acceleration. An open-source runtime built on MATLAB semantics. No license or installation required.",
    url: "/",
    siteName: "RunMat",
    type: "website",
    videos: [
      {
        url: "https://web.runmatstatic.com/video/runmat-wave-simulation.mp4",
        type: "video/mp4",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "RunMat: Free Runtime for MATLAB Code (Browser & Desktop)",
    description:
      "Execute .m files instantly with automatic GPU acceleration. An open-source runtime built on MATLAB semantics. No license or installation required.",
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
      <div className="sr-only">
        <h1>RunMat: Free Runtime for MATLAB Code (Browser & Desktop)</h1>
        <p>
          Execute .m files instantly with automatic GPU acceleration. An open-source runtime built on MATLAB semantics. No license or installation required.
        </p>
      </div>

      <Hero />

      {/* Fastest to Visualize: 3D plotting */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
              See your math in 3D
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground">
              GPU-accelerated 2D and 3D plotting, built into the same environment as your code. Your plots are part of the same computation chain as your math. No copying data between systems, no separate plotting library.
            </p>
          </div>
          <div className="mx-auto max-w-3xl">
            <div className="rounded-lg border border-border overflow-hidden elevated-panel">
              <Link href="/sandbox" className="block focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-b-none rounded-t-lg overflow-hidden">
                <LazyVideo
                  className="w-full h-auto"
                  muted
                  loop
                  playsInline
                  poster="https://web.runmatstatic.com/video/posters/3d-interactive-plotting-runmat.webp"
                  aria-label="RunMat 3D interactive plotting demo"
                >
                  <source src="https://web.runmatstatic.com/video/3d-interactive-plotting-runmat.mp4" type="video/mp4" />
                </LazyVideo>
              </Link>
            </div>
          </div>
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center flex-wrap text-center mt-8">
            <Link href="/blog/matlab-plotting-guide" className="text-[0.938rem] text-foreground underline hover:text-foreground/80">
              Plotting guide
            </Link>
            <span className="hidden sm:inline text-foreground/50">•</span>
            <Link href="/docs/plotting" className="text-[0.938rem] text-foreground underline hover:text-foreground/80">
              Plotting documentation
            </Link>
            <span className="hidden sm:inline text-foreground/50">•</span>
            <Link href="/docs/matlab-function-reference#plotting" className="text-[0.938rem] text-foreground underline hover:text-foreground/80">
              Plotting function reference
            </Link>
          </div>
        </div>
      </section>

      {/* Fastest to Read: Syntax you already know */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
              Syntax you already know.
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground sm:leading-8">
              MATLAB syntax reads like the whiteboard: one line of math, one line of code. You and your team already know it. Write the math you mean, and RunMat routes it across CPU and GPU automatically.
            </p>
          </div>
          <div className="flex justify-center w-full">
            <MatlabCodeCard />
          </div>
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center flex-wrap text-center mt-8">
            <Link href="/docs/language-coverage" className="text-[0.938rem] text-foreground underline hover:text-foreground/80">
              Language guide
            </Link>
            <span className="hidden sm:inline text-foreground/50">•</span>
            <Link href="/docs/matlab-function-reference" className="text-[0.938rem] text-foreground underline hover:text-foreground/80">
              Built-in function reference
            </Link>
            <span className="hidden sm:inline text-foreground/50">•</span>
            <Link href="/blog/in-defense-of-matlab-whiteboard-style-code" className="text-[0.938rem] text-foreground underline hover:text-foreground/80">
              Why whiteboard-style code still matters
            </Link>
          </div>
        </div>
      </section>

      {/* Fastest to Run: fusion / GPU */}
      <section id="benchmarks" className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
              The fastest runtime for your math
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground sm:leading-7">
              RunMat runs math faster because of how the runtime is engineered. Fusion merges sequential operations into fewer GPU steps; residency keeps your arrays on-device between steps. That means less memory traffic, fewer program launches, and faster scripts.
            </p>
          </div>

          <div className="mx-auto max-w-3xl mb-8">
            <BenchmarkShowcaseBlock />
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center flex-wrap text-center mt-8">
            <Link href="/docs/accelerate/fusion-intro" className="text-[0.938rem] text-foreground underline hover:text-foreground/80">
              How fusion works
            </Link>
            <span className="hidden sm:inline text-foreground/50">•</span>
            <Link href="/benchmarks" className="text-[0.938rem] text-foreground underline hover:text-foreground/80">
              See the benchmarks
            </Link>
            <span className="hidden sm:inline text-foreground/50">•</span>
            <Link href="/docs/correctness" className="text-[0.938rem] text-foreground underline hover:text-foreground/80">
              How GPU results are validated
            </Link>
          </div>
        </div>
      </section>

      {/* Future: Fastest to Write (LLM / agent section) — insert here when ready */}

      {/* Fastest to Debug: shape tracking, diagnostics */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
              Debug with full visibility
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground">
              Debug faster by seeing everything as you write. Hover any variable to see its shape and type. Click on an intermediate value to inspect it. Dimension mismatches are flagged in the editor before you run.
            </p>
          </div>
          <div className="mx-auto max-w-3xl">
            <div className="rounded-lg border border-border overflow-hidden elevated-panel">
              <Link href="/sandbox" className="block focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-b-none rounded-t-lg overflow-hidden">
                <LazyVideo
                  className="w-full h-auto"
                  muted
                  loop
                  playsInline
                  poster="https://web.runmatstatic.com/video/posters/runmat-shape-tracking.webp"
                  aria-label="RunMat shape tracking and type system demo"
                >
                  <source src="https://web.runmatstatic.com/video/runmat-shape-tracking.mp4" type="video/mp4" />
                </LazyVideo>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Fastest to Share: collaboration, cloud */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
              Every change versioned. No git required.
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground">
              Every save is a version, automatically. Per-file history and full project snapshots track every change, even on terabyte-scale datasets. Share projects with your team, no git setup, no merge conflicts.
            </p>
          </div>
          <div className="mx-auto max-w-5xl grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
            <Link
              href="/sandbox"
              className="rounded-lg border border-border overflow-hidden min-h-[380px] md:row-span-3 bg-card focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              <LazyVideo
                className="w-full h-full min-h-[380px] object-cover object-left-top"
                muted
                loop
                playsInline
                poster="https://web.runmatstatic.com/video/posters/runmat-versioning.webp"
                aria-label="RunMat versioning demo"
              >
                <source src="https://web.runmatstatic.com/video/runmat-versioning.mp4" type="video/mp4" />
              </LazyVideo>
            </Link>
            <div className="rounded-lg border border-border bg-card p-6">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <GitBranch className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">Automatic file history</h3>
              <p className="text-[0.938rem] text-foreground mt-1">Every save creates a version. Browse the timeline, restore any previous state. No commits, no staging.</p>
            </div>
            <div className="rounded-lg border border-border bg-card p-6">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <Camera className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">Project snapshots</h3>
              <p className="text-[0.938rem] text-foreground mt-1">Capture your entire project in one click. Restore instantly. A clean timeline with no merge conflicts.</p>
            </div>
            <div className="rounded-lg border border-border bg-card p-6">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <Users className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">Cloud project sharing</h3>
              <p className="text-[0.938rem] text-foreground mt-1">Share projects with colleagues instantly. No shared drives, no emailing files.</p>
            </div>
          </div>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/docs/versioning" className="text-[0.938rem] text-foreground underline hover:text-foreground/80">
              How versioning works
            </Link>
            <span className="hidden sm:inline text-foreground/50">•</span>
            <Link href="/docs/collaboration" className="text-[0.938rem] text-foreground underline hover:text-foreground/80">
              Collaboration and teams
            </Link>
          </div>
        </div>
      </section>

      {/* Fastest to Scale: GPU portability, storage */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
              Scale your math, not your toolchain
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground">
              Same code runs on Apple, Nvidia, and ARM GPUs across macOS, Windows, and Linux. For large data, a sharded cloud filesystem handles multi-petabyte datasets with parallel reads and writes designed for NIC saturation. Delta snapshots version your datasets efficiently without duplicating terabytes.
            </p>
          </div>
          <div className="mx-auto max-w-5xl grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <div className="rounded-lg border border-border bg-card p-6">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <Cpu className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">Any GPU</h3>
              <p className="text-[0.938rem] text-foreground mt-1">Metal on Mac, Vulkan on Linux and ARM, DirectX 12 on Windows. No CUDA dependency.</p>
            </div>
            <div className="rounded-lg border border-border bg-card p-6">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <Monitor className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">Any OS</h3>
              <p className="text-[0.938rem] text-foreground mt-1">macOS, Windows, Linux, and headless servers. Same runtime, same results.</p>
            </div>
            <div className="rounded-lg border border-border bg-card p-6">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <HardDrive className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">High-bandwidth cloud filesystem</h3>
              <p className="text-[0.938rem] text-foreground mt-1">Sharded, petabyte-scale storage with parallel I/O and delta snapshots for efficient versioning.</p>
            </div>
          </div>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="#" className="text-[0.938rem] text-foreground underline hover:text-foreground/80">
              Learn more about RunMat filesystem
            </Link>
            <span className="hidden sm:inline text-foreground/50">•</span>
            <Link href="/contact" className="text-[0.938rem] text-foreground underline hover:text-foreground/80">
              Contact RunMat team
            </Link>
          </div>
        </div>
      </section>

      {/* Open Source Runtime */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl mb-8">
              Open source, MIT licensed
            </h2>
            <Card className="border border-border bg-card">
              <CardContent className="py-8 space-y-4">
                <Link
                  href="https://github.com/runmat-org/runmat"
                  target="_blank"
                  rel="noreferrer"
                  className="mx-auto flex h-16 w-16 items-center justify-center rounded-full border border-border bg-secondary text-foreground hover:text-foreground/80 transition-opacity"
                >
                  <SiGithub className="h-8 w-8" />
                </Link>
                <p className="text-[0.938rem] text-foreground">
                  Read every line of code that runs your math. Fork it, audit it, self-host it. No vendor lock-in, no black boxes. The runtime is on{" "}
                  <Link href="https://github.com/runmat-org/runmat" className="underline text-[hsl(var(--brand))] hover:text-[hsl(var(--brand))]/80" target="_blank" rel="noreferrer">
                    GitHub
                  </Link>
                  {" "}and actively maintained. See{" "}
                  <Link href="/docs/correctness" className="underline text-[hsl(var(--brand))] hover:text-[hsl(var(--brand))]/80">
                    how we validate every numerical path
                  </Link>
                  .
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Designed for Enterprise */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
              Secure by design
            </h2>
            <p className="max-w-[42rem] text-[0.938rem] leading-relaxed text-foreground">
              Security, compliance, and deployment options for teams that protect proprietary research and engineering data.
            </p>
          </div>
          <div className="mx-auto max-w-5xl">
            <div className="rounded-lg border border-border bg-card p-8">
              <div className="mt-4 grid gap-6 md:grid-cols-2">
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-foreground/10 text-foreground">
                    <Lock className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-foreground">SSO &amp; SCIM</p>
                    <p className="text-[0.938rem] text-foreground">Integrate with your identity provider. Provision and deprovision users automatically.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-foreground/10 text-foreground">
                    <Shield className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-foreground">ITAR-compliant deployment</p>
                    <p className="text-[0.938rem] text-foreground">Self-hosted, air-gapped option available for export-controlled environments.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-foreground/10 text-foreground">
                    <Eye className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-foreground">Open source &amp; auditable</p>
                    <p className="text-[0.938rem] text-foreground">MIT-licensed runtime. Inspect every line of code that runs your math.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-foreground/10 text-foreground">
                    <ClipboardCheck className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-foreground">SOC 2 ready</p>
                    <p className="text-[0.938rem] text-foreground">Built to SOC 2 standards. Audit planned for Q2 2026.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="flex flex-col sm:flex-row gap-4 justify-center mt-8">
            <Link href="/contact" className="text-[0.938rem] text-foreground underline hover:text-foreground/80">
              Contact RunMat
            </Link>
            <span className="hidden sm:inline text-foreground/50">•</span>
            <Link href="/blog/mission-critical-math-airgap" className="text-[0.938rem] text-foreground underline hover:text-foreground/80">
              Air-gapped deployment guide
            </Link>
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6 max-w-3xl">
          <SandboxCta source="home-bottom-cta" secondaryLabel="View pricing" secondaryHref="/pricing" />
        </div>
      </section>
    </div>
  );
}
