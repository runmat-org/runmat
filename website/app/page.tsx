import type { Metadata } from "next";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { SiGithub } from "react-icons/si";
import { Users, GitBranch, Code, Lock, Shield, Bug, Eye, ClipboardCheck, Cpu, Monitor, HardDrive } from "lucide-react";

import MatlabCodeCard from "@/components/MatlabCodeCard";
import Hero from "@/components/Hero";
import BenchmarkShowcaseBlock from "@/components/benchmarks/BenchmarkShowcaseBlock";

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
    <div className="flex flex-col min-h-screen home-page home-page-depth">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(jsonLd).replace(/<\//g, "<\\/"),
        }}
      />
      <div className="sr-only">
        <h1>RunMat: Free Runtime for MATLAB Code (Browser & Desktop)</h1>
        <p>
          Execute .m files instantly with automatic GPU acceleration. An open-source runtime compatible with standard MATLAB code. No license or installation required.
        </p>
      </div>

      <Hero />

      {/* Fastest to Read: Syntax you already know */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-heading text-3xl leading-[1.1] sm:text-4xl md:text-5xl">
              Syntax you already know.
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-base text-muted-foreground sm:text-lg sm:leading-8">
              MATLAB syntax reads like the whiteboard — one line of math, one line of code. You and your team already know it. Write the math you mean, and RunMat routes it across CPU and GPU automatically.
            </p>
          </div>
          <div className="flex justify-center w-full">
            <MatlabCodeCard />
          </div>
          <div className="mt-8 text-center">
            <Link className="underline" href="/docs/language-coverage">Language guide</Link>
            <span className="hidden sm:inline text-muted-foreground"> • </span>
            <Link className="underline" href="/docs/matlab-function-reference">Built-in function reference</Link>
            <span className="hidden sm:inline text-muted-foreground"> • </span>
            <Link className="underline" href="/blog/in-defense-of-matlab-whiteboard-style-code">Why whiteboard-style code still matters</Link>
          </div>
        </div>
      </section>

      {/* Fastest to Run: fusion / GPU */}
      <section id="benchmarks" className="w-full py-16 md:py-24 lg:py-32 bg-muted/50">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-heading text-3xl leading-[1.1] sm:text-4xl md:text-5xl">
              The fastest runtime for your math
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-muted-foreground text-lg sm:leading-7">
              RunMat runs math faster than anything else — not because of a trick, but because of how the runtime is engineered. Fusion merges sequential operations into fewer GPU steps. Residency keeps your arrays on-device between steps. The result: less memory traffic, fewer program launches, faster scripts.
            </p>
          </div>

          <div className="mx-auto max-w-3xl mb-8">
            <BenchmarkShowcaseBlock />
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center flex-wrap text-center mt-8">
            <Link href="/docs/accelerate/fusion-intro" className="text-sm hover:text-foreground text-muted-foreground transition-colors underline">
              How fusion works
            </Link>
            <span className="hidden sm:inline text-muted-foreground">•</span>
            <Link href="/benchmarks" className="text-sm hover:text-foreground text-muted-foreground transition-colors underline">
              See the benchmarks
            </Link>
          </div>
        </div>
      </section>

      {/* Future: Fastest to Write (LLM / agent section) — insert here when ready */}

      {/* Fastest to Debug: shape tracking, diagnostics */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-heading text-3xl leading-[1.1] sm:text-4xl md:text-5xl">
              Catch errors before you run
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-muted-foreground text-lg">
              Hover any variable to see its shape and type. Get dimension mismatch warnings before you run, not after. Inspect intermediate values, trace execution, and read structured logs — all built into the same environment as your code.
            </p>
          </div>
          <div className="mx-auto max-w-3xl">
            <div className="rounded-xl border border-border overflow-hidden elevated-panel">
              <Link href="/sandbox" className="block focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-b-none rounded-t-xl overflow-hidden">
                <video
                  className="w-full h-auto"
                  autoPlay
                  muted
                  loop
                  playsInline
                  aria-label="RunMat shape tracking and type system demo"
                >
                  <source src="https://web.runmatstatic.com/video/runmat-shape-tracking.mp4" type="video/mp4" />
                </video>
              </Link>
              <div className="p-4">
                <h3 className="font-semibold text-lg">Hover to see matrix dimensions</h3>
                <p className="text-muted-foreground text-sm mt-1">
                  Red underlines warn you about dimension mismatches before execution.
                </p>
              </div>
            </div>
          </div>
          <div className="mt-6 text-center">
            <Link href="#" className="text-sm text-muted-foreground hover:text-foreground underline">
              Tracing and logging in RunMat
            </Link>
            <span className="hidden sm:inline text-muted-foreground"> • </span>
            <Link href="/sandbox" className="text-sm text-muted-foreground hover:text-foreground underline">
              Open sandbox
            </Link>
          </div>
        </div>
      </section>

      {/* Fastest to Visualize: 3D plotting */}
      <section className="w-full py-16 md:py-24 lg:py-32 bg-muted/50">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-heading text-3xl leading-[1.1] sm:text-4xl md:text-5xl">
              See your results the moment they exist
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-muted-foreground text-lg">
              GPU-accelerated 2D and 3D plotting, built into the same environment as your code. Your plots are part of the same computation chain as your math — no copying data between systems, no separate plotting library.
            </p>
          </div>
          <div className="mx-auto max-w-3xl">
            <div className="rounded-xl border border-border overflow-hidden elevated-panel">
              <Link href="/sandbox" className="block focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-b-none rounded-t-xl overflow-hidden">
                <video
                  className="w-full h-auto"
                  autoPlay
                  muted
                  loop
                  playsInline
                  aria-label="RunMat 3D interactive plotting demo"
                >
                  <source src="https://web.runmatstatic.com/video/3d-interactive-plotting-runmat.mp4" type="video/mp4" />
                </video>
              </Link>
              <div className="p-4">
                <h3 className="font-semibold text-lg">Interactive 3D plotting</h3>
                <p className="text-muted-foreground text-sm mt-1">
                  Explore your results as crisp, interactive 3D surfaces. Rotate, zoom in, and inspect your data from any angle.
                </p>
              </div>
            </div>
          </div>
          <div className="mt-6 text-center">
            <Link href="#" className="text-sm text-muted-foreground hover:text-foreground underline">
              Explore plot types
            </Link>
            <span className="hidden sm:inline text-muted-foreground"> • </span>
            <Link href="#" className="text-sm text-muted-foreground hover:text-foreground underline">
              How zero-copy plotting works
            </Link>
          </div>
        </div>
      </section>

      {/* Fastest to Share: collaboration, cloud */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-heading text-3xl leading-[1.1] sm:text-4xl md:text-5xl">
              Your whole team can work on it
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-muted-foreground text-lg">
              Cloud projects sync without git. Built-in versioning — both per-file and full project snapshots — tracks every change and efficiently stores diffs, even on terabyte-scale datasets. And there&apos;s no new language to learn: if your team writes MATLAB, they already write RunMat.
            </p>
          </div>
          <div className="mx-auto max-w-5xl grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
            <div
              className="rounded-2xl border border-border bg-muted/40 min-h-[380px] md:row-span-3 flex items-center justify-center text-muted-foreground"
              aria-hidden
            >
              Video placeholder
            </div>
            <div className="rounded-2xl border border-purple-500/30 bg-gradient-to-br from-purple-500/10 to-[#0E1421] p-6">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-gray-400 mb-3">
                <Users className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-gray-100">Real-time collaboration</h3>
              <p className="text-sm text-gray-300 mt-1">Work together on shared projects with your entire team in real-time</p>
            </div>
            <div className="rounded-2xl border border-purple-500/30 bg-gradient-to-br from-purple-500/10 to-[#0E1421] p-6">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-gray-400 mb-3">
                <GitBranch className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-gray-100">File and project versioning</h3>
              <p className="text-sm text-gray-300 mt-1">Per-file history and full project snapshots. Diffs are stored efficiently — versioning stays fast and affordable even on TB/PB-scale datasets.</p>
            </div>
            <div className="rounded-2xl border border-purple-500/30 bg-gradient-to-br from-purple-500/10 to-[#0E1421] p-6">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-gray-400 mb-3">
                <Code className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-gray-100">MATLAB syntax</h3>
              <p className="text-sm text-gray-300 mt-1">No new language to learn — use the MATLAB syntax your team already knows</p>
            </div>
          </div>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="#" className="text-sm text-muted-foreground hover:text-foreground underline">
              Learn more about RunMat Cloud
            </Link>
            <span className="hidden sm:inline text-muted-foreground">•</span>
            <Link href="/pricing" className="text-sm text-muted-foreground hover:text-foreground underline">
              Sign up free
            </Link>
          </div>
        </div>
      </section>

      {/* Fastest to Scale: GPU portability, storage */}
      <section className="w-full py-16 md:py-24 lg:py-32 bg-muted/50">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-heading text-3xl leading-[1.1] sm:text-4xl md:text-5xl">
              Scale your math, not your toolchain
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-muted-foreground text-lg">
              Scale computation by running on bigger GPUs — or scale data by streaming terabyte datasets through a high-bandwidth cloud filesystem. Same code runs on Apple, Nvidia, and ARM GPUs across macOS, Windows, and Linux. No rewrites, no device flags.
            </p>
          </div>
          <div className="mx-auto max-w-5xl grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
            <div className="flex flex-col gap-4 md:row-span-3">
              <div className="rounded-2xl border border-purple-500/30 bg-gradient-to-br from-purple-500/10 to-[#0E1421] p-6">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-gray-400 mb-3">
                  <Cpu className="h-5 w-5" />
                </span>
                <h3 className="text-lg font-semibold text-gray-100">Any GPU</h3>
                <p className="text-sm text-gray-300 mt-1">Metal on Mac, Vulkan on Linux and ARM, DirectX 12 on Windows. No CUDA dependency.</p>
              </div>
              <div className="rounded-2xl border border-purple-500/30 bg-gradient-to-br from-purple-500/10 to-[#0E1421] p-6">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-gray-400 mb-3">
                  <Monitor className="h-5 w-5" />
                </span>
                <h3 className="text-lg font-semibold text-gray-100">Any OS</h3>
                <p className="text-sm text-gray-300 mt-1">macOS, Windows, Linux, and headless servers. Same runtime, same results.</p>
              </div>
              <div className="rounded-2xl border border-purple-500/30 bg-gradient-to-br from-purple-500/10 to-[#0E1421] p-6">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-gray-400 mb-3">
                  <HardDrive className="h-5 w-5" />
                </span>
              <h3 className="text-lg font-semibold text-gray-100">High-bandwidth cloud filesystem</h3>
              <p className="text-sm text-gray-300 mt-1">Stream and process terabyte-scale datasets without downloading them first. Parallel reads saturate your network for maximum throughput.</p>
              </div>
            </div>
            <div
              className="rounded-2xl border border-border bg-muted/40 min-h-[380px] md:row-span-3 flex items-center justify-center text-muted-foreground"
              aria-hidden
            >
              Video placeholder
            </div>
          </div>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="#" className="text-sm text-muted-foreground hover:text-foreground underline">
              Learn more about RunMat filesystem
            </Link>
            <span className="hidden sm:inline text-muted-foreground">•</span>
            <Link href="mailto:team@runmat.com" className="text-sm text-muted-foreground hover:text-foreground underline">
              Contact RunMat team
            </Link>
          </div>
        </div>
      </section>

      {/* Open Source Runtime */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto max-w-3xl text-center">
            <h2 className="font-heading text-3xl leading-[1.1] sm:text-3xl md:text-5xl mb-8">
              Open source, MIT licensed
            </h2>
            <Card className="border border-border/60 bg-[#0E1421] shadow-lg">
              <CardContent className="py-8 space-y-4">
                <Link
                  href="https://github.com/runmat-org/runmat"
                  target="_blank"
                  rel="noreferrer"
                  className="mx-auto flex h-28 w-28 items-center justify-center rounded-full border border-border/60 bg-background/30 text-gray-400 hover:text-gray-200 transition-colors"
                >
                  <SiGithub className="h-14 w-14" />
                </Link>
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

      {/* Designed for Enterprise */}
      <section className="w-full py-16 md:py-24 lg:py-32 bg-muted/50">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto max-w-4xl space-y-4 text-center">
            <h2 className="font-heading text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
              Designed for enterprise
            </h2>
            <p className="text-muted-foreground text-lg">
              Security, compliance, and deployment options built for organizations that ship critical software.
            </p>
          </div>
          <div className="mx-auto mt-10 max-w-5xl">
            <div className="rounded-2xl border border-purple-500/30 bg-gradient-to-br from-purple-500/15 via-[#0E1421] to-[#0A0F1C] p-8 shadow-lg">
              <div className="mt-4 grid gap-6 md:grid-cols-2">
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-gray-400">
                    <Lock className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-gray-100">SSO &amp; SCIM</p>
                    <p className="text-base text-gray-300">Integrate with your identity provider. Provision and deprovision users automatically.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-gray-400">
                    <Shield className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-gray-100">ITAR-compliant deployment</p>
                    <p className="text-base text-gray-300">Self-hosted, air-gapped option available for export-controlled environments.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-gray-400">
                    <Bug className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-gray-100">Built in Rust</p>
                    <p className="text-base text-gray-300">Memory safe by default. No garbage collector, no runtime surprises.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-gray-400">
                    <Eye className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-gray-100">Open source &amp; auditable</p>
                    <p className="text-base text-gray-300">MIT-licensed runtime. Inspect every line of code that runs your math.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-gray-400">
                    <ClipboardCheck className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-gray-100">SOC-II</p>
                    <p className="text-base text-gray-300">Audit in progress.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="flex flex-col sm:flex-row gap-4 justify-center mt-8">
            <Link href="mailto:team@runmat.com" className="text-sm text-muted-foreground hover:text-foreground underline">
              Contact RunMat
            </Link>
            <span className="hidden sm:inline text-muted-foreground">•</span>
            <Link href="#" className="text-sm text-muted-foreground hover:text-foreground underline">
              See trust center
            </Link>
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <Card className="mx-auto max-w-3xl border border-purple-500/30 bg-gradient-to-r from-purple-500/10 to-blue-500/10 shadow-lg">
            <CardContent className="py-8 space-y-4 text-center">
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">
                Start running math in your browser
              </h2>
              <p className="text-muted-foreground text-lg">
                No install, no license, no setup. Open the sandbox and write your first script in seconds.
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
                    Launch the sandbox
                  </Link>
                </Button>
                <Button variant="outline" size="lg" asChild className="h-12 px-8 text-base">
                  <Link href="/pricing">View pricing</Link>
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
