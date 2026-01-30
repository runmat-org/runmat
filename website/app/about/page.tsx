import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowUpRight } from "lucide-react";
import { SiGithub } from "react-icons/si";

export const metadata: Metadata = {
  title: "About RunMat",
  description:
    "RunMat is a high-performance, open-source runtime for math that runs MATLAB-syntax code in the browser, on desktop, or from the CLI with GPU-speed execution.",
  alternates: { canonical: "https://runmat.org/about" },
};

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 md:px-6 lg:px-8 pt-12 md:pt-16 lg:pt-20 pb-16 md:pb-24 lg:pb-32">
        {/* Who We Are */}
        <section className="pb-12 md:pb-16 lg:pb-20">
          <div className="grid gap-10 lg:grid-cols-2 lg:items-center">
            <div className="order-2 lg:order-1 rounded-2xl border border-border/60 bg-muted/40 p-4">
              <div className="relative h-[320px] w-full overflow-hidden rounded-xl border border-border/60 bg-background/40">
                <Image
                  src="https://web.runmatstatic.com/matlab-runmat-whiteboard-c.png"
                  alt="Engineer copying matrix equations from a whiteboard into MATLAB-style code on a laptop."
                  fill
                  sizes="(min-width: 1024px) 50vw, 100vw"
                  className="object-cover"
                  priority
                />
              </div>
              <p className="mt-3 text-xs text-muted-foreground">
                Co-founders{" "}
                <Link
                  href="https://www.linkedin.com/in/nallana/"
                  target="_blank"
                  rel="noreferrer"
                  className="underline underline-offset-2 hover:text-foreground"
                >
                  Nabeel Allana
                </Link>{" "}
                and{" "}
                <Link
                  href="https://www.linkedin.com/in/julie-ruiz-64b24328/"
                  target="_blank"
                  rel="noreferrer"
                  className="underline underline-offset-2 hover:text-foreground"
                >
                  Julie Ruiz
                </Link>
                .
              </p>
            </div>
            <div className="order-1 lg:order-2 space-y-4">
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">Who We Are</h2>
              <p className="text-muted-foreground text-lg">
                RunMat is built by engineers who saw the problem firsthand — years spent watching talented people lose
                time to slow tools, expensive licenses, and workflows that couldn&apos;t keep up with modern hardware.
              </p>
              <p className="text-muted-foreground text-lg">
                We started with Dystr, an AI-powered engineering workbook. We learned that engineers don&apos;t want new
                platforms — they want their existing workflows to work better. That insight led us to RunMat: the runtime
                and tools we wished existed.
              </p>
            </div>
          </div>
        </section>

        {/* Backed By */}
        <section className="py-12 md:py-16 lg:py-20">
          <Card className="mx-auto max-w-4xl border border-border/60 bg-[#0E1421] shadow-lg">
            <CardContent className="py-8 text-center space-y-4">
              <p className="text-muted-foreground text-lg">
                Our investors include backers of SpaceX, Anduril, DeepMind, and Notion.
              </p>
              <div className="mx-auto grid max-w-xl grid-cols-2 items-center justify-items-center gap-6 rounded-lg bg-background/30 px-6 py-4 sm:grid-cols-4">
                <Image
                  src="https://web.runmatstatic.com/VC%20logos/long-journey.png"
                  alt="Long Journey Ventures logo"
                  width={160}
                  height={44}
                  className="h-8 w-auto object-contain opacity-80"
                />
                <Image
                  src="https://web.runmatstatic.com/VC%20logos/omni.png"
                  alt="Omni logo"
                  width={110}
                  height={44}
                  className="h-8 w-auto object-contain opacity-80"
                />
                <Image
                  src="https://web.runmatstatic.com/VC%20logos/unpopular-ventures.png"
                  alt="Unpopular Ventures logo"
                  width={160}
                  height={44}
                  className="h-8 w-auto object-contain opacity-80"
                />
                <Image
                  src="https://web.runmatstatic.com/VC%20logos/female-founders-fund.png"
                  alt="Female Founders Fund logo"
                  width={200}
                  height={44}
                  className="h-8 w-auto object-contain opacity-80"
                />
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Who is RunMat For */}
        <section className="py-16 md:py-24 lg:py-32">
          <div className="grid gap-10 lg:grid-cols-2 lg:items-center">
            <div className="space-y-4 text-center lg:text-left">
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">Who is RunMat For</h2>
              <p className="text-muted-foreground text-lg">
                If you do math on a computer — simulations, signal processing, image pipelines, financial modeling — and
                you&apos;ve ever felt stuck between MATLAB&apos;s readability and Python&apos;s openness, RunMat is for you.
              </p>
              <p className="text-muted-foreground text-lg">
                Whether you&apos;re migrating off expensive MATLAB licenses, looking for GPU acceleration without learning
                CUDA, or building workflows where AI writes code and humans verify it — we built this for you.
              </p>
            </div>
            <div className="rounded-2xl border border-border/60 bg-muted/40 p-4">
              <div className="relative h-[320px] w-full overflow-hidden rounded-xl border border-border/60 bg-background/40">
                <Image
                  src="https://web.runmatstatic.com/engineer-math.png"
                  alt="Engineer working through equations on a whiteboard."
                  fill
                  sizes="(min-width: 1024px) 50vw, 100vw"
                  className="object-cover"
                />
              </div>
            </div>
          </div>
        </section>

        {/* Math Without Limits */}
        <section className="py-16 md:py-24 lg:py-32">
          <div className="grid gap-10 lg:grid-cols-2 lg:items-start">
            <div className="order-2 lg:order-1 rounded-2xl border border-border/60 bg-muted/40 p-4 w-full">
              <div className="relative w-full overflow-hidden rounded-xl border border-border/60 bg-background/40 aspect-video">
                <Image
                  src="https://web.runmatstatic.com/matlab-runmat-whiteboard-c.png"
                  alt="Engineer copying matrix equations from a whiteboard into MATLAB-style code on a laptop."
                  fill
                  sizes="(min-width: 1024px) 50vw, 100vw"
                  className="object-contain"
                />
              </div>
            </div>
            <div className="order-1 lg:order-2 space-y-4">
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">Math Without Limits</h2>
              <p className="text-muted-foreground text-lg">
                Code generation is changing everything. LLMs write more code every day. But the tools engineers use to
                run and verify that code? Still stuck in the past.
              </p>
              <p className="text-muted-foreground text-lg">
                Some engineers stay on MATLAB — paying for expensive licenses, dealing with slow runtimes, and missing
                GPU support. Others moved to Python for its openness and ecosystem — but lost the readable, math-like
                syntax that makes verification easy.
              </p>
              <p className="text-muted-foreground text-lg">
                We believe MATLAB syntax is more relevant today than ever — precisely because it reads like math. As AI
                writes more of the first draft, the ability to verify code at a glance becomes the bottleneck. MATLAB
                syntax solves that.
              </p>
              <p className="text-muted-foreground text-lg">
                Imagine: generate a controller in MATLAB syntax with an LLM, verify it visually, then run it on GPU in
                seconds. That&apos;s the workflow we&apos;re enabling.
              </p>
              <p className="text-foreground text-lg font-semibold">
                To do math without limits, you need tools without limits. We&apos;re building that foundation.
              </p>
            </div>
          </div>
        </section>

        {/* Our Mission */}
        <section className="py-16 md:py-24 lg:py-32">
          <Card className="mx-auto max-w-4xl border border-border/60 bg-[#0E1421] shadow-lg">
            <CardContent className="py-8 space-y-3 text-center">
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">Our Mission</h2>
              <p className="text-xl text-foreground font-semibold">
                To enable people to do math and physics that they couldn&apos;t do before — by creating a modern,
                high-performance alternative to MATLAB.
              </p>
              <p className="text-muted-foreground text-lg">
                We believe the programming barrier that prevents engineers from doing sophisticated modeling should
                disappear. The right syntax, the right tools, and the right performance can make that happen.
              </p>
            </CardContent>
          </Card>
        </section>

        {/* What We've Built */}
        <section className="py-16 md:py-24 lg:py-32">
          <section className="mb-12">
            <div className="mx-auto max-w-4xl text-center space-y-4">
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">What is RunMat today</h2>
              <p className="text-muted-foreground text-lg">
                RunMat is a high-performance, open-source runtime for math that lets you run MATLAB-syntax code in the
                browser, on the desktop, or from the CLI, while getting GPU-speed execution.
              </p>
            </div>
          </section>
          <div className="mx-auto max-w-4xl rounded-2xl border border-border/60 bg-muted/40 p-0 mb-12">
            <div className="relative h-[440px] w-full overflow-hidden rounded-2xl bg-background/40">
              <Image
                src="https://web.runmatstatic.com/runmat-sandbox-dark.png"
                alt="RunMat sandbox interface preview."
                fill
                sizes="(min-width: 1024px) 960px, 100vw"
                className="object-contain"
              />
            </div>
          </div>
          <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-3">
            <Card className="border border-blue-500/30 bg-[#0E1421] shadow-lg">
              <CardHeader className="border-b border-border/60">
                <CardTitle className="text-xl">Runtime</CardTitle>
              </CardHeader>
              <CardContent className="text-lg text-muted-foreground space-y-2">
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-blue-300">•</span>
                  <p>Automatic GPU acceleration via kernel fusion</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-blue-300">•</span>
                  <p>~5ms startup (180x faster than GNU Octave)</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-blue-300">•</span>
                  <p>Async operations for non-blocking execution</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-blue-300">•</span>
                  <p>Built in Rust for safety and portability</p>
                </div>
              </CardContent>
            </Card>
            <Card className="border border-green-500/30 bg-[#0E1421] shadow-lg">
              <CardHeader className="border-b border-border/60">
                <CardTitle className="text-xl">Platform</CardTitle>
              </CardHeader>
              <CardContent className="text-lg text-muted-foreground space-y-2">
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Browser sandbox — run code instantly, no installation</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>World-class plotting engine</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>LSP and linter for real-time error checking and autocomplete</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Intuitive error handling with clear, actionable messages</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Better ergonomics than Python or traditional MATLAB</p>
                </div>
              </CardContent>
            </Card>
            <Card className="border border-purple-500/30 bg-[#0E1421] shadow-lg">
              <CardHeader className="border-b border-border/60">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-xl">Enterprise</CardTitle>
                  <span className="rounded-full bg-purple-500/20 px-2 py-0.5 text-xs font-medium text-purple-300">
                    Coming Soon
                  </span>
                </div>
              </CardHeader>
              <CardContent className="text-lg text-muted-foreground space-y-2">
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-purple-300">•</span>
                  <p>Air-gapped deployment options for security-sensitive environments</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-purple-300">•</span>
                  <p>Local-first architecture — your code runs on your hardware</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Where We're Headed */}
        <section className="py-16 md:py-24 lg:py-32">
          <div className="mx-auto max-w-5xl text-center mb-12 space-y-3">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">Where We&apos;re Headed</h2>
          </div>
          <Card className="mx-auto max-w-5xl rounded-2xl border border-purple-500/30 bg-gradient-to-br from-purple-500/15 via-[#0E1421] to-[#0A0F1C] p-8 shadow-lg">
            <CardContent className="space-y-6">
              <p className="text-muted-foreground text-lg">
                You can use RunMat today as a drop-in MATLAB-style runtime, and we&apos;re evolving it into an AI-native
                development environment for engineers — one that combines:
              </p>
              <div className="grid gap-4 md:grid-cols-2 text-lg text-muted-foreground">
                <div className="flex items-start gap-3">
                  <ArrowUpRight className="mt-1 h-4 w-4 text-blue-300" />
                  <span>A superior MATLAB alternative as the foundation</span>
                </div>
                <div className="flex items-start gap-3">
                  <ArrowUpRight className="mt-1 h-4 w-4 text-blue-300" />
                  <span>Model integration for AI-assisted coding</span>
                </div>
                <div className="flex items-start gap-3">
                  <ArrowUpRight className="mt-1 h-4 w-4 text-blue-300" />
                  <span>Modern development tools and workflows</span>
                </div>
                <div className="flex items-start gap-3">
                  <ArrowUpRight className="mt-1 h-4 w-4 text-blue-300" />
                  <span>Scalable deployment from laptop to cloud to air-gapped enterprise</span>
                </div>
              </div>
              <div className="rounded-xl border border-border/60 bg-background/40 p-5 text-muted-foreground text-lg">
                <span className="font-semibold text-foreground">The goal:</span> Let engineers focus on math and physics, not programming.
              </div>
              <div className="flex items-center justify-start">
                <Link href="/blog/why-we-built-runmat" className="text-blue-300 hover:text-blue-200">
                  Read about our story →
                </Link>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Open Source */}
        <section className="py-16 md:py-24 lg:py-32">
          <div className="mx-auto max-w-3xl text-center">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-8">Open Source</h2>
            <Card className="border border-border/60 bg-[#0E1421] shadow-lg">
              <CardContent className="py-8 space-y-4">
                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full border border-border/60 bg-background/30 text-sm text-muted-foreground">
                  <SiGithub className="h-7 w-7" />
                </div>
                <p className="text-muted-foreground text-lg">
                  The core runtime is MIT licensed and on{" "}
                  <Link href="https://github.com/runmat-org/runmat" className="underline">
                    GitHub
                  </Link>{" "}
                  — the JIT, the fusion engine, the GPU planner. We&apos;re committed to keeping it open source and
                  actively maintained.
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Get Started */}
        <section className="py-16 md:py-24 lg:py-32 text-center">
          <Card className="mx-auto max-w-3xl border border-purple-500/30 bg-gradient-to-r from-purple-500/10 to-blue-500/10 shadow-lg">
            <CardContent className="py-8 space-y-4">
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">
                Run MATLAB code online - no install, no license
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
                    data-ph-capture-attribute-source="about-bottom-cta"
                    data-ph-capture-attribute-cta="launch-sandbox"
                  >
                    Launch the Sandbox
                  </Link>
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  asChild
                  className="h-12 px-8 text-base bg-[#0E1B1E] dark:bg-[#0E1B1E] border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100"
                >
                  <Link href="/download">Other download options</Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </section>
      </div>
    </div>
  );
}
