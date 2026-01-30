import type { Metadata } from "next";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  AlertTriangle,
  FileText,
  RefreshCcw,
  Users,
  FastForward,
  ArrowUpRight,
} from "lucide-react";

export const metadata: Metadata = {
  title: "About RunMat",
  description:
    "RunMat is a high-performance, open-source runtime for math that runs MATLAB-syntax code in the browser, on desktop, or from the CLI with GPU-speed execution.",
  alternates: { canonical: "https://runmat.org/about" },
};

const visionCards = [
  { title: "Debug terabytes of telemetry in real time", icon: AlertTriangle },
  { title: "Achieve full traceability from lab to field", icon: FileText },
  { title: "Test and verify at any scale", icon: RefreshCcw },
  { title: "Capture and share operational knowledge", icon: Users },
  { title: "Move fast without compromising safety", icon: FastForward },
];

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 md:px-6 lg:px-8 py-16 md:py-20">
        {/* Hero */}
        <section className="w-full mb-16" id="hero">
          <div className="grid grid-cols-1 gap-10 lg:grid-cols-2 lg:items-center">
            <div className="flex flex-col space-y-6 text-left items-start">
              <div className="mb-2 p-0 text-lg font-semibold uppercase tracking-wide text-primary">
                About RunMat
              </div>
              <h1 className="font-heading text-left leading-tight tracking-tight text-[clamp(2.6rem,4.8vw,4.25rem)] sm:text-[clamp(3rem,4vw,5rem)] lg:text-[clamp(3.25rem,3.6vw,5.25rem)]">
                A modern runtime for math — open, fast, and GPU-accelerated
              </h1>
              <p className="max-w-[42rem] leading-relaxed text-muted-foreground text-lg">
                RunMat is a high-performance, open-source runtime for math that lets you run MATLAB-syntax code in the
                browser, on the desktop, or from the CLI, while getting GPU-speed execution.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto">
                <Button
                  size="lg"
                  asChild
                  className="h-12 px-8 text-base font-semibold bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-xl border-0 transition-all duration-200"
                >
                  <Link href="/sandbox">Try RunMat in Your Browser →</Link>
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  asChild
                  className="h-12 px-8 text-base bg-[#0E1B1E] dark:bg-[#0E1B1E] border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100"
                >
                  <Link href="/docs/getting-started">Get Started</Link>
                </Button>
              </div>
            </div>
            <div className="rounded-xl border border-border bg-muted/40 p-2 bg-[radial-gradient(ellipse_at_top,_rgba(124,58,237,0.25),_transparent_60%)]">
              <div className="flex h-[320px] w-full items-center justify-center rounded-lg border border-border/60 bg-background/40 text-sm text-muted-foreground">
                Hero image placeholder
              </div>
            </div>
          </div>
        </section>

        {/* What is RunMat */}
        <section className="mb-16 border-y border-border/60 py-12">
          <div className="mx-auto max-w-4xl text-center space-y-4">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">What is RunMat</h2>
            <p className="text-muted-foreground text-lg">
              RunMat is a high-performance, open-source runtime for math that lets you run MATLAB-syntax code in the
              browser, on the desktop, or from the CLI, while getting GPU-speed execution.
            </p>
          </div>
        </section>

        {/* Who We Are */}
        <section className="mb-16">
          <div className="grid gap-10 lg:grid-cols-2 lg:items-center">
            <div className="order-2 lg:order-1 rounded-2xl border border-border/60 bg-muted/40 p-4">
              <div className="flex h-[320px] w-full items-center justify-center rounded-xl border border-border/60 bg-background/40 text-sm text-muted-foreground">
                Team photo placeholder
              </div>
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
              <p className="text-muted-foreground text-lg">
                We&apos;re committed to maintaining RunMat as a long-term open-source project. If you&apos;re building
                something with RunMat, we&apos;d love to hear from you.
              </p>
            </div>
          </div>
        </section>

        {/* Backed By */}
        <section className="mb-16">
          <Card className="mx-auto max-w-4xl border border-border/60 bg-[#0E1421] shadow-lg">
            <CardContent className="py-8 text-center space-y-4">
              <h3 className="text-2xl font-semibold text-foreground">Backed By</h3>
              <p className="text-muted-foreground text-lg">
                RunMat is backed by visionary investors who funded SpaceX, Anduril, DeepMind, and Notion.
              </p>
              <div className="mx-auto flex h-16 max-w-xl items-center justify-center rounded-lg border border-border/60 bg-background/30 text-sm text-muted-foreground">
                Investor logos placeholder
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Who is RunMat For */}
        <section className="mb-16">
          <div className="mx-auto max-w-4xl space-y-4 text-center">
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
        </section>

        {/* Math Without Limits */}
        <section className="mb-16">
          <div className="grid gap-10 lg:grid-cols-2 lg:items-center">
            <div className="space-y-4">
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
            <div className="rounded-2xl border border-border/60 bg-muted/40 p-4">
              <div className="flex h-[320px] w-full items-center justify-center rounded-xl border border-border/60 bg-background/40 text-sm text-muted-foreground">
                Math section image placeholder
              </div>
            </div>
          </div>
        </section>

        {/* Our Vision */}
        <section className="mb-16">
          <div className="mx-auto max-w-5xl text-center mb-10 space-y-3">
            <div className="text-sm font-semibold uppercase tracking-wide text-primary">Our Vision</div>
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">
              We&apos;re solving existential problems for mission-critical industries
            </h2>
            <p className="text-muted-foreground text-lg">
              Placeholder vision copy goes here. This should briefly describe the scale of the problems RunMat is
              tackling and why better math tooling matters.
            </p>
          </div>
          <div className="mx-auto grid max-w-5xl gap-4 sm:grid-cols-2 lg:grid-cols-5">
            {visionCards.map(card => {
              const Icon = card.icon;
              return (
                <Card key={card.title} className="border border-border/60 bg-[#0E1421] shadow-lg">
                  <CardContent className="p-5 space-y-3">
                    <span className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-orange-500/30 bg-orange-500/10 text-orange-200">
                      <Icon className="h-4 w-4" />
                    </span>
                    <p className="text-sm text-muted-foreground">{card.title}</p>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </section>

        {/* Our Mission */}
        <section className="mb-16">
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
        <section className="mb-16">
          <div className="mx-auto max-w-5xl text-center mb-8 space-y-3">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">What We&apos;ve Built</h2>
            <p className="text-muted-foreground text-lg">
              A new open-source runtime and platform to modernize MATLAB — the IDE, the capabilities, and the execution
              speed.
            </p>
          </div>
          <div className="mx-auto max-w-5xl rounded-2xl border border-border/60 bg-muted/40 p-4 mb-8">
            <div className="flex h-[320px] w-full items-center justify-center rounded-xl border border-border/60 bg-background/40 text-sm text-muted-foreground">
              Product screenshot placeholder
            </div>
          </div>
          <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-3">
            <Card className="border border-border/60 bg-[#0E1421] shadow-lg">
              <CardHeader className="border-b border-border/60">
                <CardTitle className="text-xl">Runtime</CardTitle>
              </CardHeader>
              <CardContent className="text-lg text-muted-foreground space-y-2">
                <p>Automatic GPU acceleration via kernel fusion</p>
                <p>~5ms startup (180x faster than GNU Octave)</p>
                <p>Async operations for non-blocking execution</p>
                <p>Built in Rust for safety and portability</p>
              </CardContent>
            </Card>
            <Card className="border border-border/60 bg-[#0E1421] shadow-lg">
              <CardHeader className="border-b border-border/60">
                <CardTitle className="text-xl">Platform</CardTitle>
              </CardHeader>
              <CardContent className="text-lg text-muted-foreground space-y-2">
                <p>Browser sandbox — run code instantly, no installation</p>
                <p>World-class plotting engine</p>
                <p>LSP and linter for real-time error checking and autocomplete</p>
                <p>Intuitive error handling with clear, actionable messages</p>
                <p>Better ergonomics than Python or traditional MATLAB</p>
              </CardContent>
            </Card>
            <Card className="border border-border/60 bg-[#0E1421] shadow-lg">
              <CardHeader className="border-b border-border/60">
                <CardTitle className="text-xl">Enterprise</CardTitle>
              </CardHeader>
              <CardContent className="text-lg text-muted-foreground space-y-2">
                <p>Air-gapped deployment options for security-sensitive environments</p>
                <p>Local-first architecture — your code runs on your hardware</p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Open Source */}
        <section className="mb-16">
          <div className="mx-auto max-w-3xl text-center">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-6">Open Source</h2>
            <Card className="border border-border/60 bg-[#0E1421] shadow-lg">
              <CardContent className="py-8 space-y-4">
                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full border border-border/60 bg-background/30 text-sm text-muted-foreground">
                  GH
                </div>
                <p className="text-muted-foreground text-lg">
                  MIT licensed. The entire runtime is on{" "}
                  <Link href="https://github.com/runmat-org/runmat" className="underline">
                    GitHub
                  </Link>{" "}
                  — the JIT, the fusion engine, the GPU planner. All of it.
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Where We're Headed */}
        <section className="mb-16">
          <div className="mx-auto max-w-5xl text-center mb-8 space-y-3">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">Where We&apos;re Headed</h2>
          </div>
          <Card className="mx-auto max-w-5xl border border-border/60 bg-[#0E1421] shadow-lg">
            <CardContent className="p-8 space-y-6">
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
                <span className="font-semibold text-foreground">The goal:</span> Remove the programming barrier that
                prevents engineers from doing sophisticated math and physics modeling.
              </div>
              <div className="flex items-center justify-start">
                <Link href="/blog/why-we-built-runmat" className="text-blue-300 hover:text-blue-200">
                  Read about our story →
                </Link>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Get Started */}
        <section className="mb-8 text-center">
          <Card className="mx-auto max-w-4xl border border-purple-500/30 bg-gradient-to-r from-purple-500/10 to-blue-500/10 shadow-lg">
            <CardContent className="py-8 space-y-4">
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">Get Started</h2>
              <p className="text-muted-foreground text-lg">
                Install RunMat, try the browser sandbox, or explore the open-source runtime.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Button
                  size="lg"
                  asChild
                  className="h-12 px-8 text-base font-semibold bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-xl border-0 transition-all duration-200"
                >
                  <Link href="/docs/getting-started">Install</Link>
                </Button>
                <Button
                  size="lg"
                  asChild
                  className="h-12 px-8 text-base font-semibold bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-xl border-0 transition-all duration-200"
                >
                  <Link href="/matlab-online">Try in Browser</Link>
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  asChild
                  className="h-12 px-8 text-base bg-[#0E1B1E] dark:bg-[#0E1B1E] border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100"
                >
                  <Link href="https://github.com/runmat-org/runmat">GitHub</Link>
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  asChild
                  className="h-12 px-8 text-base bg-[#0E1B1E] dark:bg-[#0E1B1E] border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100"
                >
                  <Link href="mailto:hello@runmat.org">Contact</Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </section>
      </div>
    </div>
  );
}
