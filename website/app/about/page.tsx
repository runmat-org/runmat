import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowUpRight } from "lucide-react";
import { SiGithub } from "react-icons/si";

export const metadata: Metadata = {
  title: "About RunMat | Team, Mission, and Vision",
  description:
    "Meet the RunMat team. We're building a fast, open-source runtime for math with GPU acceleration so engineers can run MATLAB-style code in the browser, desktop, or CLI.",
  alternates: { canonical: "https://runmat.org/about" },
  openGraph: {
    type: "website",
    url: "https://runmat.org/about",
    title: "About RunMat | Team, Mission, and Vision",
    description:
      "Meet the RunMat team. We're building a fast, open-source runtime for math with GPU acceleration so engineers can run MATLAB-style code in the browser, desktop, or CLI.",
    images: [
      {
        url: "https://web.runmatstatic.com/julie-nabeel-group-about-page.png",
        width: 1200,
        height: 630,
        alt: "RunMat co-founders Nabeel Allana and Julie Ruiz",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "About RunMat | Team, Mission, and Vision",
    description:
      "Meet the RunMat team. We're building a fast, open-source runtime for math with GPU acceleration so engineers can run MATLAB-style code in the browser, desktop, or CLI.",
  },
};

const jsonLd = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "AboutPage",
      "@id": "https://runmat.org/about#webpage",
      url: "https://runmat.org/about",
      name: "About RunMat | Team, Mission, and Vision",
      description:
        "Meet the RunMat team. We're building a fast, open-source runtime for math with GPU acceleration so engineers can run MATLAB-style code in the browser, desktop, or CLI.",
      inLanguage: "en",
      isPartOf: { "@id": "https://runmat.org/#website" },
      breadcrumb: { "@id": "https://runmat.org/about#breadcrumb" },
      mainEntity: { "@id": "https://runmat.org/#organization" },
      author: { "@id": "https://runmat.org/#organization" },
      publisher: { "@id": "https://runmat.org/#organization" },
    },
    {
      "@type": "Organization",
      "@id": "https://runmat.org/#organization",
      name: "RunMat",
      alternateName: ["RunMat by Dystr", "Dystr"],
      legalName: "Dystr Inc.",
      url: "https://runmat.org",
      logo: {
        "@type": "ImageObject",
        url: "https://runmat.org/runmat-logo.svg",
        caption: "RunMat",
      },
      description:
        "RunMat is a high-performance, open-source runtime for math that lets you run MATLAB-syntax code in the browser, on the desktop, or from the CLI, while getting GPU-speed execution.",
      sameAs: [
        "https://github.com/runmat-org/runmat",
        "https://x.com/runmat_org",
        "https://dystr.com",
      ],
      knowsAbout: [
        "Scientific Computing",
        "High Performance Computing",
        "MATLAB",
        "WebGPU",
        "Compiler Design",
      ],
      contactPoint: {
        "@type": "ContactPoint",
        contactType: "customer support",
        email: "team@runmat.com",
      },
      location: {
        "@type": "Place",
        name: "Seattle, WA and San Francisco, CA",
      },
      founder: [{ "@id": "https://runmat.org/about#nabeel" }, { "@id": "https://runmat.org/about#julie" }],
    },
    {
      "@type": "Person",
      "@id": "https://runmat.org/about#nabeel",
      name: "Nabeel Allana",
      jobTitle: "CEO and Co-founder",
      image: "https://web.runmatstatic.com/julie-nabeel-group-about-page.png",
      url: "https://www.linkedin.com/in/nallana/",
      sameAs: [
        "https://www.linkedin.com/in/nallana/",
        "https://x.com/nabeelallana",
        "https://github.com/nallana",
      ],
      worksFor: { "@id": "https://runmat.org/#organization" },
      alumniOf: [
        { "@type": "Organization", name: "Apple" },
        { "@type": "Organization", name: "Toyota" },
        { "@type": "Organization", name: "BlackBerry" },
      ],
    },
    {
      "@type": "Person",
      "@id": "https://runmat.org/about#julie",
      name: "Julie Ruiz",
      jobTitle: "Co-founder",
      image: "https://web.runmatstatic.com/julie-nabeel-group-about-page.png",
      url: "https://www.linkedin.com/in/julie-ruiz-64b24328/",
      sameAs: ["https://www.linkedin.com/in/julie-ruiz-64b24328/"],
      worksFor: { "@id": "https://runmat.org/#organization" },
    },
    {
      "@type": "BreadcrumbList",
      "@id": "https://runmat.org/about#breadcrumb",
      itemListElement: [
        { "@type": "ListItem", position: 1, name: "Home", item: "https://runmat.org" },
        { "@type": "ListItem", position: 2, name: "About", item: "https://runmat.org/about" },
      ],
    },
  ],
};

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-background">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(jsonLd).replace(/<\//g, "<\\/"),
        }}
      />
      <div className="container mx-auto px-4 md:px-6 lg:px-8 pt-12 md:pt-16 lg:pt-20 pb-16 md:pb-24 lg:pb-32">
        {/* Who We Are */}
        <section className="pb-12 md:pb-16 lg:pb-20">
          <div className="grid gap-10 lg:grid-cols-2 lg:items-center">
            <div className="order-2 lg:order-1 rounded-2xl border border-border/60 bg-muted/40 p-4">
              <div className="relative h-[320px] w-full overflow-hidden rounded-xl border border-border/60 bg-background/40">
                <Image
                  src="https://web.runmatstatic.com/julie-nabeel-group-about-page.png"
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
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">Who we are</h2>
              <p className="text-muted-foreground text-lg">
                RunMat is a fast, open-source runtime that runs MATLAB-syntax code on GPU. We built it because we saw
                the problem firsthand: talented engineers losing time to slow tools, expensive licenses, and workflows
                that couldn&apos;t keep up with modern hardware.
              </p>
              <p className="text-muted-foreground text-lg">
                We started with Dystr and learned a simple thing: engineers don&apos;t want a new platform. They want
                their current workflow to run faster and feel modern. That led us to RunMat.{" "}
                <Link
                  href="/blog/why-we-built-runmat"
                  className="text-foreground underline underline-offset-2 hover:text-primary"
                >
                  Read more about why we built RunMat
                </Link>
                .
              </p>
            </div>
          </div>
        </section>

        {/* Backed By */}
        <section className="py-12 md:py-16 lg:py-20">
          <Card className="mx-auto max-w-4xl border border-border/60 bg-[#0E1421] text-slate-100 shadow-lg">
            <CardContent className="py-8 text-center space-y-4">
              <p className="text-slate-200 text-lg">Backed by</p>
              <div className="mx-auto grid max-w-2xl grid-cols-2 items-center justify-items-center gap-8 rounded-lg border border-white/10 bg-white/90 px-8 py-6 sm:grid-cols-4 dark:bg-white/10">
                <Link href="https://www.longjourney.vc/" target="_blank" rel="noreferrer">
                  <Image
                    src="https://web.runmatstatic.com/VC%20logos/long-journey.png"
                    alt="Long Journey Ventures logo"
                    width={200}
                    height={56}
                    className="h-10 w-auto object-contain sm:h-12 hover:opacity-80 transition-opacity"
                  />
                </Link>
                <Link href="https://omnivl.com/" target="_blank" rel="noreferrer">
                  <Image
                    src="https://web.runmatstatic.com/VC%20logos/omni.png"
                    alt="Omni logo"
                    width={160}
                    height={56}
                    className="h-10 w-auto object-contain sm:h-12 hover:opacity-80 transition-opacity"
                  />
                </Link>
                <Link href="https://www.unpopular.vc/" target="_blank" rel="noreferrer">
                  <Image
                    src="https://web.runmatstatic.com/VC%20logos/unpopular-ventures.png"
                    alt="Unpopular Ventures logo"
                    width={200}
                    height={56}
                    className="h-10 w-auto object-contain sm:h-12 hover:opacity-80 transition-opacity"
                  />
                </Link>
                <Link href="https://femalefoundersfund.com/" target="_blank" rel="noreferrer">
                  <Image
                    src="https://web.runmatstatic.com/VC%20logos/female-founders-fund.png"
                    alt="Female Founders Fund logo"
                    width={220}
                    height={56}
                    className="h-10 w-auto object-contain sm:h-12 hover:opacity-80 transition-opacity"
                  />
                </Link>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Who is RunMat For */}
        <section className="py-16 md:py-24 lg:py-32">
          <div className="grid gap-10 lg:grid-cols-2 lg:items-center">
            <div className="space-y-4 text-center lg:text-left">
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">Who is RunMat for</h2>
              <p className="text-muted-foreground text-lg">
                RunMat is for people who do math on a computer:
              </p>
              <ul className="text-muted-foreground text-lg space-y-1 list-disc list-inside">
                <li>Simulations and control systems</li>
                <li>Signal and image processing</li>
                <li>Finance, risk, and research</li>
              </ul>
              <p className="text-muted-foreground text-lg">
                If you&apos;ve ever felt stuck between MATLAB&apos;s readability and Python&apos;s openness, RunMat
                gives you both, plus real GPU speed.
              </p>
              <p className="text-muted-foreground text-lg">
                Whether you&apos;re migrating off expensive MATLAB licenses, looking for GPU acceleration without
                learning CUDA, or building workflows where AI writes code and humans verify it, RunMat fits.
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

        {/* Easy to Read, Fast to Run */}
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
              <p className="mt-3 text-xs text-muted-foreground">
                Blog:{" "}
                <Link
                  href="/blog/in-defense-of-matlab-whiteboard-style-code"
                  className="underline underline-offset-2 hover:text-foreground"
                >
                  In Defense of MATLAB: Whiteboard-Style Code
                </Link>
              </p>
            </div>
            <div className="order-1 lg:order-2 space-y-4">
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">Easy to read, fast to run</h2>
              <p className="text-muted-foreground text-lg">
                Code generation is speeding up. More code gets written fast, but checking results is still slow.
              </p>
              <p className="text-muted-foreground text-lg">
                Many engineers stay on MATLAB for readability, but deal with licenses and performance limits. Others
                move to Python and gain openness, but lose the math-like syntax that makes code easy to review.
              </p>
              <p className="text-muted-foreground text-lg">
                We think MATLAB-style syntax matters more than ever because it&apos;s easy to read. As AI writes more
                first drafts, humans still need to verify results fast. RunMat is built for that.
              </p>
            </div>
          </div>
        </section>

        {/* Our Mission */}
        <section className="py-16 md:py-24 lg:py-32">
          <Card className="mx-auto max-w-4xl border border-border/60 bg-[#0E1421] shadow-lg">
            <CardContent className="py-8 space-y-3 text-center">
              <h2 className="text-3xl md:text-4xl font-bold text-gray-100">Our mission</h2>
              <p className="text-xl text-gray-100 font-semibold">
                To enable people to do math and physics that they couldn&apos;t do before by creating a modern,
                high-performance alternative to MATLAB.
              </p>
              <p className="text-gray-300 text-lg">
                We focus on performance, portability, and developer experience, without breaking the way engineers
                already write math. That means a runtime that&apos;s fast, secure, and built for today&apos;s hardware.
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
                RunMat is an open-source runtime for math that runs MATLAB-style code in the browser, desktop, or
                CLI, and can automatically run work on CPU or GPU.
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
                <CardTitle className="text-xl text-gray-100">Runtime</CardTitle>
              </CardHeader>
              <CardContent className="text-lg text-gray-300 space-y-2">
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-blue-300">•</span>
                  <p>Automatic GPU acceleration via kernel fusion</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-blue-300">•</span>
                  <p>
                    ~5ms startup (about 180× faster than{" "}
                    <Link
                      href="/blog/introducing-runmat"
                      className="underline underline-offset-2 hover:text-blue-200"
                    >
                      GNU Octave
                    </Link>
                    )
                  </p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-blue-300">•</span>
                  <p>Async support so long work doesn&apos;t lock up your app</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-blue-300">•</span>
                  <p>Built in Rust for portability</p>
                </div>
              </CardContent>
            </Card>
            <Card className="border border-green-500/30 bg-[#0E1421] shadow-lg">
              <CardHeader className="border-b border-border/60">
                <CardTitle className="text-xl text-gray-100">Platform</CardTitle>
              </CardHeader>
              <CardContent className="text-lg text-gray-300 space-y-2">
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Browser sandbox: run code instantly, no install</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Fast plotting and visualization (improving quickly)</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Language tools (LSP + linter) for autocomplete and error checks</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Clear errors that help you fix issues fast</p>
                </div>
              </CardContent>
            </Card>
            <Card className="border border-purple-500/30 bg-[#0E1421] shadow-lg">
              <CardHeader className="border-b border-border/60">
                <CardTitle className="text-xl text-gray-100">Enterprise</CardTitle>
              </CardHeader>
              <CardContent className="text-lg text-gray-300 space-y-3">
                <p>
                  <span className="font-medium text-gray-100">Available now:</span> air-gapped deploys,
                  local-first runs on your hardware.
                </p>
                <p>
                  <span className="font-medium text-gray-100">Coming very soon:</span> saved projects, team
                  orgs/projects, roles + SSO, large-file storage with resumable uploads, and version history.
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Where We're Headed */}
        <section className="py-12 md:py-16 lg:py-20">
          <div className="mx-auto max-w-5xl text-center mb-12 space-y-3">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">Where we&apos;re headed</h2>
          </div>
          <div className="mx-auto max-w-5xl rounded-2xl border border-slate-200 bg-white p-8 shadow-lg space-y-6 dark:border-purple-500/30 dark:bg-gradient-to-br dark:from-purple-500/15 dark:via-[#0E1421] dark:to-[#0A0F1C]">
            <p className="text-lg text-slate-700 dark:text-gray-300">
              You can use RunMat today as a MATLAB-style runtime. Next, we&apos;re building it into a modern
              environment for engineers where:
            </p>
            <div className="space-y-3 text-lg text-slate-700 dark:text-gray-300">
              <div className="flex items-start gap-3">
                <ArrowUpRight className="mt-1 h-4 w-4 text-blue-600 dark:text-blue-300 shrink-0" />
                <span>You write or generate code faster</span>
              </div>
              <div className="flex items-start gap-3">
                <ArrowUpRight className="mt-1 h-4 w-4 text-blue-600 dark:text-blue-300 shrink-0" />
                <span>You verify results faster</span>
              </div>
              <div className="flex items-start gap-3">
                <ArrowUpRight className="mt-1 h-4 w-4 text-blue-600 dark:text-blue-300 shrink-0" />
                <span>You run work from laptop to cloud to secure enterprise setups</span>
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-5 text-lg text-slate-700 dark:border-border/60 dark:bg-white/5 dark:text-gray-300">
              <span className="font-semibold text-slate-900 dark:text-gray-100">The goal:</span> Let engineers focus on math and physics, not programming.
            </div>
          </div>
        </section>

        {/* Open Source */}
        <section className="py-12 md:py-16 lg:py-20">
          <div className="mx-auto max-w-3xl text-center">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-8">Open source at the core</h2>
            <Card className="border border-border/60 bg-[#0E1421] shadow-lg">
              <CardContent className="py-8 space-y-4">
                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full border border-border/60 bg-background/30 text-sm text-gray-400">
                  <SiGithub className="h-7 w-7" />
                </div>
                <p className="text-gray-300 text-lg">
                  The core runtime is MIT licensed and on{" "}
                  <Link href="https://github.com/runmat-org/runmat" className="underline text-blue-300 hover:text-blue-200">
                    GitHub
                  </Link>
                  . The JIT, the fusion engine, the GPU planner. We&apos;re committed to keeping it open source and
                  actively maintained.
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Get Started */}
        <section className="py-12 md:py-16 lg:py-20 text-center">
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

        {/* Contact */}
        <section className="pb-8 md:pb-12 lg:pb-16 text-center">
          <p className="text-sm text-muted-foreground">
            We&apos;re based in Seattle, WA and San Francisco, CA. Reach us at{" "}
            <a href="mailto:team@runmat.com" className="underline hover:text-foreground">
              team@runmat.com
            </a>
          </p>
        </section>
      </div>
    </div>
  );
}
