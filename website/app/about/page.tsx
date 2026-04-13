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
  alternates: { canonical: "https://runmat.com/about" },
  openGraph: {
    type: "website",
    url: "https://runmat.com/about",
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
      "@id": "https://runmat.com/about#webpage",
      url: "https://runmat.com/about",
      name: "About RunMat | Team, Mission, and Vision",
      description:
        "Meet the RunMat team. We're building a fast, open-source runtime for math with GPU acceleration so engineers can run MATLAB-style code in the browser, desktop, or CLI.",
      inLanguage: "en",
      isPartOf: { "@id": "https://runmat.com/#website" },
      breadcrumb: { "@id": "https://runmat.com/about#breadcrumb" },
      mainEntity: { "@id": "https://runmat.com/#organization" },
      author: { "@id": "https://runmat.com/#organization" },
      publisher: { "@id": "https://runmat.com/#organization" },
    },
    {
      "@type": "Organization",
      "@id": "https://runmat.com/#organization",
      name: "RunMat",
      alternateName: ["RunMat by Dystr", "Dystr"],
      legalName: "Dystr Inc.",
      foundingDate: "2022",
      url: "https://runmat.com",
      logo: {
        "@type": "ImageObject",
        url: "https://runmat.com/runmat-logo.svg",
        caption: "RunMat",
      },
      description:
        "RunMat is a high-performance, open-source runtime for math that lets you run MATLAB-syntax code in the browser, on the desktop, or from the CLI, while getting GPU-speed execution.",
      sameAs: [
        "https://github.com/runmat-org/runmat",
        "https://x.com/runmat_com",
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
      founder: [{ "@id": "https://runmat.com/about#nabeel" }, { "@id": "https://runmat.com/about#julie" }],
    },
    {
      "@type": "Person",
      "@id": "https://runmat.com/about#nabeel",
      name: "Nabeel Allana",
      jobTitle: "CEO and Co-founder",
      image: "https://web.runmatstatic.com/julie-nabeel-group-about-page.png",
      url: "https://www.linkedin.com/in/nallana/",
      sameAs: [
        "https://www.linkedin.com/in/nallana/",
        "https://x.com/nabeelallana",
        "https://github.com/nallana",
      ],
      worksFor: { "@id": "https://runmat.com/#organization" },
      alumniOf: [
        { "@type": "Organization", name: "Apple" },
        { "@type": "Organization", name: "Toyota" },
        { "@type": "Organization", name: "BlackBerry" },
      ],
    },
    {
      "@type": "Person",
      "@id": "https://runmat.com/about#julie",
      name: "Julie Ruiz",
      jobTitle: "Co-founder",
      image: "https://web.runmatstatic.com/julie-nabeel-group-about-page.png",
      url: "https://www.linkedin.com/in/julie-ruiz-64b24328/",
      sameAs: ["https://www.linkedin.com/in/julie-ruiz-64b24328/"],
      worksFor: { "@id": "https://runmat.com/#organization" },
    },
    {
      "@type": "SoftwareApplication",
      "@id": "https://runmat.com/#software",
      name: "RunMat",
      applicationCategory: "ScientificApplication",
      operatingSystem: "macOS, Linux, Windows, Web (WebAssembly)",
      url: "https://runmat.com",
      downloadUrl: "https://runmat.com/download",
      offers: {
        "@type": "Offer",
        price: "0",
        priceCurrency: "USD",
      },
      featureList: [
        "300+ core MATLAB built-in functions",
        "Cross-vendor GPU acceleration (Metal, Vulkan, DirectX 12, WebGPU)",
        "Automatic GPU kernel fusion engine",
        "Client-side browser sandbox via WebAssembly",
        "Interactive 2D and 3D GPU-rendered plotting",
        "Real-time type and shape tracking",
        "Execution tracing and structured logging",
        "Automatic file versioning and project snapshots",
        "Team collaboration with shared project state",
        "Air-gapped single-binary deployment",
      ],
      author: { "@id": "https://runmat.com/#organization" },
      publisher: { "@id": "https://runmat.com/#organization" },
    },
    {
      "@type": "BreadcrumbList",
      "@id": "https://runmat.com/about#breadcrumb",
      itemListElement: [
        { "@type": "ListItem", position: 1, name: "Home", item: "https://runmat.com" },
        { "@type": "ListItem", position: 2, name: "About", item: "https://runmat.com/about" },
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
        <h1 className="sr-only">About RunMat</h1>
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
              <p className="mt-3 text-sm text-muted-foreground">
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
                We&apos;re a small team with deep roots in engineering and enterprise software who spent years
                watching talented people lose time to slow tools and expensive licenses. We started as Dystr in 2022 and learned one thing: engineers
                don&apos;t want a new platform. They want their current workflow to run faster.
              </p>
              <p className="text-muted-foreground text-lg">
                That led us to RunMat.{" "}
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
          <Card className="mx-auto max-w-4xl border border-border/60 bg-[#0E1421] text-gray-100 shadow-lg">
            <CardContent className="py-8 text-center space-y-4">
              <p className="text-gray-300 text-lg">Backed by</p>
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
                Whether you&apos;re migrating off expensive MATLAB licenses or looking for GPU acceleration
                without learning CUDA, RunMat fits.
              </p>
            </div>
            <div className="rounded-2xl border border-border/60 bg-muted/40 p-4">
              <Link href="/sandbox" className="block rounded-xl overflow-hidden focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2">
                <video
                  className="w-full h-auto rounded-xl"
                  autoPlay
                  muted
                  loop
                  playsInline
                  preload="none"
                  poster="https://web.runmatstatic.com/video/posters/3d-interactive-plotting-runmat.webp"
                  aria-label="RunMat 3D interactive plotting demo"
                >
                  <source src="https://web.runmatstatic.com/video/3d-interactive-plotting-runmat.mp4" type="video/mp4" />
                </video>
              </Link>
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
              <p className="mt-3 text-sm text-muted-foreground">
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
                We focus on performance and portability without breaking the way engineers already write math.
                That means a runtime that&apos;s fast, secure, and runs on today&apos;s hardware out of the box.
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
                RunMat is an open-source runtime that executes MATLAB-syntax code (.m files) in the browser, on the desktop, and from the CLI. It covers 300+ core built-in functions with cross-vendor GPU acceleration via Metal, Vulkan, DirectX 12, and WebGPU, though it does not replicate Simulink or specialized toolboxes. The browser sandbox runs entirely client-side via WebAssembly. No account needed, no install, nothing leaves your machine.
              </p>
            </div>
          </section>
          <Card className="mx-auto max-w-2xl border border-border/60 bg-[#0E1421] shadow-lg mb-12">
            <CardContent className="py-6 md:py-8">
              <h3 className="text-lg font-bold text-gray-100 mb-5">Product milestones</h3>
              <div className="relative border-l-2 border-white/10 ml-1.5 space-y-5 pl-6">
                <div className="relative">
                  <span className="absolute -left-[calc(1.5rem+5px)] top-1 h-2.5 w-2.5 rounded-full bg-blue-500" />
                  <p className="text-sm font-semibold text-blue-400">Aug 2025</p>
                  <p className="text-sm font-semibold text-gray-100">CLI Launch</p>
                  <p className="text-sm text-gray-300">
                    ~5 ms startup, 150–180× faster than GNU Octave.{" "}
                    <Link href="/blog/introducing-runmat" className="underline underline-offset-2 hover:text-gray-100">Read more</Link>
                  </p>
                </div>
                <div className="relative">
                  <span className="absolute -left-[calc(1.5rem+5px)] top-1 h-2.5 w-2.5 rounded-full bg-green-500" />
                  <p className="text-sm font-semibold text-green-400">Nov 2025</p>
                  <p className="text-sm font-semibold text-gray-100">RunMat Accelerate</p>
                  <p className="text-sm text-gray-300">
                    Automatic GPU fusion — 82× faster than PyTorch on 1B-point math.{" "}
                    <Link href="/blog/runmat-accelerate-fastest-runtime-for-your-math" className="underline underline-offset-2 hover:text-gray-100">Read more</Link>
                  </p>
                </div>
                <div className="relative">
                  <span className="absolute -left-[calc(1.5rem+5px)] top-1 h-2.5 w-2.5 rounded-full bg-amber-500" />
                  <p className="text-sm font-semibold text-amber-400">Mar 2026</p>
                  <p className="text-sm font-semibold text-gray-100">Browser Sandbox + RunMat Cloud</p>
                  <p className="text-sm text-gray-300">
                    Client-side WebAssembly execution, persistent projects, versioning, and collaboration.{" "}
                    <Link href="/blog/introducing-runmat-cloud" className="underline underline-offset-2 hover:text-gray-100">Read more</Link>
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
          <div className="mx-auto grid max-w-6xl gap-6 md:grid-cols-2 lg:grid-cols-4">
            <Card className="border border-blue-500/30 bg-[#0E1421] shadow-lg">
              <CardHeader className="border-b border-border/60">
                <CardTitle className="text-xl text-gray-100">Runtime</CardTitle>
              </CardHeader>
              <CardContent className="text-lg text-gray-300 space-y-2">
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-blue-300">•</span>
                  <p>Fusion engine auto-offloads op chains to GPU. No kernel code, no rewrites</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-blue-300">•</span>
                  <p>Cross-vendor GPU: Metal, DirectX 12, Vulkan, WebGPU. Works on any hardware</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-blue-300">•</span>
                  <p>Ignition interpreter for ~5 ms startup; Turbine JIT (Cranelift) for hot paths</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-blue-300">•</span>
                  <p>Async-capable. GPU readback and long scripts never block the host</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-blue-300">•</span>
                  <p>300+ built-in functions with a generational GC and memory-safe Rust core</p>
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
                  <p>Interactive 2D and 3D plotting with GPU-rendered surfaces (rotate, zoom, pan)</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Language tools (LSP + linter) for autocomplete and error checks</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Clear errors that help you fix issues fast</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Real-time type and shape tracking with live syntax validation</p>
                </div>
              </CardContent>
            </Card>
            <Card className="border border-amber-500/30 bg-[#0E1421] shadow-lg">
              <CardHeader className="border-b border-border/60">
                <CardTitle className="text-xl text-gray-100">RunMat Cloud</CardTitle>
              </CardHeader>
              <CardContent className="text-lg text-gray-300 space-y-2">
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-amber-300">•</span>
                  <p>Persistent projects with automatic run history</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-amber-300">•</span>
                  <p>Automatic file versioning and project snapshots</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-amber-300">•</span>
                  <p>Team collaboration with shared project state</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-amber-300">•</span>
                  <p>Chunked, content-addressed large-file storage</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-amber-300">•</span>
                  <p>
                    Free Hobby tier &middot;{" "}
                    <Link href="/pricing" className="underline underline-offset-2 hover:text-amber-200">Pro &amp; Team tiers</Link>
                  </p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-amber-300">•</span>
                  <p>
                    <Link href="/blog/introducing-runmat-cloud" className="underline underline-offset-2 hover:text-amber-200">Learn more</Link>
                  </p>
                </div>
              </CardContent>
            </Card>
            <Card className="border border-purple-500/30 bg-[#0E1421] shadow-lg">
              <CardHeader className="border-b border-border/60">
                <CardTitle className="text-xl text-gray-100">Enterprise</CardTitle>
              </CardHeader>
              <CardContent className="text-lg text-gray-300 space-y-2">
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-purple-300">•</span>
                  <p>Air-gapped, on-prem deployment as a single binary</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-purple-300">•</span>
                  <p>Built for ITAR and strict data-residency requirements</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-purple-300">•</span>
                  <p>SOC 2 audit-ready architecture</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-purple-300">•</span>
                  <p>SSO / SAML and SCIM provisioning</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-purple-300">•</span>
                  <p>Audit trails for compliance</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-purple-300">•</span>
                  <p>
                    <Link href="/blog/mission-critical-math-airgap" className="underline underline-offset-2 hover:text-purple-200">Full platform inside your airgap</Link>
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Where We're Headed */}
        <section className="py-10 md:py-12 lg:py-16">
          <div className="mx-auto max-w-5xl text-center mb-8 space-y-2">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">Where we&apos;re headed</h2>
          </div>
          <Card className="mx-auto max-w-4xl rounded-xl border border-border/60 bg-[#0E1421] shadow-lg">
            <CardContent className="py-6 md:py-8 space-y-4">
            <p className="text-lg text-gray-300">
              You can use RunMat today as a MATLAB-style runtime. Next, we&apos;re building it into a modern
              environment for engineers where:
            </p>
            <div className="space-y-2 text-lg text-gray-300">
              <div className="flex items-start gap-3">
                <ArrowUpRight className="mt-1 h-4 w-4 text-blue-300 shrink-0" />
                <span>Write or generate code faster</span>
              </div>
              <div className="flex items-start gap-3">
                <ArrowUpRight className="mt-1 h-4 w-4 text-blue-300 shrink-0" />
                <span>Verify results faster</span>
              </div>
              <div className="flex items-start gap-3">
                <ArrowUpRight className="mt-1 h-4 w-4 text-blue-300 shrink-0" />
                <span>Run work on a laptop, in the cloud, or inside a secure enterprise setup</span>
              </div>
            </div>
            <div className="rounded-xl border border-white/10 bg-white/5 p-4 text-lg text-gray-300">
              <span className="font-semibold text-gray-100">The goal:</span>{" "}
              <span>Let engineers focus on math and physics, not programming.</span>
            </div>
            </CardContent>
          </Card>
        </section>

        {/* Open Source */}
        <section className="py-12 md:py-16 lg:py-20">
          <div className="mx-auto max-w-3xl text-center">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-8">Open source at the core</h2>
            <Card className="border border-border/60 bg-[#0E1421] shadow-lg">
              <CardContent className="py-8 space-y-4">
                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full border border-border/60 bg-background/30 text-sm text-gray-300">
                  <SiGithub className="h-7 w-7" />
                </div>
                <p className="text-gray-300 text-lg">
                  The core runtime is MIT licensed and on{" "}
                  <Link href="https://github.com/runmat-org/runmat" className="underline text-blue-300 hover:text-blue-200">
                    GitHub
                  </Link>
                  . That includes the JIT, the fusion engine, and the GPU planner. We&apos;re committed to
                  keeping it open source and actively maintained.
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
                    Launch the sandbox
                  </Link>
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  asChild
                  className="h-12 px-8 text-base bg-[#0E1421] dark:bg-[#0E1421] border-border/60 text-gray-100"
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
            We&apos;re based in Seattle, WA and San Francisco, CA.{" "}
            <Link href="/contact" className="underline hover:text-foreground">
              Get in touch
            </Link>{" "}
            or email us at{" "}
            <a href="mailto:team@runmat.com" className="underline hover:text-foreground">
              team@runmat.com
            </a>
          </p>
        </section>
      </div>
    </div>
  );
}
