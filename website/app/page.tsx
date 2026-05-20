import type { Metadata } from "next";
import Link from "next/link";

import Hero from "@/components/Hero";
import Media from "@/components/Media";
import { Button } from "@/components/ui/button";

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
        "caption": "RunMat",
      },
      "description":
        "RunMat is a GPU-first platform for engineering math with MATLAB syntax, real-time visualization, and an agent that understands runtime state.",
      "sameAs": [
        "https://github.com/runmat-org/runmat",
        "https://x.com/runmat_com",
        "https://dystr.com",
      ],
      "knowsAbout": [
        "Scientific Computing",
        "High Performance Computing",
        "MATLAB",
        "WebGPU",
        "Compiler Design",
      ],
      "contactPoint": {
        "@type": "ContactPoint",
        "contactType": "customer support",
        "email": "team@runmat.com",
      },
    },
    {
      "@type": "WebSite",
      "@id": "https://runmat.com/#website",
      "url": "https://runmat.com",
      "name": "RunMat",
      "description": "A GPU-first platform for engineering math.",
      "publisher": { "@id": "https://runmat.com/#organization" },
      "potentialAction": {
        "@type": "SearchAction",
        "target": {
          "@type": "EntryPoint",
          "urlTemplate": "https://runmat.com/search?q={search_term_string}",
        },
        "query-input": "required name=search_term_string",
      },
    },
    {
      "@type": "SoftwareApplication",
      "@id": "https://runmat.com/#software",
      "name": "RunMat",
      "description":
        "RunMat runs MATLAB syntax workloads with real-time feedback, instant visualization, and GPU acceleration across desktop, browser, and CLI surfaces.",
      "license": "https://opensource.org/licenses/MIT",
      "applicationCategory": "ScientificApplication",
      "applicationSubCategory": "Numerical Analysis & Simulation",
      "operatingSystem": ["Windows", "macOS", "Linux", "Browser"],
      "softwareVersion": "Beta",
      "featureList": [
        "MATLAB syntax workloads",
        "Cross-platform GPU support",
        "Real-time plotting",
        "Runtime-aware agent workflows",
        "Desktop, browser, and CLI surfaces",
      ],
      "offers": {
        "@type": "Offer",
        "price": "0",
        "priceCurrency": "USD",
        "availability": "https://schema.org/InStock",
      },
      "author": { "@id": "https://runmat.com/#organization" },
      "publisher": { "@id": "https://runmat.com/#organization" },
      "downloadUrl": "https://runmat.com/download",
      "mainEntityOfPage": { "@id": "https://runmat.com/#website" },
    },
  ],
};

export const metadata: Metadata = {
  title: "RunMat: GPU-First Platform for Engineering Math",
  description:
    "Run MATLAB syntax workloads with real-time feedback, instant visualization, and an agent that understands your runtime state.",
  keywords: [
    "run matlab online",
    "free matlab runtime",
    "matlab alternative",
    "gpu engineering math",
    "scientific computing",
    "matlab ai assistant",
    "matlab cli",
    "matlab plotting",
  ],
  alternates: { canonical: "/" },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
  openGraph: {
    title: "RunMat: GPU-First Platform for Engineering Math",
    description:
      "Run MATLAB syntax workloads with real-time feedback and an agent that understands your runtime state.",
    url: "/",
    siteName: "RunMat",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "RunMat: GPU-First Platform for Engineering Math",
    description:
      "Run MATLAB syntax workloads with real-time feedback, instant visualization, and an agent that understands your runtime state.",
  },
};

const hero = {
  title: "Run math blazing fast",
  description:
    "Run MATLAB syntax workloads with real-time feedback and GPU acceleration across desktop, browser, and CLI.",
  primaryCta: {
    label: "Download for macOS",
    href: "/download",
  },
  secondaryCta: {
    label: "Open Browser Sandbox",
    href: "/sandbox",
  },
  supportingLinks: {
    prefix: "View",
    links: [
      {
        label: "CLI",
        href: "/docs/cli",
      },
      {
        label: "other download options",
        href: "/download",
      },
    ],
    suffix: ".",
  },
  media: {
    label: "Hero product media",
    note: "Final app/browser graphic goes here.",
    tone: "surface" as const,
  },
};

const features = [
  {
    title: "Open source, high performance GPU runtime",
    body: (
      <>
        Execute MATLAB denominated code with a modern compiler architecture and cross-platform GPU support.{" "}
        <Link href="/runtime" className="underline underline-offset-4">
          Learn more.
        </Link>
      </>
    ),
    mediaLabel: "Runtime code media",
    mediaNote: "MATLAB code snippet with a matrix math error underline. Put logos for Apple, NVIDIA and ARM GPU under.",
    tone: "muted" as const,
  },
  {
    title: "Run complex simulations effortlessly",
    body: "RunMat's plotting pipeline is a series of camera and projection transforms on your existing tensors in GPU memory. Linearly independent operation chains are dispatched to parallel GPU cores for execution to saturate your hardware's bandwidth.",
    mediaLabel: "Simulation media",
    mediaNote: "Showing a 3d simulation of raindrops on a surface of water.",
    tone: "brand" as const,
  },
  {
    title: "Designed for the agent era",
    body: (
      <>
        RunMat is designed for agent-in-the-loop engineering. Agents in RunMat
        create atomic, reversible edits, and can see plots and runtime variables
        in real time as they run, allowing you to complete weeks of permutations
        in days.{" "}
        <Link href="/agent" className="underline underline-offset-4">
          Learn more.
        </Link>
      </>
    ),
    mediaLabel: "Agent workflow media",
    mediaNote: "Agent working with code, plots, and runtime state.",
    tone: "muted" as const,
  },
  //{
  //  title: "Model agnostic, platform agnostic",
  //  body:
  //    "RunMat works with any model for inference. Use the latest frontier models from OpenAI, Claude or Gemini, or mix and match as you work.",
  //  mediaLabel: "Model provider media",
  //  mediaNote: "Provider and platform options.",
  //  tone: "surface" as const,
  //},
  {
    title: "Made for modern engineering challenges",
    body:
      "Inspect previous run and variable history, revert to any previous edit version, and collaborate in real time.",
    mediaLabel: "Engineering history media",
    mediaNote: "Run history, variable history, and collaboration.",
    tone: "brand" as const,
  },
];

const surfaces = [
  {
    title: "Start in the desktop app",
    cta: "Download for macOS",
    href: "/download",
    mediaLabel: "Desktop app media",
  },
  {
    title: "Open browser sandbox",
    cta: "Try in your browser",
    href: "/sandbox",
    mediaLabel: "Browser sandbox media",
  },
  {
    title: "Use the CLI",
    cta: "curl -fsSL runmat.com/install | sh",
    href: "/docs/cli",
    mediaLabel: "CLI media",
  },
];

export default function HomePage() {
  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground home-page">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(jsonLd).replace(/<\//g, "<\\/"),
        }}
      />

      <Hero {...hero} />

      <section className="w-full bg-background py-20 md:py-28">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto max-w-6xl space-y-28 md:space-y-36">
            {features.map((feature, index) => (
              <div
                key={feature.title}
                className="grid items-center gap-10 md:grid-cols-12 md:gap-12"
              >
                <div
                  className={
                    index % 2 === 0
                      ? "md:col-span-4"
                      : "md:col-span-4 md:col-start-9"
                  }
                >
                  <h2 className="text-3xl font-medium tracking-tight text-foreground md:text-4xl">
                    {feature.title}
                  </h2>
                  <p className="mt-5 text-sm leading-6 text-muted-foreground">
                    {feature.body}
                  </p>
                </div>
                <div
                  className={
                    index % 2 === 0
                      ? "md:col-span-8"
                      : "md:col-span-8 md:col-start-1 md:row-start-1"
                  }
                >
                  <Media
                    label={feature.mediaLabel}
                    note={feature.mediaNote}
                    tone={feature.tone}
                    className="min-h-[360px] md:min-h-[430px]"
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="w-full bg-background py-20 md:py-28">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto max-w-3xl text-center">
            <h2 className="text-4xl font-medium tracking-tight text-foreground md:text-5xl">
              The same RunMat everywhere you run math
            </h2>
            <p className="mt-5 text-sm leading-6 text-muted-foreground">
              RunMat is available as rust source code, a CLI, a desktop app, or
              in your web browser.
            </p>
          </div>

          <div className="mx-auto mt-12 grid max-w-6xl gap-5 md:grid-cols-3">
            {surfaces.map((surface) => (
              <Link
                key={surface.title}
                href={surface.href}
                className="group rounded-2xl border border-border bg-card p-3 transition-colors hover:bg-accent"
              >
                <Media
                  label={surface.mediaLabel}
                  tone="muted"
                  className="min-h-[210px] rounded-xl p-4"
                />
                <div className="px-2 pb-3 pt-5">
                  <h3 className="text-lg font-medium text-card-foreground">
                    {surface.title}
                  </h3>
                  <div className="mt-5 rounded-full bg-primary px-4 py-2 text-center text-xs font-medium text-primary-foreground transition-opacity group-hover:opacity-90">
                    {surface.cta}
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      <section className="w-full bg-background py-20 md:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto max-w-3xl text-center">
            <h2 className="text-4xl font-medium tracking-tight text-foreground md:text-6xl">
              Try RunMat today
            </h2>
            <p className="mt-5 text-base text-muted-foreground">
              GPU-first platform for engineering math.
            </p>
            <div className="mt-8">
              <Button
                size="lg"
                asChild
                className="h-11 rounded-full bg-primary px-7 text-sm font-medium text-primary-foreground shadow-none hover:bg-primary/90"
              >
                <Link href="/download">Download for macOS</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
