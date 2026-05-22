import type { Metadata } from "next";
import Link from "next/link";

import Hero from "@/components/Hero";
import LazyVideo from "@/components/LazyVideo";
import { Button } from "@/components/ui/button";
import { DetectedDownloadLabel } from "@/components/DetectedDownloadLabel";
import { CARD_PATTERNS } from "@/components/card-patterns";
import { cn } from "@/lib/utils";

const heroPosterSrc = "https://web.runmatstatic.com/video/posters/3D-wave-surface-runmat.webp";
const heroVideoSrc = "https://web.runmatstatic.com/video/3D-wave-surface-runmat.mp4";

const jsonLd = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "WebPage",
      "@id": "https://runmat.com/#webpage",
      "url": "https://runmat.com",
      "name": "RunMat: GPU-First Platform for Engineering Math",
      "description":
        "Run MATLAB syntax workloads with real-time feedback, instant visualization, and an agent that understands your runtime state.",
      "isPartOf": { "@id": "https://runmat.com/#website" },
      "about": { "@id": "https://runmat.com/#software" },
      "primaryImageOfPage": {
        "@type": "ImageObject",
        "url": heroPosterSrc,
      },
      "video": { "@id": "https://runmat.com/#hero-video" },
    },
    {
      "@type": "VideoObject",
      "@id": "https://runmat.com/#hero-video",
      "name": "RunMat 3D wave surface simulation",
      "description":
        "RunMat renders and iterates on a 3D wave surface simulation with runtime-aware engineering feedback.",
      "thumbnailUrl": heroPosterSrc,
      "contentUrl": heroVideoSrc,
      "uploadDate": "2026-05-22T00:00:00Z",
      "publisher": { "@id": "https://runmat.com/#organization" },
      "mainEntityOfPage": { "@id": "https://runmat.com/#webpage" },
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
    images: [
      {
        url: heroPosterSrc,
        width: 3840,
        height: 2156,
        alt: "RunMat 3D wave surface simulation",
      },
    ],
    videos: [
      {
        url: heroVideoSrc,
        type: "video/mp4",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "RunMat: GPU-First Platform for Engineering Math",
    description:
      "Run MATLAB syntax workloads with real-time feedback, instant visualization, and an agent that understands your runtime state.",
    images: [heroPosterSrc],
  },
};

const hero = {
  title: "Run math blazing fast",
  description:
    "Run MATLAB syntax workloads with real-time feedback and GPU acceleration across desktop, browser, and CLI.",
  primaryCta: {
    label: <DetectedDownloadLabel />,
    href: "/download/latest",
  },
  secondaryCta: {
    label: "Open Browser Sandbox",
    href: "/sandbox",
  },
  supportingLinks: {
    prefix: "View",
    links: [
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
    poster: heroPosterSrc,
    video: heroVideoSrc,
  },
};

const features = [
  {
    title: "Open source, high performance GPU runtime",
    body: (
      <>
        Execute MATLAB-syntax code with a modern compiler architecture and cross-platform GPU support.{" "}
        <Link href="/docs/how-it-works" className="underline underline-offset-4">
          Learn more.
        </Link>
      </>
    ),
    mediaLabel: "Runtime code media",
    mediaNote: "MATLAB code snippet with a matrix math error underline. Put logos for Apple, NVIDIA and ARM GPU under.",
    mediaImage: "https://web.runmatstatic.com/gpu-mistmatch.webp",
    tone: "muted" as const,
  },
  {
    title: "Run complex simulations effortlessly",
    body: (
      <>
        RunMat&apos;s plotting pipeline is a series of camera and projection
        transforms on your existing tensors in GPU memory. Linearly independent
        operation chains are dispatched to parallel GPU cores for execution to
        saturate your hardware&apos;s bandwidth.{" "}
        <Link href="/docs/plotting" className="underline underline-offset-4">
          Learn more.
        </Link>
      </>
    ),
    mediaLabel: "Simulation media",
    mediaNote: "Showing a 3d simulation of raindrops on a surface of water.",
    mediaPoster: "https://web.runmatstatic.com/video/posters/runmat-wave-simulation-homepage.webp",
    mediaVideo: "https://web.runmatstatic.com/video/runmat-wave-simulation-homepage.mp4",
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
        <Link href="/matlab-ai-agent" className="underline underline-offset-4">
          Learn more.
        </Link>
      </>
    ),
    mediaLabel: "Agent workflow media",
    mediaNote: "Agent working with code, plots, and runtime state.",
    mediaImage: "https://web.runmatstatic.com/runmat-agent-diff-rain.webp",
    tone: "muted" as const,
  },
  {
    title: "Made for modern engineering challenges",
    body: (
      <>
        Inspect previous run and variable history, revert to any previous edit
        version, and collaborate in real time.{" "}
        <Link href="/docs/versioning" className="underline underline-offset-4">
          Learn more.
        </Link>
      </>
    ),
    mediaLabel: "Engineering history media",
    mediaNote: "Run history, variable history, and collaboration.",
    mediaPoster: "https://web.runmatstatic.com/video/posters/fft-run-versions.webp",
    mediaVideo: "https://web.runmatstatic.com/video/fft-run-versions.mp4",
    tone: "brand" as const,
  },
] as const;

const surfaces = [
  {
    title: "Download the desktop app",
    cta: <DetectedDownloadLabel />,
    href: "/download/latest",
    mediaLabel: "Desktop app media",
    mediaIndex: 2,
    icon: "desktop",
  },
  {
    title: "Open browser sandbox",
    cta: "Try in your browser",
    href: "/sandbox",
    mediaLabel: "Browser sandbox media",
    mediaIndex: 10,
    icon: "browser",
  },
  {
    title: "Install the CLI",
    cta: "curl -fsSL runmat.com/install | sh",
    href: "/docs/cli",
    mediaLabel: "CLI media",
    mediaIndex: 5,
    icon: "cli",
  },
] as const;

const mediaToneClasses = {
  muted: "bg-muted text-foreground",
  brand: "bg-brand/15 text-foreground",
  surface: "bg-card text-card-foreground",
} as const;

function SurfaceIcon({ type }: { type: "desktop" | "browser" | "cli" }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 64 64"
      className="absolute left-1/2 top-1/2 z-10 h-16 w-16 -translate-x-1/2 -translate-y-1/2 text-gray-800/50 dark:text-gray-200/50"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {type === "desktop" ? (
        <>
          <rect x="10" y="12" width="44" height="30" rx="4" />
          <path d="M24 52h16M28 42v10M36 42v10" />
          <path d="M18 21h28M18 29h18" />
        </>
      ) : null}
      {type === "browser" ? (
        <>
          <rect x="9" y="12" width="46" height="38" rx="5" />
          <path d="M9 23h46" />
          <circle cx="18" cy="18" r="1.5" />
          <circle cx="24" cy="18" r="1.5" />
          <path d="M19 38c5-13 10-18 16-15 5 2 8 8 11 15" />
          <path d="M18 42h28" />
        </>
      ) : null}
      {type === "cli" ? (
        <>
          <rect x="9" y="14" width="46" height="36" rx="5" />
          <path d="M18 28l8 6-8 6" />
          <path d="M32 40h14" />
        </>
      ) : null}
    </svg>
  );
}

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
                className="grid items-center gap-10 2xl:grid-cols-12 2xl:gap-12"
              >
                <div
                  className={
                    index % 2 === 0
                      ? "max-w-3xl 2xl:col-span-4 2xl:max-w-none"
                      : "max-w-3xl 2xl:col-span-4 2xl:col-start-9 2xl:max-w-none"
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
                      ? "2xl:col-span-8"
                      : "2xl:col-span-8 2xl:col-start-1 2xl:row-start-1"
                  }
                >
                  <div
                    role="img"
                    aria-label={feature.mediaLabel}
                    className={cn(
                      "relative aspect-[16/9] w-full overflow-hidden rounded-2xl border border-border 2xl:aspect-auto 2xl:min-h-[430px]",
                      mediaToneClasses[feature.tone],
                    )}
                  >
                    {"mediaVideo" in feature ? (
                      <LazyVideo
                        className="absolute inset-0 h-full w-full object-cover"
                        muted
                        loop
                        playsInline
                        deferPosterUntilVisible
                        initialPosterVariant="poster"
                        poster={feature.mediaPoster}
                        aria-label={feature.mediaLabel}
                      >
                        <source src={feature.mediaVideo} type="video/mp4" />
                      </LazyVideo>
                    ) : (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={feature.mediaImage}
                        alt={feature.mediaLabel}
                        loading="lazy"
                        decoding="async"
                        className="absolute inset-0 h-full w-full object-cover"
                      />
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="w-full bg-background pb-20 md:pb-32">
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
            {surfaces.map((surface) => {
              const Pattern = CARD_PATTERNS[surface.mediaIndex % CARD_PATTERNS.length];

              return (
                <Link
                  key={surface.title}
                  href={surface.href}
                  className="group rounded-2xl border border-border bg-card p-3 transition-colors hover:bg-accent"
                >
                  <div
                    role="img"
                    aria-label={surface.mediaLabel}
                    className="relative min-h-[150px] w-full overflow-hidden rounded-xl border border-border bg-muted text-foreground"
                  >
                    <Pattern />
                    <SurfaceIcon type={surface.icon} />
                  </div>
                  <div className="px-2 pb-3 pt-5">
                    <h3 className="text-lg font-medium text-card-foreground">
                      {surface.title}
                    </h3>
                    <div className="mt-5 rounded-none bg-[hsl(var(--brand))] px-4 py-2 text-center text-xs font-semibold text-white transition-colors group-hover:bg-[hsl(var(--brand))]/90">
                      {surface.cta}
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      </section>

      <section className="w-full bg-background pb-20 md:pb-32">
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
                className="h-11 rounded-none border-0 bg-[hsl(var(--brand))] px-7 text-sm font-semibold text-white shadow-none hover:bg-[hsl(var(--brand))]/90"
              >
                <Link href="/download/latest">
                  <DetectedDownloadLabel />
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
