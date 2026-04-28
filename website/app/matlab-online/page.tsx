import type { Metadata } from "next";
import React from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { SandboxCta } from "@/components/SandboxCta";
import dynamic from "next/dynamic";

const MatlabInlineCodeBlock = dynamic(() => import("@/components/MatlabInlineCodeBlock"), {
  loading: () => <div className="w-full h-[60px] rounded-md bg-muted/40 animate-pulse" />,
});

const BenchmarkShowcaseBlock = dynamic(
  () => import("@/components/benchmarks/BenchmarkShowcaseBlock"),
  { loading: () => <div className="w-full h-[300px] rounded-xl bg-muted/40 animate-pulse" /> },
);
import {
  BarChart3,
  Globe,
  Code2,
  Zap,
  CheckCircle,
  Clock,
  AlertTriangle,
  GitBranch,
  Camera,
  Users,
  Lock,
  Shield,
  Eye,
  ClipboardCheck,
} from "lucide-react";

export const metadata: Metadata = {
  title: "Run MATLAB Code Online - Instant GPU Sandbox",
  description:
    "Execute MATLAB-syntax code instantly in your browser. Open-source runtime with automatic GPU acceleration. No install required—start coding now.",
  alternates: { canonical: "https://runmat.com/matlab-online" },
  keywords: [
    "matlab online", "matlab online free", "run matlab online",
    "matlab alternative", "free matlab", "matlab browser",
    "matlab no license", "matlab gpu", "octave alternative",
    "webgpu matlab", "matlab web app",
  ],
  openGraph: {
    title: "Run MATLAB Code Online - Instant GPU Sandbox",
    description:
      "Execute MATLAB-syntax code instantly in your browser. Open-source runtime with automatic GPU acceleration. No install required—start coding now.",
    url: "/matlab-online",
    siteName: "RunMat",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Run MATLAB Code Online - Instant GPU Sandbox",
    description:
      "Execute MATLAB-syntax code instantly in your browser. Open-source runtime with automatic GPU acceleration. No install required—start coding now.",
  },
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
};

const heroVideoSrc = "https://web.runmatstatic.com/video/3d-interactive-plotting-runmat.mp4";
const heroPosterSrc = "https://web.runmatstatic.com/video/posters/3d-interactive-plotting-runmat.webp";

const faqItems: { question: string; answer: string; answerContent?: React.ReactNode }[] = [
  {
    question: "Is RunMat the same as MATLAB?",
    answer:
      "No. RunMat is an independent project with an open-source runtime that executes MATLAB-syntax code. It is not affiliated with or endorsed by MathWorks, the makers of MATLAB.",
  },
  {
    question: "Can I run my existing MATLAB scripts in RunMat?",
    answer:
      "Many MATLAB scripts run without modification, especially those using core language features and common built-in functions. Scripts relying on specialized toolboxes or advanced features may need adjustments. Try your script in the sandbox to see what works.",
  },
  {
    question: "Is RunMat really free?",
    answer:
      "The RunMat runtime is open source under the MIT license. You can run MATLAB-syntax code in the browser without usage fees or time limits. Cloud features like storage, versioning, and project sharing start on the Hobby tier (100 MB). Paid plans add more storage and team features -- see pricing.",
    answerContent: <>The RunMat runtime is open source under the MIT license. You can run MATLAB-syntax code in the browser without usage fees or time limits. Cloud features like storage, versioning, and project sharing start on the Hobby tier (100 MB). Paid plans add more storage and team features -- see <Link href="/pricing" className="underline hover:text-foreground">pricing</Link>.</>,
  },
  {
    question: "Do I need to create an account?",
    answer:
      "No. The browser sandbox works immediately without sign-up. Creating a free account unlocks cloud storage (100 MB), file versioning, and project sharing.",
  },
  {
    question: "Does RunMat work offline?",
    answer:
      "Yes. Once the sandbox page loads, it can run code without an internet connection. For full offline use with local file access, use the RunMat CLI today. The desktop app with a full IDE experience is coming soon.",
    answerContent: <>Yes. Once the sandbox page loads, it can run code without an internet connection. For full offline use with local file access, use the <Link href="/docs/cli" className="underline hover:text-foreground">RunMat CLI</Link> today. The desktop app with a full IDE experience is coming soon.</>,
  },
  {
    question: "How does RunMat run code in the browser?",
    answer:
      "RunMat compiles to WebAssembly, which runs natively in your browser at near-native speed. For GPU-accelerated operations, it uses WebGPU (available in Chrome, Edge, Safari 18+, and Firefox 139+).",
  },
  {
    question: "Is my code private?",
    answer:
      "Yes. Your code runs entirely on your device. Nothing is sent to a server unless you explicitly choose to save files to the cloud.",
  },
  {
    question: "Can RunMat use my GPU?",
    answer:
      "Yes, in browsers that support WebGPU (Chrome, Edge, Safari 18+, and Firefox 139+). RunMat automatically offloads eligible operations to the GPU for faster execution.",
  },
  {
    question: "What’s the difference between RunMat and GNU Octave?",
    answer:
      "Both run MATLAB-style syntax, but RunMat is designed for performance (JIT compilation, GPU acceleration) and runs natively in the browser. Octave is a mature desktop application with broader toolbox compatibility but no browser-native execution.",
  },
  {
    question: "Is there a desktop version?",
    answer:
      "The RunMat desktop app is coming very soon. It will provide the same interface as the browser sandbox with full local file system access. In the meantime, the CLI is available today for local script execution.",
    answerContent: <>The RunMat desktop app is coming very soon. It will provide the same interface as the browser sandbox with full local file system access. In the meantime, the <Link href="/docs/cli" className="underline hover:text-foreground">CLI</Link> is available today for local script execution.</>,
  },
  {
    question: "Does RunMat support plotting?",
    answer:
      "Yes. RunMat supports 20+ interactive plot types including plot, scatter, hist, histogram, surf, mesh, contour, contourf, imagesc, bar, pie, stem, quiver, area, errorbar, plot3, semilogx, semilogy, and loglog, all GPU-accelerated. 3D surface plots can be rotated, zoomed, and panned directly in the browser.",
    answerContent: <>Yes. RunMat supports 20+ interactive plot types including plot, scatter, hist, histogram, surf, mesh, contour, contourf, imagesc, bar, pie, stem, quiver, area, errorbar, plot3, semilogx, semilogy, and loglog, all GPU-accelerated. 3D surface plots can be rotated, zoomed, and panned directly in the browser. See the <Link href="/blog/matlab-plotting-guide" className="underline hover:text-foreground">plotting guide</Link> for runnable examples.</>,
  },
];

const jsonLd = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "Organization",
      "@id": "https://runmat.com/#organization",
      name: "RunMat",
      alternateName: ["RunMat by Dystr", "Dystr"],
      legalName: "Dystr Inc.",
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
    },
    {
      "@type": "WebPage",
      "@id": "https://runmat.com/matlab-online#webpage",
      url: "https://runmat.com/matlab-online",
      name: "Run MATLAB Code Online - Instant GPU Sandbox",
      description:
        "Execute MATLAB-syntax code instantly in your browser. Open-source runtime with automatic GPU acceleration. No install required—start coding now.",
      inLanguage: "en",
      datePublished: "2026-02-03T00:00:00Z",
      dateModified: "2026-04-21T00:00:00Z",
      isPartOf: { "@id": "https://runmat.com/#website" },
      breadcrumb: { "@id": "https://runmat.com/matlab-online#breadcrumb" },
      author: { "@id": "https://runmat.com/#organization" },
      publisher: { "@id": "https://runmat.com/#organization" },
      video: { "@id": "https://runmat.com/matlab-online#hero-video" },
      mainEntity: [
        { "@id": "https://runmat.com/matlab-online#faq" },
        { "@id": "https://runmat.com/matlab-online#howto" },
        { "@id": "https://runmat.com/matlab-online#software" },
      ],
    },
    {
      "@type": "VideoObject",
      "@id": "https://runmat.com/matlab-online#hero-video",
      name: "RunMat MATLAB-style code in the browser",
      description:
        "Short demo of RunMat running MATLAB-syntax code in the browser with GPU acceleration.",
      thumbnailUrl: heroPosterSrc,
      contentUrl: heroVideoSrc,
      uploadDate: "2026-02-03T00:00:00Z",
      duration: "PT15S",
    },
    {
      "@type": "BreadcrumbList",
      "@id": "https://runmat.com/matlab-online#breadcrumb",
      itemListElement: [
        { "@type": "ListItem", position: 1, name: "Home", item: "https://runmat.com" },
        {
          "@type": "ListItem",
          position: 2,
          name: "MATLAB Online",
          item: "https://runmat.com/matlab-online",
        },
      ],
    },
    {
      "@type": "SoftwareApplication",
      "@id": "https://runmat.com/matlab-online#software",
      name: "RunMat",
      description:
        "RunMat is a high-performance, open-source runtime for math that runs MATLAB-syntax code in the browser with GPU acceleration and no license required.",
      applicationCategory: "ScientificApplication",
      applicationSubCategory: "EngineeringApplication",
      operatingSystem: ["Browser", "Windows", "macOS", "Linux"],
      featureList: [
        "Interactive 2D and 3D plotting with GPU acceleration",
        "Real-time type and shape tracking with dimension error detection",
        "Execution tracing and diagnostic logging",
        "JIT-accelerated MATLAB-style syntax",
        "Automatic GPU fusion and memory management",
        "Browser, desktop, and CLI execution",
        "No account or license required",
        "Cross-platform GPU acceleration (Metal, Vulkan, DirectX 12, WebGPU)",
      ],
      offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
      url: "https://runmat.com/sandbox",
      author: { "@id": "https://runmat.com/#organization" },
      publisher: { "@id": "https://runmat.com/#organization" },
      mainEntityOfPage: { "@id": "https://runmat.com/matlab-online#webpage" },
    },
    {
      "@type": "FAQPage",
      "@id": "https://runmat.com/matlab-online#faq",
      mainEntityOfPage: { "@id": "https://runmat.com/matlab-online#webpage" },
      mainEntity: faqItems.map(item => ({
        "@type": "Question",
        name: item.question,
        acceptedAnswer: {
          "@type": "Answer",
          text: item.answer,
        },
      })),
    },
    {
      "@type": "HowTo",
      "@id": "https://runmat.com/matlab-online#howto",
      name: "Run MATLAB-style code in your browser",
      mainEntityOfPage: { "@id": "https://runmat.com/matlab-online#webpage" },
      step: [
        {
          "@type": "HowToStep",
          position: 1,
          name: "Open the sandbox",
          text: "Visit the RunMat sandbox in your browser.",
        },
        {
          "@type": "HowToStep",
          position: 2,
          name: "Write or paste code",
          text: "Enter MATLAB-style code in the editor.",
        },
        {
          "@type": "HowToStep",
          position: 3,
          name: "Run and see results",
          text: "Execute the script and view outputs instantly.",
        },
      ],
    },
  ],
};

export default function MatlabOnlinePage() {
  return (
    <div className="min-h-screen bg-background">
      <link
        rel="preload"
        as="image"
        href="https://web.runmatstatic.com/video/posters/3d-interactive-plotting-runmat.webp"
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(jsonLd).replace(/<\//g, "<\\/"),
        }}
      />

      {/* Hero */}
      <section className="w-full py-16 md:py-24 lg:py-32" id="hero">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="grid grid-cols-1 gap-10 lg:grid-cols-2 lg:items-center">
            <div className="flex flex-col space-y-6 text-left items-start">
              <div className="mb-2 p-0 text-sm font-semibold uppercase tracking-wider text-foreground">
                MATLAB online alternative
              </div>
              <h1 className="font-bold text-left leading-tight tracking-tight text-3xl sm:text-4xl md:text-5xl">
                Run your MATLAB code in the browser blazing fast
              </h1>
              <p className="max-w-[42rem] leading-relaxed text-foreground text-[0.938rem]">
                RunMat executes MATLAB-syntax code in your browser with auto GPU acceleration. No account or licence
                needed.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto">
                <Button
                  size="lg"
                  asChild
                  className="h-12 px-8 text-base font-semibold rounded-none bg-[hsl(var(--brand))] text-white hover:bg-[hsl(var(--brand))]/90 border-0 shadow-none"
                >
                  <Link
                    href="/sandbox"
                    data-ph-capture-attribute-destination="sandbox"
                    data-ph-capture-attribute-source="matlab-online-hero"
                    data-ph-capture-attribute-cta="try-runmat-browser"
                  >
                    Try RunMat in your browser
                  </Link>
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  asChild
                  className="h-12 px-8 text-base rounded-none bg-card border-border text-foreground"
                >
                  <Link href="/docs/desktop-browser-guide">View getting started</Link>
                </Button>
              </div>
            </div>
            <div className="rounded-xl border border-border bg-card p-2">
              <video
                className="w-full h-auto rounded-lg"
                autoPlay
                muted
                loop
                playsInline
                preload="none"
                poster={heroPosterSrc}
                aria-label="RunMat MATLAB-style code example demo"
              >
                <source src={heroVideoSrc} type="video/mp4" />
              </video>
            </div>
          </div>
        </div>
      </section>

      {/* Challenges with MATLAB Online */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto max-w-5xl rounded-lg border border-amber-500/30 bg-card shadow-sm">
            <div className="px-6 py-8">
              <div className="flex flex-col gap-6">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="mt-1 h-5 w-5 text-amber-600 dark:text-amber-400" />
                  <div>
                    <h3 className="text-lg font-semibold text-foreground">MATLAB Online has friction</h3>
                    <p className="text-[0.938rem] text-foreground mt-2">
                      MATLAB Online runs your code on MathWorks&apos; servers, requires an account, and caps free usage at 20 hours/month with 15-minute idle timeouts. Engineers and students hit these limits regularly:
                    </p>
                  </div>
                </div>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="rounded-lg border border-amber-500/30 bg-card px-5 py-4">
                    <p className="text-sm font-semibold text-foreground">Account barriers</p>
                    <p className="text-sm text-foreground mt-1">
                      Sign-up process and license requirements create unnecessary friction for quick tasks.
                    </p>
                  </div>
                  <div className="rounded-lg border border-amber-500/30 bg-card px-5 py-4">
                    <p className="text-sm font-semibold text-foreground">Idle timeouts &amp; hour caps</p>
                    <p className="text-sm text-foreground mt-1">
                      Sessions timeout after 15 minutes of inactivity. Free tier is capped at 20 hours/month.
                    </p>
                  </div>
                  <div className="rounded-lg border border-amber-500/30 bg-card px-5 py-4">
                    <p className="text-sm font-semibold text-foreground">Cloud dependency</p>
                    <p className="text-sm text-foreground mt-1">
                      Code must be uploaded to remote servers, raising privacy and connectivity concerns.
                    </p>
                  </div>
                  <div className="rounded-lg border border-amber-500/30 bg-card px-5 py-4">
                    <p className="text-sm font-semibold text-foreground">No local GPU access</p>
                    <p className="text-sm text-foreground mt-1">
                      Code runs on MathWorks&apos; servers, so you cannot use your own GPU for acceleration.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* RunMat as an alternative */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl text-foreground">
              Meet RunMat: no license required
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground">
              RunMat is an open-source runtime that understands MATLAB syntax and runs it directly in your browser.
              Your code executes on your own device via WebAssembly, with GPU acceleration in browsers that support
              WebGPU.
            </p>
          </div>
          <div className="mx-auto mt-12 max-w-5xl">
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              <div className="rounded-lg border border-border bg-muted/40 p-6">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                  <CheckCircle className="h-5 w-5" />
                </span>
                <p className="text-base font-medium text-foreground">No account required</p>
                <p className="text-[0.938rem] text-foreground mt-1">Open the sandbox and start coding immediately. No sign-up, no license key.</p>
              </div>
              <div className="rounded-lg border border-border bg-muted/40 p-6">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                  <Code2 className="h-5 w-5" />
                </span>
                <p className="text-base font-medium text-foreground">Client-side execution</p>
                <p className="text-[0.938rem] text-foreground mt-1">
                  Your code compiles to WebAssembly and runs locally in your browser. Nothing leaves your device unless you choose to save to the cloud.
                </p>
              </div>
              <div className="rounded-lg border border-border bg-muted/40 p-6">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                  <Zap className="h-5 w-5" />
                </span>
                <p className="text-base font-medium text-foreground">GPU acceleration on any vendor</p>
                <p className="text-[0.938rem] text-foreground mt-1">
                  Metal on Mac, Vulkan on Linux, DirectX 12 on Windows. WebGPU in the browser. No CUDA dependency.
                </p>
                <Link href="/blog/how-to-use-gpu-in-matlab" className="text-xs text-muted-foreground hover:text-foreground underline mt-2 inline-block">Learn more</Link>
              </div>
              <div className="rounded-lg border border-border bg-muted/40 p-6">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                  <BarChart3 className="h-5 w-5" />
                </span>
                <p className="text-base font-medium text-foreground">Interactive 2D &amp; 3D plotting</p>
                <p className="text-[0.938rem] text-foreground mt-1">GPU-rendered surfaces you can rotate, zoom, and pan. Plots live in the same computation chain as your math.</p>
              </div>
              <div className="rounded-lg border border-border bg-muted/40 p-6">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                  <AlertTriangle className="h-5 w-5" />
                </span>
                <p className="text-base font-medium text-foreground">Type &amp; shape tracking</p>
                <p className="text-[0.938rem] text-foreground mt-1">Hover any variable to see its dimensions. Mismatched matrix sizes get red underlines before you run.</p>
              </div>
              <div className="rounded-lg border border-border bg-muted/40 p-6">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                  <Globe className="h-5 w-5" />
                </span>
                <p className="text-base font-medium text-foreground">Works offline</p>
                <p className="text-[0.938rem] text-foreground mt-1">
                  After the initial page load, RunMat runs without an internet connection. The CLI provides full local file access today; the desktop app is coming soon.
                </p>
              </div>
            </div>
            <div className="mt-10 flex justify-center">
              <Button
                size="lg"
                asChild
                className="h-12 px-8 text-base font-semibold rounded-none bg-[hsl(var(--brand))] text-white hover:bg-[hsl(var(--brand))]/90 border-0 shadow-none"
              >
                <Link
                  href="/sandbox"
                  data-ph-capture-attribute-destination="sandbox"
                  data-ph-capture-attribute-source="matlab-online-features"
                  data-ph-capture-attribute-cta="try-runmat-browser"
                >
                  Try RunMat in your browser
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl text-foreground">How it works</h2>
            <p className="max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground">Get started in three simple steps.</p>
          </div>
          <div className="mx-auto max-w-5xl grid grid-cols-1 lg:grid-cols-2 gap-10 items-center">
            <div className="space-y-8">
              <div className="flex items-start gap-4">
                <span className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[hsl(var(--brand))]/10 text-[hsl(var(--brand))] text-lg font-bold">
                  1
                </span>
                <div>
                  <h3 className="text-lg font-semibold text-foreground">Open the sandbox</h3>
                  <p className="text-[0.938rem] text-foreground mt-1">Go to <Link href="/sandbox" className="underline text-[hsl(var(--brand))] hover:text-[hsl(var(--brand))]/80">runmat.com/sandbox</Link>. No downloads, no sign-up, no license key.</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <span className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[hsl(var(--brand))]/10 text-[hsl(var(--brand))] text-lg font-bold">
                  2
                </span>
                <div>
                  <h3 className="text-lg font-semibold text-foreground">Write or paste code</h3>
                  <p className="text-[0.938rem] text-foreground mt-1">Type MATLAB-style code directly in the editor, or paste existing scripts from your projects.</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <span className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[hsl(var(--brand))]/10 text-[hsl(var(--brand))] text-lg font-bold">
                  3
                </span>
                <div>
                  <h3 className="text-lg font-semibold text-foreground">Run and see results</h3>
                  <p className="text-[0.938rem] text-foreground mt-1">Execute your code instantly. View outputs, plots, and results in real time, all in your browser.</p>
                </div>
              </div>
            </div>
            <div className="rounded-xl border border-border overflow-hidden">
              <video
                className="w-full h-auto"
                autoPlay
                muted
                loop
                playsInline
                preload="metadata"
                poster="https://web.runmatstatic.com/video/posters/runmat-wave-simulation.webp"
                aria-label="RunMat wave simulation demo"
              >
                <source src="https://web.runmatstatic.com/video/runmat-wave-simulation.mp4" type="video/mp4" />
              </video>
            </div>
          </div>
        </div>
      </section>

      {/* Versioning and collaboration */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl text-foreground">Every change versioned. No git required.</h2>
            <p className="max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground">
              Every save creates a version automatically. Per-file history and full project snapshots are included on all{" "}
              <Link href="/pricing" className="underline hover:text-foreground/80">Cloud tiers</Link>, starting at $0 with 100 MB on the Hobby tier. Paid plans add project sharing with your team -- no git setup or merge conflicts.
            </p>
          </div>
          <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-3">
            <div className="rounded-lg border border-border bg-card p-6">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-border bg-foreground/10 text-foreground mb-3">
                <GitBranch className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">Automatic file history</h3>
              <p className="text-[0.938rem] text-foreground mt-1">Browse the timeline and restore any previous state. No commits, no staging area.</p>
              <Link href="/blog/version-control-for-engineers-who-dont-use-git" className="text-xs text-muted-foreground hover:text-foreground underline mt-2 inline-block">Learn more</Link>
            </div>
            <div className="rounded-lg border border-border bg-card p-6">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-border bg-foreground/10 text-foreground mb-3">
                <Camera className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">Project snapshots</h3>
              <p className="text-[0.938rem] text-foreground mt-1">Capture your entire project in one click. Restore instantly, even across terabyte-scale datasets.</p>
              <Link href="/docs/versioning" className="text-xs text-muted-foreground hover:text-foreground underline mt-2 inline-block">Learn more</Link>
            </div>
            <div className="rounded-lg border border-border bg-card p-6">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-border bg-foreground/10 text-foreground mb-3">
                <Users className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">Cloud project sharing</h3>
              <p className="text-[0.938rem] text-foreground mt-1">Share projects with colleagues instantly. No shared drives, no emailing files back and forth.</p>
              <Link href="/blog/from-ad-hoc-checkpoints-to-reliable-large-data-persistence" className="text-xs text-muted-foreground hover:text-foreground underline mt-2 inline-block">Learn more</Link>
            </div>
          </div>
        </div>
      </section>

      {/* Benchmarks */}
      <section className="w-full py-16 md:py-24 lg:py-32">
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

      {/* Comparison */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl text-foreground">RunMat vs. MATLAB Online</h2>
            <p className="max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground">RunMat runs client-side with GPU acceleration and no account. MATLAB Online requires a license, runs on MathWorks&apos; servers, and caps free usage.</p>
          </div>
          <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-2">
            <Card className="border border-border bg-card shadow-sm">
              <CardContent className="p-6 space-y-4">
                <div>
                  <h3 className="text-lg font-semibold text-foreground">RunMat</h3>
                  <p className="text-[0.938rem] text-foreground">High-performance, open-source runtime for math</p>
                </div>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-3 text-green-600 dark:text-green-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-600 dark:text-green-400">
                      ✓
                    </span>
                    Open-source runtime
                  </li>
                  <li className="flex items-start gap-3 text-green-600 dark:text-green-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-600 dark:text-green-400">
                      ✓
                    </span>
                    No account required
                  </li>
                  <li className="flex items-start gap-3 text-green-600 dark:text-green-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-600 dark:text-green-400">
                      ✓
                    </span>
                    Client-side execution
                  </li>
                  <li className="flex items-start gap-3 text-green-600 dark:text-green-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-600 dark:text-green-400">
                      ✓
                    </span>
                    Cross-platform GPU (Metal, Vulkan, DX12, WebGPU)
                  </li>
                  <li className="flex items-start gap-3 text-green-600 dark:text-green-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-600 dark:text-green-400">
                      ✓
                    </span>
                    Works offline
                  </li>
                  <li className="flex items-start gap-3 text-green-600 dark:text-green-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-600 dark:text-green-400">
                      ✓
                    </span>
                    Core matrix operations
                  </li>
                  <li className="flex items-start gap-3 text-green-600 dark:text-green-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-600 dark:text-green-400">
                      ✓
                    </span>
                    Interactive 2D &amp; 3D plotting
                  </li>
                  <li className="flex items-start gap-3 text-green-600 dark:text-green-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-600 dark:text-green-400">
                      ✓
                    </span>
                    Real-time type &amp; shape tracking
                  </li>
                  <li className="flex items-start gap-3 text-green-600 dark:text-green-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-600 dark:text-green-400">
                      ✓
                    </span>
                    Execution tracing &amp; diagnostics
                  </li>
                  <li className="flex items-start gap-3 text-green-600 dark:text-green-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-600 dark:text-green-400">
                      ✓
                    </span>
                    Automatic file versioning &amp; snapshots (Cloud)
                  </li>
                  <li className="flex items-start gap-3 text-foreground">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-foreground">
                      –
                    </span>
                    Limited package / toolbox support
                  </li>
                  <li className="flex items-start gap-3 text-foreground">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-foreground">
                      –
                    </span>
                    Subset of MATLAB functions
                  </li>
                </ul>
              </CardContent>
            </Card>
            <Card className="border border-border/60 bg-card shadow-sm">
              <CardContent className="p-6 space-y-4">
                <div>
                  <h3 className="text-lg font-semibold text-foreground">MATLAB Online</h3>
                  <p className="text-[0.938rem] text-foreground">MathWorks official platform</p>
                </div>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-3 text-red-600 dark:text-red-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-red-600 dark:text-red-400">
                      ✕
                    </span>
                    Requires paid license
                  </li>
                  <li className="flex items-start gap-3 text-red-600 dark:text-red-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-red-600 dark:text-red-400">
                      ✕
                    </span>
                    Account &amp; sign-in required
                  </li>
                  <li className="flex items-start gap-3 text-red-600 dark:text-red-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-red-600 dark:text-red-400">
                      ✕
                    </span>
                    Cloud-based execution
                  </li>
                  <li className="flex items-start gap-3 text-green-600 dark:text-green-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-600 dark:text-green-400">
                      ✓
                    </span>
                    GPU support available
                  </li>
                  <li className="flex items-start gap-3 text-red-600 dark:text-red-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-red-600 dark:text-red-400">
                      ✕
                    </span>
                    Requires internet connection
                  </li>
                  <li className="flex items-start gap-3 text-green-600 dark:text-green-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-600 dark:text-green-400">
                      ✓
                    </span>
                    Full MATLAB language
                  </li>
                  <li className="flex items-start gap-3 text-green-600 dark:text-green-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-600 dark:text-green-400">
                      ✓
                    </span>
                    Complete toolbox ecosystem
                  </li>
                  <li className="flex items-start gap-3 text-green-600 dark:text-green-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-600 dark:text-green-400">
                      ✓
                    </span>
                    Official MathWorks support
                  </li>
                  <li className="flex items-start gap-3 text-red-600 dark:text-red-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-red-600 dark:text-red-400">
                      ✕
                    </span>
                    No built-in file versioning
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* What works today */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl text-foreground">What works today</h2>
            <p className="max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground">Core matrix workflows, plotting, and debugging ship today. Here is what is in progress.</p>
          </div>
          <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-2">
            <Card className="border border-green-500/30 bg-card shadow-sm">
              <CardHeader className="flex flex-row items-center gap-3 border-b border-border/60">
                <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" />
                <CardTitle className="text-lg text-foreground">Works well</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-foreground text-sm">
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-600 dark:text-green-400">•</span>
                  <p>Matrix and array operations (indexing, slicing, reshaping)</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-600 dark:text-green-400">•</span>
                  <p>Arithmetic, logical, and relational operators</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-600 dark:text-green-400">•</span>
                  <p>Control flow (if/else, for, while, switch)</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-600 dark:text-green-400">•</span>
                  <p>User-defined functions with multiple outputs</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-600 dark:text-green-400">•</span>
                  <p>Cells, structs, and basic classdef OOP</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-600 dark:text-green-400">•</span>
                  <p>400+ built-in functions</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-600 dark:text-green-400">•</span>
                  <p>20+ plot types (plot, scatter, surf, mesh, contour, contourf, imagesc, bar, pie, stem, quiver, area, errorbar, plot3, semilogx/semilogy/loglog, and more) with GPU acceleration, subplots, figure handles, and interactive 3D camera</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-600 dark:text-green-400">•</span>
                  <p>Real-time type and shape tracking (hover to see matrix dimensions)</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-600 dark:text-green-400">•</span>
                  <p>Live syntax validation (red underlines for dimension mismatches and errors)</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-600 dark:text-green-400">•</span>
                  <p>Execution tracing and diagnostic logging</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-600 dark:text-green-400">•</span>
                  <p>Async code execution (non-blocking runs)</p>
                </div>
              </CardContent>
            </Card>
            <Card className="border border-amber-500/30 bg-card shadow-sm">
              <CardHeader className="flex flex-row items-center gap-3 border-b border-border/60">
                <Clock className="h-4 w-4 text-amber-600 dark:text-amber-400" />
                <CardTitle className="text-lg text-foreground">Limitations &amp; future work</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-foreground text-sm">
                <div>
                  <h4 className="text-sm font-semibold text-foreground mb-2">In progress</h4>
                  <div className="space-y-2">
                    <div className="flex items-start gap-2">
                      <span className="mt-1 text-amber-600 dark:text-amber-400">•</span>
                      <p>Extensible package support (signal processing, optimization, etc.)</p>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="mt-1 text-amber-600 dark:text-amber-400">•</span>
                      <p>Some edge-case MATLAB semantics</p>
                    </div>
                  </div>
                </div>
                <div className="border-t border-border/60 pt-4">
                  <h4 className="text-sm font-semibold text-foreground mb-2">Not supported</h4>
                  <div className="space-y-2">
                    <div className="flex items-start gap-2">
                      <span className="mt-1 text-muted-foreground">•</span>
                      <p>Simulink or graphical block diagrams</p>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="mt-1 text-muted-foreground">•</span>
                      <p>MATLAB-specific file formats (.slx, .mlapp)</p>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="mt-1 text-muted-foreground">•</span>
                      <p>Java/COM interop</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
          <p className="mx-auto mt-6 max-w-4xl text-sm text-foreground text-center">
            For a detailed list, see the{" "}
            <Link href="/docs/language-coverage" className="underline">
              language coverage guide
            </Link>{" "}
            and{" "}
            <Link href="/docs/matlab-function-reference" className="underline">
              function reference
            </Link>
            . For workflow guides, browse{" "}
            <Link href="/resources" className="underline">
              resources
            </Link>{" "}
            or read our deep dives on{" "}
            <Link href="/blog/how-to-use-gpu-in-matlab" className="underline">
              GPU acceleration
            </Link>{" "}
            and the{" "}
            <Link href="/blog/matlab-fft-guide" className="underline">
              FFT family in RunMat
            </Link>
            .
          </p>
        </div>
      </section>

      {/* Enterprise */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto max-w-5xl">
            <div className="rounded-lg border border-border bg-muted/40 p-8">
              <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl text-foreground mb-8">Built for teams</h2>
              <div className="grid gap-6 md:grid-cols-2">
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-border bg-secondary text-foreground">
                    <Lock className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-foreground">SSO &amp; SCIM</p>
                    <p className="text-[0.938rem] text-foreground">Integrate with your identity provider. Provision and deprovision users automatically.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-border bg-secondary text-foreground">
                    <Shield className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-foreground">ITAR-compliant deployment</p>
                    <p className="text-[0.938rem] text-foreground">Self-hosted, air-gapped option for export-controlled environments.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-border bg-secondary text-foreground">
                    <Eye className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-foreground">Open source &amp; auditable</p>
                    <p className="text-[0.938rem] text-foreground">MIT-licensed runtime. Inspect every line of code that runs your math.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-border bg-secondary text-foreground">
                    <ClipboardCheck className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-foreground">SOC 2 ready</p>
                    <p className="text-[0.938rem] text-foreground">Built to SOC 2 standards. Audit planned for Q2 2026.</p>
                  </div>
                </div>
              </div>
              <div className="mt-6">
                <Link href="/pricing" className="text-sm text-muted-foreground hover:text-foreground underline">
                  See enterprise pricing
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* FAQs */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl text-foreground">Frequently asked questions</h2>
            <p className="max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground">
              Common questions about RunMat and MATLAB compatibility.
            </p>
          </div>
          <div className="mx-auto grid max-w-5xl gap-4 md:grid-cols-2">
            {faqItems.map(item => (
              <details
                key={item.question}
                className="group self-start rounded-xl border border-border/60 bg-card shadow-sm"
              >
                <summary className="flex cursor-pointer list-none items-center justify-between px-6 py-4 text-foreground">
                  <span className="text-sm font-medium">{item.question}</span>
                  <span className="text-muted-foreground transition-transform duration-200 group-open:rotate-180">
                    ⌄
                  </span>
                </summary>
                <div className="px-6 pb-4 text-sm text-foreground">
                  {item.answerContent ?? item.answer}
                </div>
              </details>
            ))}
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="mx-auto max-w-3xl">
            <SandboxCta source="matlab-online-bottom-cta" />
          </div>
        </div>
      </section>
    </div>
  );
}
