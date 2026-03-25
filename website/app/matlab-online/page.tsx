import type { Metadata } from "next";
import React from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import MatlabInlineCodeBlock from "@/components/MatlabInlineCodeBlock";
import BenchmarkShowcaseBlock from "@/components/benchmarks/BenchmarkShowcaseBlock";
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
const heroPosterSrc = "https://web.runmatstatic.com/matlab-online-product-screenshot.png";

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
      "Yes. RunMat supports interactive 2D and 3D plotting. 3D surface plots can be rotated, zoomed, and panned directly in the browser. Additional chart types are being added.",
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
      dateModified: "2026-03-05T00:00:00Z",
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
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(jsonLd).replace(/<\//g, "<\\/"),
        }}
      />

      <div className="container mx-auto px-4 md:px-6 lg:px-8">
        {/* Hero */}
        <section className="w-full py-16 md:py-24 lg:py-32" id="hero">
          <div className="grid grid-cols-1 gap-10 lg:grid-cols-2 lg:items-center">
            <div className="flex flex-col space-y-6 text-left items-start">
              <div className="mb-2 p-0 text-lg font-semibold uppercase tracking-wide text-primary">
                MATLAB online alternative
              </div>
              <h1 className="font-heading text-left leading-tight tracking-tight text-[clamp(2.6rem,4.8vw,4.25rem)] sm:text-[clamp(3rem,4vw,5rem)] lg:text-[clamp(3.25rem,3.6vw,5.25rem)]">
                Run your MATLAB code in the browser blazing fast
              </h1>
              <p className="max-w-[42rem] leading-relaxed text-muted-foreground text-lg">
                RunMat executes MATLAB-syntax code in your browser with auto GPU acceleration. No account or licence
                needed.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto">
                <Button
                  size="lg"
                  asChild
                  className="h-12 px-8 text-base font-semibold bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-xl border-0 transition-all duration-200"
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
                  className="h-12 px-8 text-base bg-[#0E1B1E] dark:bg-[#0E1B1E] border-gray-200 dark:border-gray-700 text-gray-100 dark:text-gray-100"
                >
                  <Link href="/docs/desktop-browser-guide">View getting started</Link>
                </Button>
              </div>
            </div>
            <div className="rounded-xl border border-border bg-muted/40 p-2 bg-[radial-gradient(ellipse_at_top,_rgba(124,58,237,0.25),_transparent_60%)]">
              <video
                className="w-full h-auto rounded-lg"
                autoPlay
                muted
                loop
                playsInline
                preload="metadata"
                poster={heroPosterSrc}
                aria-label="RunMat MATLAB-style code example demo"
              >
                <source src={heroVideoSrc} type="video/mp4" />
              </video>
            </div>
          </div>
        </section>

        {/* Challenges with MATLAB Online */}
        <section className="py-16 md:py-24 lg:py-32">
          <div className="mx-auto max-w-5xl rounded-xl border border-amber-500/30 bg-gradient-to-r from-amber-500/15 to-transparent shadow-lg">
            <div className="px-6 py-8">
              <div className="flex flex-col gap-6">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="mt-1 h-5 w-5 text-amber-300" />
                  <div>
                    <h3 className="text-xl font-semibold text-foreground">MATLAB Online has friction</h3>
                    <p className="text-lg text-muted-foreground mt-2">
                      MATLAB Online runs your code on MathWorks&apos; servers, requires an account, and caps free usage at 20 hours/month with 15-minute idle timeouts. Engineers and students hit these limits regularly:
                    </p>
                  </div>
                </div>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="rounded-lg border border-amber-500/30 bg-[#0E1421] px-5 py-4">
                    <p className="text-xl font-semibold text-gray-100">Account barriers</p>
                    <p className="text-lg text-gray-300 mt-2">
                      Sign-up process and license requirements create unnecessary friction for quick tasks.
                    </p>
                  </div>
                  <div className="rounded-lg border border-amber-500/30 bg-[#0E1421] px-5 py-4">
                    <p className="text-xl font-semibold text-gray-100">Idle timeouts &amp; hour caps</p>
                    <p className="text-lg text-gray-300 mt-2">
                      Sessions timeout after 15 minutes of inactivity. Free tier is capped at 20 hours/month.
                    </p>
                  </div>
                  <div className="rounded-lg border border-amber-500/30 bg-[#0E1421] px-5 py-4">
                    <p className="text-xl font-semibold text-gray-100">Cloud dependency</p>
                    <p className="text-lg text-gray-300 mt-2">
                      Code must be uploaded to remote servers, raising privacy and connectivity concerns.
                    </p>
                  </div>
                  <div className="rounded-lg border border-amber-500/30 bg-[#0E1421] px-5 py-4">
                    <p className="text-xl font-semibold text-gray-100">No local GPU access</p>
                    <p className="text-lg text-gray-300 mt-2">
                      Code runs on MathWorks&apos; servers, so you cannot use your own GPU for acceleration.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* RunMat as an alternative */}
        <section className="py-16 md:py-24 lg:py-32">
          <div className="mx-auto max-w-4xl space-y-4 text-center">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">
              Meet RunMat: no license required
            </h2>
            <p className="text-muted-foreground text-lg">
              RunMat is an open-source runtime that understands MATLAB syntax and runs it directly in your browser.
              Your code executes on your own device via WebAssembly, with GPU acceleration in browsers that support
              WebGPU.
            </p>
          </div>
          <div className="mx-auto mt-10 max-w-5xl space-y-6">
            <div className="rounded-2xl border border-purple-500/30 bg-gradient-to-br from-purple-500/15 via-[#0E1421] to-[#0A0F1C] p-8 shadow-lg">
              <div className="text-center space-y-2">
                <h3 className="text-xl md:text-2xl font-semibold text-gray-100">
                  Key differences from MATLAB Online
                </h3>
              </div>
              <div className="mt-8 grid gap-6 md:grid-cols-2">
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-200">
                    <CheckCircle className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-gray-100">No account required</p>
                    <p className="text-lg text-gray-300">Open the sandbox and start coding immediately. No sign-up, no license key.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-200">
                    <Code2 className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-gray-100">Client-side execution</p>
                    <p className="text-lg text-gray-300">
                      Your code compiles to WebAssembly and runs locally in your browser. Nothing leaves your device unless you choose to save to the cloud.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-200">
                    <Zap className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-gray-100">GPU acceleration on any vendor</p>
                    <p className="text-lg text-gray-300">
                      Metal on Mac, Vulkan on Linux, DirectX 12 on Windows. WebGPU in the browser. No CUDA dependency.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-200">
                    <BarChart3 className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-gray-100">Interactive 2D &amp; 3D plotting</p>
                    <p className="text-lg text-gray-300">GPU-rendered surfaces you can rotate, zoom, and pan. Plots live in the same computation chain as your math.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-200">
                    <AlertTriangle className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-gray-100">Type &amp; shape tracking</p>
                    <p className="text-lg text-gray-300">Hover any variable to see its dimensions. Mismatched matrix sizes get red underlines before you run.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-200">
                    <Globe className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-gray-100">Works offline</p>
                    <p className="text-lg text-gray-300">
                      After the initial page load, RunMat runs without an internet connection. The CLI provides full local file access today; the desktop app is coming soon.
                    </p>
                  </div>
                </div>
              </div>
              <div className="mt-8 flex justify-center">
                <Button
                  size="lg"
                  asChild
                  className="h-12 px-8 text-base font-semibold bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-xl border-0 transition-all duration-200"
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
        <section className="py-16 md:py-24 lg:py-32">
          <div className="mx-auto max-w-4xl text-center mb-10 space-y-3">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">How it works</h2>
            <p className="text-muted-foreground text-lg">Get started in three simple steps.</p>
          </div>
          <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-3 md:items-stretch">
            <Card className="border border-border/60 bg-[#0E1421] shadow-lg overflow-hidden flex flex-col h-full">
              <div className="bg-gradient-to-r from-purple-500/20 to-transparent p-6 flex items-start justify-start h-[200px]">
                <Link
                  href="https://runmat.com/sandbox"
                  target="_blank"
                  rel="noreferrer"
                  className="rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-100 px-3 py-2 text-base flex items-center gap-2 transition hover:border-purple-400/60 hover:text-purple-50"
                  data-ph-capture-attribute-destination="sandbox"
                  data-ph-capture-attribute-source="matlab-online-how-it-works"
                  data-ph-capture-attribute-cta="runmat-org-sandbox"
                >
                  <span className="h-2 w-2 rounded-full bg-red-400"></span>
                  <span className="h-2 w-2 rounded-full bg-yellow-400"></span>
                  <span className="h-2 w-2 rounded-full bg-green-400"></span>
                  <span className="ml-2">runmat.com/sandbox</span>
                </Link>
              </div>
              <CardContent className="pt-6 space-y-2 flex-1">
                <div className="inline-flex h-7 w-7 items-center justify-center rounded-full border border-purple-500/40 bg-purple-500/10 text-purple-100 text-base">
                  1
                </div>
                <h3 className="text-xl font-semibold text-gray-100">Open the sandbox</h3>
                <p className="text-lg text-gray-300">
                  Click the button to launch RunMat in your browser. No downloads or sign-ups required.
                </p>
              </CardContent>
            </Card>
            <Card className="border border-border/60 bg-[#0E1421] shadow-lg overflow-hidden flex flex-col h-full">
              <div className="bg-gradient-to-r from-purple-500/20 to-transparent p-6 flex flex-col items-start justify-start gap-3 min-h-[200px]">
                <div className="how-it-works-code w-full rounded-md overflow-hidden bg-[hsl(var(--code-surface))] dark:bg-[hsl(var(--code-surface))]">
                  <MatlabInlineCodeBlock
                    code={"A = [3 1 3 2];\nC = unique(A);"}
                    showRunButton
                    preClassName="[&_code]:text-[0.75rem] [&_code]:leading-relaxed !bg-transparent"
                  />
                </div>
              </div>
              <CardContent className="pt-6 space-y-2 flex-1">
                <div className="inline-flex h-7 w-7 items-center justify-center rounded-full border border-purple-500/40 bg-purple-500/10 text-purple-100 text-base">
                  2
                </div>
                <h3 className="text-xl font-semibold text-gray-100">Write or paste code</h3>
                <p className="text-lg text-gray-300">
                  Type your MATLAB-style code directly, or paste existing scripts from your projects.
                </p>
              </CardContent>
            </Card>
            <Card className="border border-border/60 bg-[#0E1421] shadow-lg overflow-hidden flex flex-col h-full">
              <div className="bg-gradient-to-r from-purple-500/20 to-transparent p-6 flex items-start justify-start h-[200px]">
                <div className="how-it-works-code w-full rounded-md overflow-hidden bg-[hsl(var(--code-surface))] dark:bg-[hsl(var(--code-surface))]">
                  <MatlabInlineCodeBlock
                    code={"C =\n\n     1     2     3"}
                    preClassName="[&_code]:text-[0.75rem] [&_code]:leading-relaxed !bg-transparent"
                  />
                </div>
              </div>
              <CardContent className="pt-6 space-y-2 flex-1">
                <div className="inline-flex h-7 w-7 items-center justify-center rounded-full border border-purple-500/40 bg-purple-500/10 text-purple-100 text-base">
                  3
                </div>
                <h3 className="text-xl font-semibold text-gray-100">Run and see results</h3>
                <p className="text-lg text-gray-300">
                  Execute your code instantly. View outputs, plots, and results in real time.
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Versioning and collaboration */}
        <section className="py-16 md:py-24 lg:py-32">
          <div className="mx-auto max-w-4xl text-center mb-10 space-y-3">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">Every change versioned. No git required.</h2>
            <p className="text-muted-foreground text-lg">
              Every save creates a version automatically. Per-file history and full project snapshots are included on all{" "}
              <Link href="/pricing" className="underline hover:text-foreground">Cloud tiers</Link>, starting at $0 with 100 MB on the Hobby tier. Paid plans add project sharing with your team -- no git setup or merge conflicts.
            </p>
          </div>
          <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-3">
            <div className="rounded-2xl border border-purple-500/30 bg-gradient-to-br from-purple-500/10 to-[#0E1421] p-6">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-gray-400 mb-3">
                <GitBranch className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-gray-100">Automatic file history</h3>
              <p className="text-sm text-gray-300 mt-1">Browse the timeline and restore any previous state. No commits, no staging area.</p>
              <Link href="/blog/version-control-for-engineers-who-dont-use-git" className="text-xs text-muted-foreground hover:text-foreground underline mt-2 inline-block">Learn more</Link>
            </div>
            <div className="rounded-2xl border border-purple-500/30 bg-gradient-to-br from-purple-500/10 to-[#0E1421] p-6">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-gray-400 mb-3">
                <Camera className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-gray-100">Project snapshots</h3>
              <p className="text-sm text-gray-300 mt-1">Capture your entire project in one click. Restore instantly, even across terabyte-scale datasets.</p>
              <Link href="/docs/versioning" className="text-xs text-muted-foreground hover:text-foreground underline mt-2 inline-block">Learn more</Link>
            </div>
            <div className="rounded-2xl border border-purple-500/30 bg-gradient-to-br from-purple-500/10 to-[#0E1421] p-6">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-gray-400 mb-3">
                <Users className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-gray-100">Cloud project sharing</h3>
              <p className="text-sm text-gray-300 mt-1">Share projects with colleagues instantly. No shared drives, no emailing files back and forth.</p>
              <Link href="/blog/from-ad-hoc-checkpoints-to-reliable-large-data-persistence" className="text-xs text-muted-foreground hover:text-foreground underline mt-2 inline-block">Learn more</Link>
            </div>
          </div>
        </section>

        {/* Benchmarks */}
        <section className="py-16 md:py-24 lg:py-32">
          <div className="mx-auto max-w-4xl text-center mb-10 space-y-3">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">The fastest runtime for your math</h2>
            <p className="text-muted-foreground text-lg">
              RunMat fuses sequential operations into fewer GPU steps and keeps arrays on-device between steps. Less memory traffic, fewer kernel launches, faster scripts.
            </p>
          </div>
          <div className="mx-auto max-w-3xl mb-6">
            <BenchmarkShowcaseBlock />
          </div>
          <p className="mx-auto max-w-3xl text-sm text-muted-foreground text-center">
            Times shown are CLI results. The same benchmarks run in the sandbox; browser GPU throttling affects absolute times. No MATLAB comparison per MathWorks&apos; terms.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mt-6">
            <Link href="/docs/accelerate/fusion-intro" className="text-sm hover:text-foreground text-muted-foreground transition-colors underline">
              How fusion works
            </Link>
            <span className="hidden sm:inline text-muted-foreground">&middot;</span>
            <Link href="/benchmarks" className="text-sm hover:text-foreground text-muted-foreground transition-colors underline">
              All benchmarks
            </Link>
          </div>
        </section>

        {/* Comparison */}
        <section className="py-16 md:py-24 lg:py-32">
          <div className="mx-auto max-w-5xl text-center mb-10 space-y-3">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">RunMat vs. MATLAB Online</h2>
            <p className="text-muted-foreground text-lg">RunMat runs client-side with GPU acceleration and no account. MATLAB Online requires a license, runs on MathWorks&apos; servers, and caps free usage.</p>
          </div>
          <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-2">
            <Card className="border border-purple-500/30 bg-[#0E1421] shadow-lg">
              <CardContent className="p-6 space-y-4">
                <div>
                  <h3 className="text-xl font-semibold text-purple-200">RunMat</h3>
                  <p className="text-lg text-gray-300">High-performance, open-source runtime for math</p>
                </div>
                <ul className="space-y-3 text-lg">
                  <li className="flex items-start gap-3 text-green-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-300">
                      ✓
                    </span>
                    Open-source runtime
                  </li>
                  <li className="flex items-start gap-3 text-green-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-300">
                      ✓
                    </span>
                    No account required
                  </li>
                  <li className="flex items-start gap-3 text-green-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-300">
                      ✓
                    </span>
                    Client-side execution
                  </li>
                  <li className="flex items-start gap-3 text-green-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-300">
                      ✓
                    </span>
                    Cross-platform GPU (Metal, Vulkan, DX12, WebGPU)
                  </li>
                  <li className="flex items-start gap-3 text-green-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-300">
                      ✓
                    </span>
                    Works offline
                  </li>
                  <li className="flex items-start gap-3 text-green-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-300">
                      ✓
                    </span>
                    Core matrix operations
                  </li>
                  <li className="flex items-start gap-3 text-green-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-300">
                      ✓
                    </span>
                    Interactive 2D &amp; 3D plotting
                  </li>
                  <li className="flex items-start gap-3 text-green-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-300">
                      ✓
                    </span>
                    Real-time type &amp; shape tracking
                  </li>
                  <li className="flex items-start gap-3 text-green-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-300">
                      ✓
                    </span>
                    Execution tracing &amp; diagnostics
                  </li>
                  <li className="flex items-start gap-3 text-green-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-300">
                      ✓
                    </span>
                    Automatic file versioning &amp; snapshots (Cloud)
                  </li>
                  <li className="flex items-start gap-3 text-gray-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-gray-400">
                      –
                    </span>
                    Limited package / toolbox support
                  </li>
                  <li className="flex items-start gap-3 text-gray-400">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-gray-400">
                      –
                    </span>
                    Subset of MATLAB functions
                  </li>
                </ul>
              </CardContent>
            </Card>
            <Card className="border border-border/60 bg-[#0E1421] shadow-lg">
              <CardContent className="p-6 space-y-4">
                <div>
                  <h3 className="text-xl font-semibold text-gray-100">MATLAB Online</h3>
                  <p className="text-lg text-gray-300">MathWorks official platform</p>
                </div>
                <ul className="space-y-3 text-lg">
                  <li className="flex items-start gap-3 text-red-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-red-300">
                      ✕
                    </span>
                    Requires paid license
                  </li>
                  <li className="flex items-start gap-3 text-red-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-red-300">
                      ✕
                    </span>
                    Account &amp; sign-in required
                  </li>
                  <li className="flex items-start gap-3 text-red-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-red-300">
                      ✕
                    </span>
                    Cloud-based execution
                  </li>
                  <li className="flex items-start gap-3 text-green-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-300">
                      ✓
                    </span>
                    GPU support available
                  </li>
                  <li className="flex items-start gap-3 text-red-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-red-300">
                      ✕
                    </span>
                    Requires internet connection
                  </li>
                  <li className="flex items-start gap-3 text-green-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-300">
                      ✓
                    </span>
                    Full MATLAB language
                  </li>
                  <li className="flex items-start gap-3 text-green-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-300">
                      ✓
                    </span>
                    Complete toolbox ecosystem
                  </li>
                  <li className="flex items-start gap-3 text-green-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-300">
                      ✓
                    </span>
                    Official MathWorks support
                  </li>
                  <li className="flex items-start gap-3 text-red-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-red-300">
                      ✕
                    </span>
                    No built-in file versioning
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* What works today */}
        <section className="py-16 md:py-24 lg:py-32">
          <div className="mx-auto max-w-5xl text-center mb-10 space-y-3">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">What works today</h2>
            <p className="text-muted-foreground text-lg">Core matrix workflows, plotting, and debugging ship today. Here is what is in progress.</p>
          </div>
          <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-2">
            <Card className="border border-green-500/30 bg-[#0E1421] shadow-lg">
              <CardHeader className="flex flex-row items-center gap-3 border-b border-border/60">
                <CheckCircle className="h-4 w-4 text-green-300" />
                <CardTitle className="text-xl text-gray-100">Works well</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-gray-300 text-lg">
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Matrix and array operations (indexing, slicing, reshaping)</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Arithmetic, logical, and relational operators</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Control flow (if/else, for, while, switch)</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>User-defined functions with multiple outputs</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Cells, structs, and basic classdef OOP</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>300+ built-in functions</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Interactive 2D and 3D plotting (rotate, zoom, pan)</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Real-time type and shape tracking (hover to see matrix dimensions)</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Live syntax validation (red underlines for dimension mismatches and errors)</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Execution tracing and diagnostic logging</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Async code execution (non-blocking runs)</p>
                </div>
              </CardContent>
            </Card>
            <Card className="border border-amber-500/30 bg-[#0E1421] shadow-lg">
              <CardHeader className="flex flex-row items-center gap-3 border-b border-border/60">
                <Clock className="h-4 w-4 text-amber-300" />
                <CardTitle className="text-xl text-gray-100">Limitations &amp; future work</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-gray-300 text-lg">
                <div>
                  <h4 className="text-lg font-semibold text-gray-100 mb-2">In progress</h4>
                  <div className="space-y-2">
                    <div className="flex items-start gap-2">
                      <span className="mt-1 text-amber-300">•</span>
                      <p>Advanced plotting (subplots, additional chart types, figure handles)</p>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="mt-1 text-amber-300">•</span>
                      <p>Extensible package support (signal processing, optimization, etc.)</p>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="mt-1 text-amber-300">•</span>
                      <p>Some edge-case MATLAB semantics</p>
                    </div>
                  </div>
                </div>
                <div className="border-t border-border/60 pt-4">
                  <h4 className="text-lg font-semibold text-gray-100 mb-2">Not supported</h4>
                  <div className="space-y-2">
                    <div className="flex items-start gap-2">
                      <span className="mt-1 text-slate-300">•</span>
                      <p>Simulink or graphical block diagrams</p>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="mt-1 text-slate-300">•</span>
                      <p>MATLAB-specific file formats (.slx, .mlapp)</p>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="mt-1 text-slate-300">•</span>
                      <p>Java/COM interop</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
          <p className="mx-auto mt-6 max-w-4xl text-lg text-muted-foreground text-center">
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
            </Link>
            .
          </p>
        </section>

        {/* Enterprise */}
        <section className="py-16 md:py-24 lg:py-32">
          <div className="mx-auto max-w-5xl">
            <div className="rounded-2xl border border-purple-500/30 bg-gradient-to-br from-purple-500/15 via-[#0E1421] to-[#0A0F1C] p-8 shadow-lg">
              <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-6">Built for teams</h2>
              <div className="grid gap-6 md:grid-cols-2">
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
                    <p className="text-base text-gray-300">Self-hosted, air-gapped option for export-controlled environments.</p>
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
                    <p className="text-xl font-medium text-gray-100">SOC 2 ready</p>
                    <p className="text-base text-gray-300">Built to SOC 2 standards. Audit planned for Q2 2026.</p>
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
        </section>

        {/* FAQs */}
        <section className="py-16 md:py-24 lg:py-32">
          <div className="mx-auto max-w-4xl text-center mb-8 space-y-3">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">Frequently asked questions</h2>
            <p className="text-muted-foreground text-lg">
              Common questions about RunMat and MATLAB compatibility.
            </p>
          </div>
          <div className="mx-auto grid max-w-5xl gap-4 md:grid-cols-2">
            {faqItems.map(item => (
              <details
                key={item.question}
                className="group self-start rounded-xl border border-border/60 bg-[#0E1421] shadow-lg"
              >
                <summary className="flex cursor-pointer list-none items-center justify-between px-6 py-4 text-gray-100">
                  <span className="text-lg font-medium">{item.question}</span>
                  <span className="text-gray-400 transition-transform duration-200 group-open:rotate-180">
                    ⌄
                  </span>
                </summary>
                <div className="px-6 pb-4 text-lg text-gray-300">
                  {item.answerContent ?? item.answer}
                </div>
              </details>
            ))}
          </div>
        </section>

        {/* Final CTA */}
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
                    data-ph-capture-attribute-source="matlab-online-bottom-cta"
                    data-ph-capture-attribute-cta="launch-sandbox"
                  >
                    Launch the sandbox
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
