import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import MatlabInlineCodeBlock from "@/components/MatlabInlineCodeBlock";
import {
  BookOpen,
  BarChart3,
  FileText,
  Info,
  Globe,
  Code2,
  Zap,
  CheckCircle,
  Clock,
  XCircle,
  AlertTriangle,
} from "lucide-react";

export const metadata: Metadata = {
  title: "Free Online MATLAB Alternative with GPU Acceleration",
  description:
    "Run MATLAB-style code online with RunMat. Browser-native execution, GPU acceleration, and no license required. Try the sandbox in seconds.",
  alternates: { canonical: "https://runmat.org/run-matlab-online" },
};

const faqItems = [
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
      "The RunMat runtime is open source under the MIT license. You can run MATLAB-syntax code in the browser or desktop without usage fees or time limits.",
  },
  {
    question: "Do I need to create an account?",
    answer:
      "No. The browser sandbox works immediately without sign-up. Creating an account is optional and only needed if you want cloud file persistence.",
  },
  {
    question: "Does RunMat work offline?",
    answer:
      "Yes. Once the sandbox page loads, it can run code without an internet connection. For full offline use with local file access, use the RunMat desktop app.",
  },
  {
    question: "How does RunMat run code in the browser?",
    answer:
      "RunMat compiles to WebAssembly, which runs natively in your browser at near-native speed. For GPU-accelerated operations, it uses WebGPU (available in Chrome and Edge).",
  },
  {
    question: "Is my code private?",
    answer:
      "Yes. Your code runs entirely on your device. Nothing is sent to a server unless you explicitly choose to save files to the cloud.",
  },
  {
    question: "Can RunMat use my GPU?",
    answer:
      "Yes, in browsers that support WebGPU (currently Chrome and Edge on most systems). RunMat automatically offloads eligible operations to the GPU for faster execution.",
  },
  {
    question: "What’s the difference between RunMat and GNU Octave?",
    answer:
      "Both run MATLAB-style syntax, but RunMat is designed for performance (JIT compilation, GPU acceleration) and runs natively in the browser. Octave is a mature desktop application with broader toolbox compatibility but no browser-native execution.",
  },
  {
    question: "Is there a desktop version?",
    answer:
      "Yes. The RunMat desktop app provides the same interface as the browser sandbox with full local file system access. See the download page.",
  },
];

const jsonLd = {
  "@context": "https://schema.org",
  "@graph": [
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
    },
    {
      "@type": "WebPage",
      "@id": "https://runmat.org/run-matlab-online#webpage",
      url: "https://runmat.org/run-matlab-online",
      name: "Free Online MATLAB Alternative with GPU Acceleration",
      description:
        "Run MATLAB-style code online with RunMat. Browser-native execution, GPU acceleration, and no license required. Try the sandbox in seconds.",
      inLanguage: "en",
      isPartOf: { "@id": "https://runmat.org/#website" },
      breadcrumb: { "@id": "https://runmat.org/run-matlab-online#breadcrumb" },
      author: { "@id": "https://runmat.org/#organization" },
      publisher: { "@id": "https://runmat.org/#organization" },
      mainEntity: [
        { "@id": "https://runmat.org/run-matlab-online#faq" },
        { "@id": "https://runmat.org/run-matlab-online#howto" },
        { "@id": "https://runmat.org/run-matlab-online#software" },
      ],
    },
    {
      "@type": "BreadcrumbList",
      "@id": "https://runmat.org/run-matlab-online#breadcrumb",
      itemListElement: [
        { "@type": "ListItem", position: 1, name: "Home", item: "https://runmat.org" },
        {
          "@type": "ListItem",
          position: 2,
          name: "MATLAB Online",
          item: "https://runmat.org/run-matlab-online",
        },
      ],
    },
    {
      "@type": "SoftwareApplication",
      "@id": "https://runmat.org/run-matlab-online#software",
      name: "RunMat",
      description:
        "RunMat is a high-performance, open-source runtime for math that runs MATLAB-syntax code in the browser with GPU acceleration and no license required.",
      applicationCategory: "ScientificApplication",
      applicationSubCategory: "EngineeringApplication",
      operatingSystem: ["Browser", "Windows", "macOS", "Linux"],
      offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
      url: "https://runmat.org/sandbox",
      author: { "@id": "https://runmat.org/#organization" },
      publisher: { "@id": "https://runmat.org/#organization" },
      mainEntityOfPage: { "@id": "https://runmat.org/run-matlab-online#webpage" },
    },
    {
      "@type": "FAQPage",
      "@id": "https://runmat.org/run-matlab-online#faq",
      mainEntityOfPage: { "@id": "https://runmat.org/run-matlab-online#webpage" },
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
      "@id": "https://runmat.org/run-matlab-online#howto",
      name: "Run MATLAB-style code in your browser",
      mainEntityOfPage: { "@id": "https://runmat.org/run-matlab-online#webpage" },
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

      <div className="container mx-auto px-4 md:px-6 lg:px-8 py-16 md:py-20">
        {/* Hero */}
        <section className="w-full mb-16" id="hero">
          <div className="grid grid-cols-1 gap-10 lg:grid-cols-2 lg:items-center">
            <div className="flex flex-col space-y-6 text-left items-start">
              <div className="mb-2 p-0 text-lg font-semibold uppercase tracking-wide text-primary">
                MATLAB Online Alternative
              </div>
              <h1 className="font-heading text-left leading-tight tracking-tight text-[clamp(2.6rem,4.8vw,4.25rem)] sm:text-[clamp(3rem,4vw,5rem)] lg:text-[clamp(3.25rem,3.6vw,5.25rem)]">
                Run Your MATLAB Code in the Browser Blazing Fast
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
                    data-ph-capture-attribute-source="run-matlab-online-hero"
                    data-ph-capture-attribute-cta="try-runmat-browser"
                  >
                    Try RunMat in Your Browser →
                  </Link>
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  asChild
                  className="h-12 px-8 text-base bg-[#0E1B1E] dark:bg-[#0E1B1E] border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100"
                >
                  <Link href="/docs/desktop-browser-guide">View Getting Started</Link>
                </Button>
              </div>
            </div>
            <div className="rounded-xl border border-border bg-muted/40 p-2 bg-[radial-gradient(ellipse_at_top,_rgba(124,58,237,0.25),_transparent_60%)]">
              <Image
                src="https://web.runmatstatic.com/matlab-online-product-screenshot.png"
                alt="RunMat MATLAB-style code example"
                width={1200}
                height={800}
                className="w-full h-auto rounded-lg"
                priority
              />
            </div>
          </div>
        </section>

        {/* What is MATLAB Online? */}
        <section className="mb-16 border-y border-border/60 py-12">
          <div className="mx-auto max-w-4xl text-center space-y-4">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">What is MATLAB Online?</h2>
            <p className="text-muted-foreground text-lg">
              MATLAB Online is MathWorks&apos; official browser-based interface for running MATLAB code. It&apos;s used by
              students, engineers, and researchers for numerical computing, data analysis, and algorithm development.
            </p>
          </div>
        </section>

        {/* Challenges with MATLAB Online */}
        <section className="mb-16">
          <div className="mx-auto max-w-5xl rounded-xl border border-amber-500/30 bg-gradient-to-r from-amber-500/15 to-transparent shadow-lg">
            <div className="px-6 py-8">
              <div className="flex flex-col gap-6">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="mt-1 h-5 w-5 text-amber-300" />
                  <div>
                    <h3 className="text-xl font-semibold text-foreground">But there&apos;s a challenge…</h3>
                    <p className="text-lg text-muted-foreground mt-2">
                      MATLAB Online requires a MathWorks account, and your code runs on their cloud servers, not on your
                      local machine. Many engineers run into these limitations:
                    </p>
                  </div>
                </div>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="rounded-lg border border-amber-500/30 bg-[#0E1421] px-5 py-4">
                    <p className="text-xl font-semibold text-foreground">Account barriers</p>
                    <p className="text-lg text-muted-foreground mt-2">
                      Sign-up process and license requirements create unnecessary friction for quick tasks.
                    </p>
                  </div>
                  <div className="rounded-lg border border-amber-500/30 bg-[#0E1421] px-5 py-4">
                    <p className="text-xl font-semibold text-foreground">Idle timeouts &amp; hour caps</p>
                    <p className="text-lg text-muted-foreground mt-2">
                      Sessions timeout after 15 minutes of inactivity. Free tier is capped at 20 hours/month.
                    </p>
                  </div>
                  <div className="rounded-lg border border-amber-500/30 bg-[#0E1421] px-5 py-4">
                    <p className="text-xl font-semibold text-foreground">Cloud dependency</p>
                    <p className="text-lg text-muted-foreground mt-2">
                      Code must be uploaded to remote servers, raising privacy and connectivity concerns.
                    </p>
                  </div>
                  <div className="rounded-lg border border-amber-500/30 bg-[#0E1421] px-5 py-4">
                    <p className="text-xl font-semibold text-foreground">No local GPU access</p>
                    <p className="text-lg text-muted-foreground mt-2">
                      Code runs on MathWorks&apos; servers, so you cannot use your own GPU for acceleration.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* RunMat as an alternative */}
        <section className="mb-16">
          <div className="mx-auto max-w-4xl space-y-4 text-center">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">
              Meet RunMat: A Free Alternative
            </h2>
            <p className="text-muted-foreground text-lg">
              RunMat is an open-source runtime that understands MATLAB syntax and runs it directly in your browser.
              Your code executes on your own device via WebAssembly, and GPU code acceleration in browers that support
              WebGPU.
            </p>
          </div>
          <div className="mx-auto mt-10 max-w-5xl space-y-6">
            <div className="rounded-2xl border border-purple-500/30 bg-gradient-to-br from-purple-500/15 via-[#0E1421] to-[#0A0F1C] p-8 shadow-lg">
              <div className="text-center space-y-2">
                <h3 className="text-xl md:text-2xl font-semibold text-foreground">
                  Key differences from MATLAB Online
                </h3>
              </div>
              <div className="mt-8 grid gap-6 md:grid-cols-2">
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-200">
                    <CheckCircle className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-foreground">No account required</p>
                    <p className="text-lg text-muted-foreground">Open the sandbox and start coding immediately.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-200">
                    <Clock className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-foreground">No usage limits</p>
                    <p className="text-lg text-muted-foreground">
                      No monthly caps, no execution timeouts, no idle disconnects.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-200">
                    <Code2 className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-foreground">Client-side execution</p>
                    <p className="text-lg text-muted-foreground">
                      Your code runs locally in your browser, not on a remote server.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-200">
                    <Zap className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-foreground">GPU acceleration</p>
                    <p className="text-lg text-muted-foreground">
                      WebGPU-enabled browsers can accelerate computations automatically.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-200">
                    <Info className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-foreground">Free &amp; open source</p>
                    <p className="text-lg text-muted-foreground">
                      MIT-licensed runtime, community-developed, and always accessible.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-200">
                    <Globe className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-foreground">Works offline</p>
                    <p className="text-lg text-muted-foreground">
                      After initial load, runs without internet connection.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-200">
                    <AlertTriangle className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-foreground">Live syntax validation</p>
                    <p className="text-lg text-muted-foreground">Red underlines highlight errors before you run.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <span className="mt-1 inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-200">
                    <FileText className="h-5 w-5" />
                  </span>
                  <div>
                    <p className="text-xl font-medium text-foreground">Tracing &amp; logging</p>
                    <p className="text-lg text-muted-foreground">Detailed execution trace and console output.</p>
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
                    data-ph-capture-attribute-source="run-matlab-online-features"
                    data-ph-capture-attribute-cta="try-runmat-browser"
                  >
                    Try RunMat in Your Browser →
                  </Link>
                </Button>
              </div>
            </div>
            <div className="grid gap-6 md:grid-cols-2">
              <Card className="border border-border/60 bg-[#0E1421] shadow-lg">
                <CardHeader className="border-b border-border/60">
                  <CardTitle className="text-xl">Compatibility and coverage</CardTitle>
                </CardHeader>
                <CardContent className="text-lg text-muted-foreground space-y-3">
                  <p>
                    RunMat focuses on MATLAB-compatible syntax for core numerical workflows. See the{" "}
                    <Link href="/docs/language-coverage" className="underline">
                      language coverage guide
                    </Link>{" "}
                    and{" "}
                    <Link href="/docs/matlab-function-reference" className="underline">
                      function reference
                    </Link>{" "}
                    for current support.
                  </p>
                </CardContent>
              </Card>
              <Card className="border border-border/60 bg-[#0E1421] shadow-lg">
                <CardHeader className="border-b border-border/60">
                  <CardTitle className="text-xl">Performance context</CardTitle>
                </CardHeader>
                <CardContent className="text-lg text-muted-foreground space-y-3">
                  <p>
                    Review the{" "}
                    <Link href="/benchmarks" className="underline">
                      benchmarks
                    </Link>{" "}
                    and{" "}
                    <Link href="/docs/how-it-works" className="underline">
                      how RunMat works
                    </Link>{" "}
                    under the hood for detailed performance insights.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
          <div className="mx-auto mt-8 grid max-w-5xl gap-6 md:grid-cols-3">
            <Link href="/docs" className="block">
              <Card className="h-full border border-border/60 bg-[#0E1421] shadow-lg">
                <CardHeader className="space-y-3">
                  <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-100">
                    <BookOpen className="h-5 w-5" />
                  </span>
                  <CardTitle className="flex items-center gap-2">
                    Docs <span className="text-muted-foreground">→</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-lg text-muted-foreground">
                  Language coverage, function reference, and how RunMat works.
                </CardContent>
              </Card>
            </Link>
            <Link href="/benchmarks" className="block">
              <Card className="h-full border border-border/60 bg-[#0E1421] shadow-lg">
                <CardHeader className="space-y-3">
                  <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-100">
                    <BarChart3 className="h-5 w-5" />
                  </span>
                  <CardTitle className="flex items-center gap-2">
                    Benchmarks <span className="text-muted-foreground">→</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-lg text-muted-foreground">
                  Reproducible performance results across common math workloads.
                </CardContent>
              </Card>
            </Link>
            <Link href="/blog" className="block">
              <Card className="h-full border border-border/60 bg-[#0E1421] shadow-lg">
                <CardHeader className="space-y-3">
                  <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-100">
                    <FileText className="h-5 w-5" />
                  </span>
                  <CardTitle className="flex items-center gap-2">
                    Blog <span className="text-muted-foreground">→</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-lg text-muted-foreground">
                  Practical guides, comparisons, and release notes.
                </CardContent>
              </Card>
            </Link>
          </div>
        </section>

        {/* How it works */}
        <section className="mb-16">
          <div className="mx-auto max-w-4xl text-center mb-10 space-y-3">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">How It Works</h2>
            <p className="text-muted-foreground text-lg">Get started in three simple steps.</p>
          </div>
          <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-3 md:items-stretch">
            <Card className="border border-border/60 bg-[#0E1421] shadow-lg overflow-hidden flex flex-col h-full">
              <div className="bg-gradient-to-r from-purple-500/20 to-transparent p-6 flex items-start justify-start h-[200px]">
                <Link
                  href="https://runmat.org/sandbox"
                  target="_blank"
                  rel="noreferrer"
                  className="rounded-lg border border-purple-500/30 bg-purple-500/10 text-purple-100 px-3 py-2 text-base flex items-center gap-2 transition hover:border-purple-400/60 hover:text-purple-50"
                  data-ph-capture-attribute-destination="sandbox"
                  data-ph-capture-attribute-source="run-matlab-online-how-it-works"
                  data-ph-capture-attribute-cta="runmat-org-sandbox"
                >
                  <span className="h-2 w-2 rounded-full bg-red-400"></span>
                  <span className="h-2 w-2 rounded-full bg-yellow-400"></span>
                  <span className="h-2 w-2 rounded-full bg-green-400"></span>
                  <span className="ml-2">runmat.org/sandbox</span>
                </Link>
              </div>
              <CardContent className="pt-6 space-y-2 flex-1">
                <div className="inline-flex h-7 w-7 items-center justify-center rounded-full border border-purple-500/40 bg-purple-500/10 text-purple-100 text-base">
                  1
                </div>
                <h3 className="text-xl font-semibold text-foreground">Open the Sandbox</h3>
                <p className="text-lg text-muted-foreground">
                  Click the button to launch RunMat in your browser. No downloads or sign-ups required.
                </p>
              </CardContent>
            </Card>
            <Card className="border border-border/60 bg-[#0E1421] shadow-lg overflow-hidden flex flex-col h-full">
              <div className="bg-gradient-to-r from-purple-500/20 to-transparent p-6 flex flex-col items-start justify-start gap-3 h-[200px]">
                <MatlabInlineCodeBlock
                  code={"A = [3 1 3 2];\nC = unique(A);"}
                  showRunButton
                  preClassName="[&_code]:text-[0.75rem] [&_code]:leading-relaxed"
                />
              </div>
              <CardContent className="pt-6 space-y-2 flex-1">
                <div className="inline-flex h-7 w-7 items-center justify-center rounded-full border border-purple-500/40 bg-purple-500/10 text-purple-100 text-base">
                  2
                </div>
                <h3 className="text-xl font-semibold text-foreground">Write or Paste Code</h3>
                <p className="text-lg text-muted-foreground">
                  Type your MATLAB-style code directly, or paste existing scripts from your projects.
                </p>
              </CardContent>
            </Card>
            <Card className="border border-border/60 bg-[#0E1421] shadow-lg overflow-hidden flex flex-col h-full">
              <div className="bg-gradient-to-r from-purple-500/20 to-transparent p-6 flex items-start justify-start h-[200px]">
                <MatlabInlineCodeBlock
                  code={"C =\n\n     1     2     3"}
                  preClassName="[&_code]:text-[0.75rem] [&_code]:leading-relaxed"
                />
              </div>
              <CardContent className="pt-6 space-y-2 flex-1">
                <div className="inline-flex h-7 w-7 items-center justify-center rounded-full border border-purple-500/40 bg-purple-500/10 text-purple-100 text-base">
                  3
                </div>
                <h3 className="text-xl font-semibold text-foreground">Run and See Results</h3>
                <p className="text-lg text-muted-foreground">
                  Execute your code instantly. View outputs, plots, and results in real time.
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Comparison */}
        <section className="mb-16">
          <div className="mx-auto max-w-5xl text-center mb-10 space-y-3">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">RunMat vs. MATLAB Online</h2>
            <p className="text-muted-foreground text-lg">A clear comparison of capabilities and requirements.</p>
          </div>
          <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-2">
            <Card className="border border-purple-500/30 bg-[#0E1421] shadow-lg">
              <CardContent className="p-6 space-y-4">
                <div>
                  <h3 className="text-xl font-semibold text-purple-200">RunMat</h3>
                  <p className="text-lg text-muted-foreground">High-performance, open-source runtime for math</p>
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
                    WebGPU acceleration
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
                    Live syntax validation
                  </li>
                  <li className="flex items-start gap-3 text-green-300">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-green-300">
                      ✓
                    </span>
                    Execution tracing &amp; logging
                  </li>
                  <li className="flex items-start gap-3 text-muted-foreground">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-muted-foreground">
                      –
                    </span>
                    Limited package / toolbox support
                  </li>
                  <li className="flex items-start gap-3 text-muted-foreground">
                    <span className="mt-0.5 inline-flex items-center justify-center text-base text-muted-foreground">
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
                  <h3 className="text-xl font-semibold text-foreground">MATLAB Online</h3>
                  <p className="text-lg text-muted-foreground">MathWorks official platform</p>
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
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* What works today */}
        <section className="mb-16">
          <div className="mx-auto max-w-5xl text-center mb-10 space-y-3">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">What Works Today</h2>
           
          </div>
          <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-2">
            <Card className="border border-green-500/30 bg-[#0E1421] shadow-lg">
              <CardHeader className="flex flex-row items-center gap-3 border-b border-border/60">
                <CheckCircle className="h-4 w-4 text-green-300" />
                <CardTitle className="text-xl">Works Well</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-muted-foreground text-lg">
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
                  <p>Simple 2D plotting (line, scatter)</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Live syntax validation (red underlines for errors)</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="mt-1 text-green-300">•</span>
                  <p>Execution tracing and console logging</p>
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
                <CardTitle className="text-xl">Limitations &amp; Future Work</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-muted-foreground text-lg">
                <div>
                  <h4 className="text-lg font-semibold text-muted-foreground mb-2">In Progress</h4>
                  <div className="space-y-2">
                    <div className="flex items-start gap-2">
                      <span className="mt-1 text-amber-300">•</span>
                      <p>Advanced plotting (3D, surfaces, subplots, figure handles)</p>
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
                  <h4 className="text-lg font-semibold text-muted-foreground mb-2">Not Supported</h4>
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

        {/* FAQs */}
        <section className="mb-16">
          <div className="mx-auto max-w-4xl text-center mb-8 space-y-3">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">Frequently Asked Questions</h2>
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
                <summary className="flex cursor-pointer list-none items-center justify-between px-6 py-4 text-foreground">
                  <span className="text-lg font-medium">{item.question}</span>
                  <span className="text-muted-foreground transition-transform duration-200 group-open:rotate-180">
                    ⌄
                  </span>
                </summary>
                <div className="px-6 pb-4 text-lg text-muted-foreground">
                  {item.answer}
                </div>
              </details>
            ))}
          </div>
        </section>

        {/* Final CTA */}
        <section className="mb-8 text-center">
          <Card className="mx-auto max-w-3xl border border-purple-500/30 bg-gradient-to-r from-purple-500/10 to-blue-500/10 shadow-lg">
            <CardContent className="py-8 space-y-4">
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">
                Run MATLAB code online - no install, no license
              </h2>
              <p className="text-muted-foreground text-lg">
                Start running math immediately in your browser.
              </p>
              <div className="flex justify-center">
                <Button
                  size="lg"
                  asChild
                  className="h-12 px-8 text-base font-semibold bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-xl border-0 transition-all duration-200"
                >
                  <Link
                    href="/sandbox"
                    data-ph-capture-attribute-destination="sandbox"
                    data-ph-capture-attribute-source="run-matlab-online-bottom-cta"
                    data-ph-capture-attribute-cta="launch-sandbox"
                  >
                    Launch the Sandbox
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </section>
      </div>
    </div>
  );
}

