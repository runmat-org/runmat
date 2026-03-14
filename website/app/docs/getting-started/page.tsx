import type { Metadata } from "next";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  Download, 
  Terminal, 
  FileText,
  ArrowRight,
  Zap,
  ExternalLink,
  BarChart3,
  Check,
  X,
} from "lucide-react";
import Link from "next/link";
import { OSInstallCommand } from "@/components/OSInstallCommand";
import { GettingStartedTabs } from "@/components/GettingStartedTabs";

const pageTitle = "Getting Started | Docs";
const pageDescription =
  "Get started with RunMat: interactive plotting, real-time diagnostics, and GPU acceleration for MATLAB-style code. Browser or CLI.";

export const metadata: Metadata = {
  title: pageTitle,
  description: pageDescription,
  alternates: { canonical: "https://runmat.com/docs/getting-started" },
  keywords: [
    "RunMat getting started", "MATLAB alternative setup", "RunMat CLI install",
    "RunMat browser IDE", "GPU acceleration tutorial", "RunMat sandbox",
  ],
  openGraph: {
    title: pageTitle,
    description: pageDescription,
    url: "/docs/getting-started",
    siteName: "RunMat",
    type: "article",
  },
  twitter: {
    card: "summary_large_image",
    title: pageTitle,
    description: pageDescription,
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

const jsonLd = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "WebPage",
      "@id": "https://runmat.com/docs/getting-started#webpage",
      url: "https://runmat.com/docs/getting-started",
      name: pageTitle,
      description: pageDescription,
      inLanguage: "en",
      dateModified: "2026-03-05",
      isPartOf: { "@id": "https://runmat.com/#website" },
      author: { "@id": "https://runmat.com/#organization" },
      publisher: { "@id": "https://runmat.com/#organization" },
    },
    {
      "@type": "HowTo",
      name: "Get started with RunMat",
      description: pageDescription,
      step: [
        {
          "@type": "HowToStep",
          name: "Try RunMat in your browser",
          text: "Visit runmat.com/sandbox to open the browser-based IDE. No installation or account required.",
          url: "https://runmat.com/sandbox",
        },
        {
          "@type": "HowToStep",
          name: "Install the CLI",
          text: "Run 'curl -fsSL https://runmat.com/install | sh' to install the RunMat CLI for native GPU performance and local file access.",
          url: "https://runmat.com/download",
        },
        {
          "@type": "HowToStep",
          name: "Set up Jupyter",
          text: "Run 'runmat --install-kernel' to use RunMat as a Jupyter kernel for interactive notebooks.",
          url: "https://runmat.com/docs/getting-started#jupyter-notebook-integration",
        },
      ],
    },
  ],
};

export default function GettingStartedPage() {
  return (
    <div className="min-h-screen bg-background">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(jsonLd).replace(/<\//g, "<\\/"),
        }}
      />
      <div className="container mx-auto px-4 md:px-0 py-16 md:py-4">
        
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl lg:text-5xl font-bold mb-6">
            Getting Started with RunMat
          </h1>
          <p className="text-xl text-muted-foreground leading-relaxed mb-6">
            Get up and running in minutes. Try RunMat in your browser with no installation, or install the CLI for the terminal and local scripts.
          </p>
          {/* Try RunMat Now — primary CTA */}
          <Button
            asChild
            size="lg"
            className="h-12 px-8 text-base font-semibold bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-xl border-0 transition-all duration-200 hover:from-blue-600 hover:to-purple-700"
          >
            <Link
              href="/sandbox"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center"
              data-ph-capture-attribute-destination="sandbox"
              data-ph-capture-attribute-source="docs-getting-started-hero"
              data-ph-capture-attribute-cta="launch-browser-app"
            >
              Launch Browser App
              <ExternalLink className="ml-2 h-4 w-4" aria-hidden="true" />
            </Link>
          </Button>
          <p className="text-sm text-muted-foreground mt-2">
            No installation required. Works in Chrome, Edge, Firefox, and Safari.
          </p>
        </div>

        {/* Getting Started Tabs: Browser | CLI | Jupyter */}
        <section id="getting-started-tabs" className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">
            Choose your path
          </h2>
          <GettingStartedTabs />
        </section>

        {/* Feature comparison */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">
            Compare surfaces
          </h2>
          <div className="rounded-xl border border-border/60 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[540px] text-sm">
                <thead>
                  <tr className="border-b border-border/40">
                    <th className="text-left py-4 px-5 text-sm font-medium text-muted-foreground w-[34%]">Feature</th>
                    <th className="text-center py-4 px-3 w-[22%]">
                      <div className="text-sm font-semibold text-foreground">CLI</div>
                    </th>
                    <th className="text-center py-4 px-3 w-[22%]">
                      <div className="text-sm font-semibold text-foreground">Sandbox</div>
                      <div className="text-xs font-normal text-muted-foreground mt-0.5">No account</div>
                    </th>
                    <th className="text-center py-4 px-3 w-[22%]">
                      <div className="text-sm font-semibold text-foreground">Sandbox + Cloud</div>
                      <div className="text-xs font-normal text-muted-foreground mt-0.5">Signed in</div>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {([
                    ["Works without install", "no", "yes", "yes"],
                    ["Account required", "no", "no", "yes-text:Yes (free)"],
                    ["GPU acceleration", "text:Native (Metal, Vulkan, DX12)", "text:WebGPU (browser-throttled)", "text:WebGPU (browser-throttled)"],
                    ["Interactive IDE", "no", "yes", "yes"],
                    ["Interactive plotting", "yes-text:GUI window", "yes-text:In-editor", "yes-text:In-editor"],
                    ["Variable inspector", "no", "yes", "yes"],
                    ["File storage", "text:Local filesystem", "text:In-memory (cleared on tab close)", "text:Cloud (persists across sessions)"],
                    ["File versioning", "text:No (use git)", "no", "yes-text:Automatic"],
                    ["Project sharing", "no", "no", "text:Paid plans"],
                    ["Jupyter support", "yes", "no", "no"],
                    ["Offline support", "yes", "no", "no"],
                  ] as [string, string, string, string][]).map(([feature, cli, sandbox, cloud]) => (
                    <tr key={feature} className="border-b border-border/30">
                      <td className="py-3 px-5 text-muted-foreground">{feature}</td>
                      {[cli, sandbox, cloud].map((cell, i) => (
                        <td key={`${feature}-${i}`} className="py-3 px-3 text-center">
                          {cell === "yes" ? (
                            <Check className="inline-block h-4 w-4 text-green-500" aria-label="Yes" />
                          ) : cell === "no" ? (
                            <X className="inline-block h-4 w-4 text-muted-foreground/40" aria-label="No" />
                          ) : cell.startsWith("yes-text:") ? (
                            <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                              <Check className="h-4 w-4 text-green-500 shrink-0" aria-hidden />
                              {cell.replace("yes-text:", "")}
                            </span>
                          ) : (
                            <span className="text-xs text-muted-foreground">{cell.replace("text:", "")}</span>
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* Next Steps */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">
            Next steps
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Download className="h-5 w-5 mr-2 text-foreground" />
                  Want local file access?
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Install the CLI for native GPU performance and local file access.
                </p>
                <Button variant="outline" className="w-full" asChild>
                  <Link href="/download" className="flex items-center justify-center">
                    Install the CLI
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <BarChart3 className="h-5 w-5 mr-2 text-blue-600" />
                  Plotting &amp; diagnostics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Interactive 2D and 3D plots, real-time shape tracking, and dimension mismatch warnings — all built into the editor.
                </p>
                <Button variant="outline" className="w-full" asChild>
                  <Link href="/docs/desktop-browser-guide" className="flex items-center justify-center">
                    Browser guide
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <FileText className="h-5 w-5 mr-2 text-blue-600" />
                  Learn the fundamentals
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Dive deeper into how RunMat compiles and accelerates your code.
                </p>
                <Button variant="outline" className="w-full" asChild>
                  <Link href="/docs/how-it-works" className="flex items-center justify-center">
                    How RunMat works
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Terminal className="h-5 w-5 mr-2 text-green-600 dark:text-green-400" />
                  Explore examples
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  See RunMat in action with real-world examples.
                </p>
                <Button variant="outline" className="w-full" asChild>
                  <Link href="/benchmarks" className="flex items-center justify-center">
                    Benchmarks
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Zap className="h-5 w-5 mr-2 text-orange-600" />
                  Discover RunMat on the GPU
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  How RunMat turns MATLAB scripts into GPU-accelerated workloads
                </p>
                <Button variant="outline" className="w-full" asChild>
                  <Link href="/docs/accelerate/fusion-intro" className="flex items-center justify-center">
                    RunMat Accelerate
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Zap className="h-5 w-5 mr-2 text-purple-600" />
                  Understand the design
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Learn why RunMat keeps a slim core and package-first model.
                </p>
                <Button variant="outline" className="w-full" asChild>
                  <Link href="/docs/design-philosophy" className="flex items-center justify-center">
                    Design philosophy
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Help */}
        <section>
          <Card className="bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800">
            <CardContent className="p-6">
              <h3 className="text-lg font-semibold mb-3 text-foreground">
                Need help?
              </h3>
              <p className="text-muted-foreground mb-4">
                Join our community and get support from other RunMat users and developers.
              </p>
              <div className="flex flex-col sm:flex-row gap-3">
                <Button variant="outline" size="sm" asChild>
                  <Link href="https://github.com/runmat-org/runmat/discussions">
                    GitHub Discussions
                  </Link>
                </Button>
                <Button variant="outline" size="sm" asChild>
                  <Link href="https://github.com/runmat-org/runmat/issues">
                    Report Issues
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