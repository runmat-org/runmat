import { Metadata } from "next";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  Package,
  Code,
  Zap,
  Cpu,
  Monitor,
} from "lucide-react";
import { SiGithub } from "react-icons/si";
import { OSInstallCommand } from "@/components/OSInstallCommand";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Download RunMat",
  description: "Download RunMat for your platform. Fast, free, and open source MATLAB-compatible runtime with V8-inspired performance.",
  alternates: { canonical: "https://runmat.com/download" },
  openGraph: {
    url: "https://runmat.com/download",
  },
};

export default function DownloadPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 md:px-6 lg:px-8">

        {/* Header */}
        <section className="w-full py-16 md:py-24 lg:py-32">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h1 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
              Download RunMat
            </h1>
            <p className="max-w-[42rem] leading-relaxed text-[0.938rem] text-foreground">
              Get started with RunMat in seconds. Free, open source, MIT licensed.
            </p>
          </div>

          {/* Quick Install */}
          <div className="mx-auto max-w-3xl">
            <Card className="bg-card border-border">
              <CardContent className="p-4 text-center">
                <OSInstallCommand className="max-w-4xl mx-auto shadow-none border-none" />
              </CardContent>
            </Card>
            <p className="text-center text-[0.938rem] text-foreground mt-6">
              Or <Link href="/sandbox" className="underline text-[hsl(var(--brand))] hover:text-[hsl(var(--brand))]/80">try RunMat in your browser</Link> with no installation.
            </p>
          </div>
        </section>

        {/* Alternative Installation Methods */}
        <section className="w-full py-16 md:py-24 lg:py-32">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
              Alternative installation methods
            </h2>
          </div>
          
          <div className="mx-auto max-w-5xl grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="rounded-lg border border-border bg-card p-6">
              <div className="flex items-center gap-2 mb-4">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground">
                  <Package className="h-5 w-5" />
                </span>
                <h3 className="text-lg font-semibold text-foreground">Homebrew (macOS/Linux)</h3>
              </div>
              <div className="space-y-4">
                <div>
                  <p className="text-sm font-medium text-foreground mb-2">One-liner install</p>
                  <div className="bg-[var(--editor-background)] text-white rounded-md p-3 font-mono text-sm">
                    brew install runmat-org/tap/runmat
                  </div>
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium text-foreground">Tap then install</p>
                  <div className="bg-[var(--editor-background)] text-white rounded-md p-3 font-mono text-sm">
                    brew tap runmat-org/tap
                  </div>
                  <div className="bg-[var(--editor-background)] text-white rounded-md p-3 font-mono text-sm">
                    brew install runmat
                  </div>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-border bg-card p-6">
              <div className="flex items-center gap-2 mb-4">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground">
                  <Cpu className="h-5 w-5" />
                </span>
                <h3 className="text-lg font-semibold text-foreground">Cargo (Rust)</h3>
              </div>
              <div className="space-y-4">
                <div>
                  <p className="text-sm font-medium text-foreground mb-2">Install from crates.io</p>
                  <div className="bg-[var(--editor-background)] text-white rounded-md p-3 font-mono text-sm">
                    cargo install runmat
                  </div>
                </div>
                <div>
                  <p className="text-sm font-medium text-foreground mb-2">From source (latest)</p>
                  <div className="bg-[var(--editor-background)] text-white rounded-md p-3 font-mono text-sm break-all">
                    cargo install --git https://github.com/runmat-org/runmat
                  </div>
                </div>
                <p className="text-sm text-muted-foreground">
                  Requires Rust 1.70+ with LLVM support
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Development Setup */}
        <section className="w-full py-16 md:py-24 lg:py-32">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
              Development environment
            </h2>
          </div>
          
          <div className="mx-auto max-w-5xl grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded-lg border border-border bg-card p-6 opacity-60">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <Monitor className="h-5 w-5" />
              </span>
              <div className="flex items-center gap-2 mb-1">
                <h3 className="text-lg font-semibold text-foreground">Desktop App</h3>
                <span className="text-xs text-muted-foreground border border-border rounded-md px-2 py-0.5">Coming soon</span>
              </div>
              <p className="text-[0.938rem] text-foreground mt-1">
                Full IDE with native GPU performance and local file system access.
              </p>
            </div>

            <div className="rounded-lg border border-border bg-card p-6 opacity-60">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <Code className="h-5 w-5" />
              </span>
              <div className="flex items-center gap-2 mb-1">
                <h3 className="text-lg font-semibold text-foreground">VS Code Extension</h3>
                <span className="text-xs text-muted-foreground border border-border rounded-md px-2 py-0.5">Coming soon</span>
              </div>
              <p className="text-[0.938rem] text-foreground mt-1">
                Syntax highlighting, IntelliSense, and integrated debugging for RunMat.
              </p>
            </div>

            <div className="rounded-lg border border-border bg-card p-6 opacity-60">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <Code className="h-5 w-5" />
              </span>
              <div className="flex items-center gap-2 mb-1">
                <h3 className="text-lg font-semibold text-foreground">IntelliJ Plugin</h3>
                <span className="text-xs text-muted-foreground border border-border rounded-md px-2 py-0.5">Coming soon</span>
              </div>
              <p className="text-[0.938rem] text-foreground mt-1">
                Syntax highlighting, IntelliSense, and integrated debugging for RunMat.
              </p>
            </div>
          </div>
        </section>

        {/* Next Steps */}
        <section className="w-full py-16 md:py-24 lg:py-32">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
              Next steps
            </h2>
          </div>
          
          <div className="mx-auto max-w-5xl grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="rounded-lg border border-border bg-card p-6">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <Code className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">Start coding</h3>
              <p className="text-[0.938rem] text-foreground mt-1 mb-4">
                Jump into the interactive REPL or run your existing MATLAB scripts.
              </p>
              <div className="bg-[var(--editor-background)] text-white rounded-md p-3 font-mono text-sm mb-4">
                runmat
              </div>
              <Button variant="outline" size="sm" asChild className="w-full rounded-none">
                <Link href="/docs/getting-started">
                  Getting started guide
                </Link>
              </Button>
            </div>
            <div className="rounded-lg border border-border bg-card p-6">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/10 text-foreground mb-3">
                <Zap className="h-5 w-5" />
              </span>
              <h3 className="text-lg font-semibold text-foreground">Set up Jupyter kernel</h3>
              <p className="text-[0.938rem] text-foreground mt-1 mb-4">
                Use RunMat as a Jupyter kernel for interactive notebooks.
              </p>
              <div className="bg-[var(--editor-background)] text-white rounded-md p-3 font-mono text-sm mb-4">
                runmat --install-kernel
              </div>
              <Button variant="outline" size="sm" asChild className="w-full rounded-none">
                <Link href="/docs/getting-started#jupyter-notebook-integration">
                  Jupyter setup guide
                </Link>
              </Button>
            </div>
          </div>
        </section>

        {/* Source Code */}
        <section className="w-full py-16 md:py-24 lg:py-32">
          <div className="mx-auto max-w-3xl text-center">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl mb-12">
              Open source
            </h2>
            <div className="rounded-lg border border-border bg-card p-8 space-y-4">
              <Link
                href="https://github.com/runmat-org/runmat"
                target="_blank"
                rel="noreferrer"
                className="mx-auto flex h-14 w-14 items-center justify-center rounded-full border border-border bg-secondary text-foreground hover:opacity-80 transition-opacity"
              >
                <SiGithub className="h-7 w-7" />
              </Link>
              <p className="text-[0.938rem] text-foreground">
                RunMat is completely open source. Explore the code, contribute, or build it yourself.
              </p>
              <div className="flex flex-col sm:flex-row gap-3 justify-center pt-1">
                <Button asChild className="rounded-none bg-[hsl(var(--brand))] text-white border-0 shadow-none hover:bg-[hsl(var(--brand))]/90">
                  <Link href="https://github.com/runmat-org/runmat" target="_blank" rel="noopener noreferrer">
                    <SiGithub className="mr-2 h-4 w-4" />
                    View on GitHub
                  </Link>
                </Button>
                <Button variant="ghost" asChild className="text-muted-foreground hover:text-foreground">
                  <Link href="https://crates.io/crates/runmat" target="_blank" rel="noopener noreferrer">
                    <Package className="mr-2 h-4 w-4" />
                    View on crates.io
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        </section>

      </div>
    </div>
  );
}
