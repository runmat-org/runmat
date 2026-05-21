import { Metadata } from "next";
import { Button } from "@/components/ui/button";
import {
  Package,
  Cpu,
  ArrowRight,
  Download as DownloadIcon,
  ChevronRight,
} from "lucide-react";
import { SiGithub } from "react-icons/si";
import { OSInstallCommand } from "@/components/OSInstallCommand";
import Link from "next/link";

const desktopDownloads: { os: string; arch: string; format: string; href: string }[] = [
  {
    os: "macOS",
    arch: "Apple Silicon",
    format: "DMG",
    href: "/download/latest?platform=darwin-aarch64",
  },
  {
    os: "macOS",
    arch: "Intel",
    format: "DMG",
    href: "/download/latest?platform=darwin-x86_64",
  },
  {
    os: "Windows",
    arch: "x86_64",
    format: "Installer",
    href: "/download/latest?platform=windows-x86_64",
  },
  {
    os: "Linux",
    arch: "x86_64",
    format: "AppImage",
    href: "/download/latest?platform=linux-x86_64",
  },
];

export const metadata: Metadata = {
  title: "Download RunMat | Desktop app and CLI",
  description:
    "Download the RunMat desktop app or install the free, open-source RunMat CLI. macOS, Linux, and Windows.",
  alternates: { canonical: "https://runmat.com/download" },
  openGraph: {
    title: "Download RunMat | Desktop app and CLI",
    description:
      "Download the RunMat desktop app or install the free, open-source RunMat CLI. macOS, Linux, and Windows.",
    url: "https://runmat.com/download",
  },
};

export default function DownloadPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 md:px-6">

        {/* Hero + install */}
        <section className="w-full pt-16 md:pt-24 lg:pt-32 pb-10 md:pb-12">
          <div className="mx-auto max-w-3xl space-y-5 text-center">
            <h1 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
              Download RunMat
            </h1>
            <p className="mx-auto max-w-[40rem] text-[0.938rem] leading-relaxed text-foreground">
              Get the RunMat desktop app for your machine, or install the open-source CLI runtime.
            </p>
          </div>

          <div className="mx-auto mt-10 max-w-3xl">
            <div className="rounded-lg border border-border bg-card p-5 sm:p-6">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="text-left">
                  <h2 className="text-sm font-semibold text-foreground">
                    Desktop app
                  </h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Download the latest signed installer for your current environment.
                  </p>
                </div>
                <Button
                  asChild
                  size="lg"
                  className="h-11 px-7 text-sm font-semibold rounded-none bg-[hsl(var(--brand))] text-white border-0 shadow-none hover:bg-[hsl(var(--brand))]/90"
                >
                  <Link href="/download/latest">
                    Download for this device
                    <ArrowRight className="ml-2 h-4 w-4" aria-hidden="true" />
                  </Link>
                </Button>
              </div>
            </div>

            <div className="mt-5 flex flex-col sm:flex-row items-center justify-center gap-3">
              <Button
                asChild
                size="lg"
                variant="outline"
                className="h-11 px-7 text-sm rounded-none bg-card border-border text-foreground"
              >
                <Link
                  href="/sandbox"
                  data-ph-capture-attribute-destination="sandbox"
                  data-ph-capture-attribute-source="download-hero"
                  data-ph-capture-attribute-cta="try-in-browser"
                >
                  Try in browser instead
                  <ArrowRight className="ml-2 h-4 w-4" aria-hidden="true" />
                </Link>
              </Button>
              <Button
                asChild
                size="lg"
                variant="outline"
                className="h-11 px-7 text-sm rounded-none bg-card border-border text-foreground"
              >
                <Link href="https://github.com/runmat-org/runmat" target="_blank" rel="noopener noreferrer">
                  <SiGithub className="mr-2 h-4 w-4" aria-hidden="true" />
                  View source on GitHub
                </Link>
              </Button>
            </div>

            <p className="mt-5 text-center text-xs text-muted-foreground">
              Signed desktop installers &middot; CLI remains free and MIT licensed
            </p>
          </div>
        </section>

        {/* Desktop downloads */}
        <section className="pb-16 md:pb-24">
          <div className="mx-auto max-w-3xl">
            <div className="mb-4 text-center">
              <h2 className="text-xl font-semibold text-foreground">Choose a desktop installer</h2>
              <p className="mt-2 text-sm text-muted-foreground">
                If automatic detection cannot determine your Mac architecture, pick the installer explicitly.
              </p>
            </div>
            <ul className="divide-y divide-border rounded-lg border border-border bg-card overflow-hidden">
              {desktopDownloads.map((download) => (
                <li key={download.href} className="flex items-center justify-between gap-3 px-4 py-3">
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-foreground">
                      {download.os}{" "}
                      <span className="text-muted-foreground">&middot; {download.arch}</span>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Latest {download.format} from the matching RunMat desktop release channel.
                    </div>
                  </div>
                  <Button asChild variant="outline" size="sm" className="rounded-none flex-shrink-0">
                    <Link href={download.href}>
                      Download
                      <ArrowRight className="ml-1.5 h-3.5 w-3.5" aria-hidden="true" />
                    </Link>
                  </Button>
                </li>
              ))}
            </ul>
          </div>
        </section>

        {/* CLI install */}
        <section className="pb-16 md:pb-24">
          <div className="mx-auto max-w-3xl">
            <div className="mb-4 text-center">
              <h2 className="text-xl font-semibold text-foreground">Install the CLI runtime</h2>
              <p className="mt-2 text-sm text-muted-foreground">
                Prefer a terminal workflow? Install the open-source runtime in seconds.
              </p>
            </div>
            <OSInstallCommand />
          </div>
        </section>

        {/* First command */}
        <section className="pb-16 md:pb-24">
          <div className="mx-auto max-w-3xl">
            <div className="mb-4 text-center">
              <h2 className="text-xl font-semibold text-foreground">Then run RunMat</h2>
              <p className="mt-2 text-sm text-muted-foreground">
                Drop into the interactive REPL. No project setup needed.
              </p>
            </div>
            <div className="rounded-md bg-[var(--editor-background)] px-5 py-4 font-mono text-sm sm:text-base text-white space-y-1">
              <div><span className="text-white/40 select-none">$ </span>runmat</div>
              <div className="text-white/40 text-xs sm:text-sm">RunMat REPL &mdash; type help for commands</div>
              <div><span className="text-[#7dd3fc] select-none">&gt;&gt; </span>A = magic(4)</div>
              <div className="text-white/80 text-xs sm:text-sm pl-4">
                A =<br />
                &nbsp;&nbsp;&nbsp;16&nbsp;&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;13<br />
                &nbsp;&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;11&nbsp;&nbsp;&nbsp;10&nbsp;&nbsp;&nbsp;&nbsp;8<br />
                &nbsp;&nbsp;&nbsp;&nbsp;9&nbsp;&nbsp;&nbsp;&nbsp;7&nbsp;&nbsp;&nbsp;&nbsp;6&nbsp;&nbsp;&nbsp;12<br />
                &nbsp;&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;14&nbsp;&nbsp;&nbsp;15&nbsp;&nbsp;&nbsp;&nbsp;1
              </div>
            </div>
            <p className="mt-3 text-center text-xs text-muted-foreground">
              Verify your install with{" "}
              <code className="px-1.5 py-0.5 rounded bg-muted text-foreground font-mono text-[0.8rem]">
                runmat --version
              </code>
            </p>
          </div>
        </section>

        {/* Coming soon strip */}
        <section className="pb-16 md:pb-24">
          <div className="mx-auto max-w-3xl rounded-md border border-border bg-card px-5 py-3 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
            <p className="text-sm text-foreground">
              <span className="font-medium text-foreground">Coming soon:</span>{" "}
              <span className="text-muted-foreground">
                native desktop app, VS Code extension, JetBrains plugin.
              </span>
            </p>
            <Link
              href="https://github.com/runmat-org/runmat"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-sm text-[hsl(var(--brand))] hover:opacity-80"
            >
              Watch the repo to get notified
              <ChevronRight className="h-4 w-4" aria-hidden="true" />
            </Link>
          </div>
        </section>

        {/* Alternative installation */}
        <section className="pb-16 md:pb-24">
          <div className="mx-auto max-w-5xl">
            <div className="mb-6 text-center">
              <h2 className="text-xl font-semibold text-foreground">Other installation methods</h2>
              <p className="mt-2 text-sm text-muted-foreground">
                Prefer a package manager? RunMat is on Homebrew and crates.io.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="rounded-lg border border-border bg-card p-6">
                <div className="flex items-center gap-2 mb-4">
                  <span className="inline-flex h-9 w-9 items-center justify-center rounded-md bg-foreground/10 text-foreground">
                    <Package className="h-4 w-4" aria-hidden="true" />
                  </span>
                  <h3 className="text-base font-semibold text-foreground">Homebrew (macOS &amp; Linux)</h3>
                </div>
                <div className="bg-[var(--editor-background)] text-white rounded-md p-3 font-mono text-sm">
                  <span className="text-white/40 select-none">$ </span>brew install runmat-org/tap/runmat
                </div>
                <details className="mt-3 group">
                  <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1">
                    Prefer to tap manually?
                    <ChevronRight className="h-3 w-3 transition-transform group-open:rotate-90" aria-hidden="true" />
                  </summary>
                  <div className="mt-2 space-y-2">
                    <div className="bg-[var(--editor-background)] text-white rounded-md p-3 font-mono text-sm">
                      <span className="text-white/40 select-none">$ </span>brew tap runmat-org/tap
                    </div>
                    <div className="bg-[var(--editor-background)] text-white rounded-md p-3 font-mono text-sm">
                      <span className="text-white/40 select-none">$ </span>brew install runmat
                    </div>
                  </div>
                </details>
              </div>

              <div className="rounded-lg border border-border bg-card p-6">
                <div className="flex items-center gap-2 mb-4">
                  <span className="inline-flex h-9 w-9 items-center justify-center rounded-md bg-foreground/10 text-foreground">
                    <Cpu className="h-4 w-4" aria-hidden="true" />
                  </span>
                  <h3 className="text-base font-semibold text-foreground">Cargo (Rust)</h3>
                </div>
                <div className="space-y-3">
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Install from crates.io</p>
                    <div className="bg-[var(--editor-background)] text-white rounded-md p-3 font-mono text-sm">
                      <span className="text-white/40 select-none">$ </span>cargo install runmat
                    </div>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Or build the latest from source</p>
                    <div className="bg-[var(--editor-background)] text-white rounded-md p-3 font-mono text-sm break-all">
                      <span className="text-white/40 select-none">$ </span>cargo install --git https://github.com/runmat-org/runmat
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Requires a recent stable Rust toolchain.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* CLI release archives */}
        <section className="pb-16 md:pb-24">
          <div className="mx-auto max-w-3xl">
            <details className="rounded-lg border border-border bg-card group">
              <summary className="cursor-pointer list-none px-5 py-4 flex items-center justify-between gap-3">
                <div className="flex items-center gap-3 min-w-0">
                  <span className="inline-flex h-9 w-9 items-center justify-center rounded-md bg-foreground/10 text-foreground flex-shrink-0">
                    <DownloadIcon className="h-4 w-4" aria-hidden="true" />
                  </span>
                  <div className="min-w-0">
                    <div className="text-base font-semibold text-foreground">CLI release archives</div>
                    <div className="text-xs text-muted-foreground">
                      For air-gapped, enterprise, or restricted CLI installs.
                    </div>
                  </div>
                </div>
                <ChevronRight className="h-4 w-4 text-muted-foreground transition-transform group-open:rotate-90 flex-shrink-0" aria-hidden="true" />
              </summary>
              <div className="px-5 pb-5 pt-1 space-y-3">
                <p className="text-sm text-muted-foreground">
                  Grab pre-built CLI/runtime archives directly from the public GitHub release page.
                </p>
                <Link
                  href="https://github.com/runmat-org/runmat/releases/latest"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
                >
                  See all release assets and checksums on GitHub
                  <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
                </Link>
              </div>
            </details>
          </div>
        </section>

        {/* Next steps */}
        <section className="pb-16 md:pb-24">
          <div className="mx-auto max-w-5xl">
            <div className="mb-6 text-center">
              <h2 className="text-xl font-semibold text-foreground">Next steps</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="rounded-lg border border-border bg-card p-6">
                <h3 className="text-base font-semibold text-foreground">Run your first script</h3>
                <p className="text-sm text-muted-foreground mt-1 mb-4">
                  Execute existing MATLAB files or open the REPL.
                </p>
                <div className="bg-[var(--editor-background)] text-white rounded-md p-3 font-mono text-sm mb-4">
                  <span className="text-white/40 select-none">$ </span>runmat my_script.m
                </div>
                <Button variant="outline" size="sm" asChild className="w-full rounded-none">
                  <Link href="/docs/getting-started">
                    Getting started guide
                  </Link>
                </Button>
              </div>
              <div className="rounded-lg border border-border bg-card p-6">
                <h3 className="text-base font-semibold text-foreground">Use as a Jupyter kernel</h3>
                <p className="text-sm text-muted-foreground mt-1 mb-4">
                  Run RunMat inside JupyterLab or Notebook.
                </p>
                <div className="bg-[var(--editor-background)] text-white rounded-md p-3 font-mono text-sm mb-4">
                  <span className="text-white/40 select-none">$ </span>runmat --install-kernel
                </div>
                <Button variant="outline" size="sm" asChild className="w-full rounded-none">
                  <Link href="/docs/getting-started#jupyter-notebook-integration">
                    Jupyter setup guide
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        </section>

        {/* Star on GitHub */}
        <section className="pb-16 md:pb-24 lg:pb-32">
          <div className="mx-auto max-w-2xl rounded-lg border border-border bg-card px-6 py-8 sm:px-8 sm:py-10 text-center space-y-4">
            <h2 className="text-lg sm:text-xl font-semibold text-foreground">
              Like RunMat? Star us on GitHub.
            </h2>
            <p className="text-foreground text-[0.938rem] max-w-md mx-auto">
              We&rsquo;re building RunMat in the open. A star helps more engineers find it.
            </p>
            <div className="flex justify-center pt-1">
              <Button
                asChild
                size="lg"
                className="h-11 px-7 text-sm font-semibold rounded-none bg-[hsl(var(--brand))] text-white border-0 shadow-none hover:bg-[hsl(var(--brand))]/90"
              >
                <Link
                  href="https://github.com/runmat-org/runmat"
                  target="_blank"
                  rel="noopener noreferrer"
                  data-ph-capture-attribute-destination="github"
                  data-ph-capture-attribute-source="download-footer"
                  data-ph-capture-attribute-cta="star-on-github"
                >
                  <SiGithub className="mr-2 h-4 w-4" aria-hidden="true" />
                  Star on GitHub
                </Link>
              </Button>
            </div>
          </div>
        </section>

      </div>
    </div>
  );
}
