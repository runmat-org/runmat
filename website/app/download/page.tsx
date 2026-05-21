import { Metadata } from "next";
import { Button } from "@/components/ui/button";
import {
  Package,
  Cpu,
  ArrowRight,
  Download as DownloadIcon,
  ChevronRight,
  Download,
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
              Download RunMat for full performance, or use RunMat in your browser for zero setup.
            </p>
          </div>

          <div className="mx-auto mt-10 max-w-3xl flex flex-col justify-center items-center">
            <Button
                  asChild
                  size="lg"
                  className="h-11 px-7 text-sm font-semibold rounded-none bg-[hsl(var(--brand))] text-white border-0 shadow-none hover:bg-[hsl(var(--brand))]/90"
                >
                  <Link href="/download/latest">
                Download RunMat Desktop
                <Download className="ml h-4 w-4" aria-hidden="true" />
                  </Link>
            </Button>

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
                  Try in browser
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
                  View GitHub
                </Link>
              </Button>
            </div>
          </div>
        </section>

        {/* Desktop downloads */}
        <section className="pb-16 md:pb-24">
          <div className="mx-auto max-w-3xl">
            <ul className="divide-y divide-border rounded-lg border border-border bg-card overflow-hidden">
              {desktopDownloads.map((download) => (
                <li key={download.href} className="flex items-center justify-between gap-3 px-4 py-3">
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-foreground">
                      {download.os}{" "}
                      <span className="text-muted-foreground">&middot; {download.arch} ({download.format})</span>
                    </div>
                  </div>
                  <Button asChild variant="outline" size="sm" className="rounded-none flex-shrink-0">
                    <Link href={download.href}>
                      Download
                      <Download className="ml-1.5 h-3.5 w-3.5" aria-hidden="true" />
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
              <h2 className="text-xl font-semibold text-foreground">Install CLI only</h2>
            </div>
            <OSInstallCommand />
          </div>
        </section>

        {/* Alternative installation */}
        <section className="pb-16 md:pb-24">
          <div className="mx-auto max-w-5xl">
            <div className="mb-6 text-center">
              <h2 className="text-xl font-semibold text-foreground">Alternative CLI installation methods</h2>
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
                      <span className="text-white/40 select-none">$ </span>cargo install --git
                      <br />
                      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                      https://github.com/runmat-org/runmat
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
