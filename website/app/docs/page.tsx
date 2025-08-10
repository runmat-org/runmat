import { Metadata } from "next";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { BookOpen, Zap, Code, Cpu, Settings, Terminal, GitBranch } from "lucide-react";

export const metadata: Metadata = {
  title: "RunMat Documentation - Complete Guides and API Reference",
  description: "Complete guides and API reference to get you started with RunMat. Learn installation, usage, MATLAB migration, and advanced features.",
  openGraph: {
    title: "RunMat Documentation - Complete Guides and API Reference",
    description: "Complete guides and API reference to get you started with RunMat.",
    type: "website",
  },
};

export default function DocsPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-16 md:px-6 md:py-24">
        <div className="mx-auto max-w-[58rem] text-center">
          <Badge variant="secondary" className="mb-4">
            Documentation
          </Badge>
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl">
            RunMat Documentation
          </h1>
          <p className="mt-6 text-xl text-muted-foreground">
            Complete guides and API reference to get you started with RunMat
          </p>
        </div>

        <div className="mt-16 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {/* Getting Started */}
          <Link href="/docs/getting-started" className="block">
            <Card className="group relative overflow-hidden transition-colors hover:bg-muted/50 cursor-pointer h-full">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <BookOpen className="h-6 w-6 text-primary" />
                  <CardTitle>Getting Started</CardTitle>
                </div>
                <CardDescription>
                  Quick start guide to install and run your first RunMat program
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-sm text-muted-foreground">
                  Read Guide →
                </div>
              </CardContent>
            </Card>
          </Link>

          {/* How It Works */}
          <Link href="/docs/how-it-works" className="block">
            <Card className="group relative overflow-hidden transition-colors hover:bg-muted/50 cursor-pointer h-full">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Cpu className="h-6 w-6 text-primary" />
                  <CardTitle>How It Works</CardTitle>
                </div>
                <CardDescription>
                  Deep dive into RunMat&apos;s architecture, JIT compilation, and performance
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-sm text-muted-foreground">
                  Learn More →
                </div>
              </CardContent>
            </Card>
          </Link>

          {/* CLI Reference (from repo docs) */}
          <Link href="/docs/cli" className="block">
            <Card className="group relative overflow-hidden transition-colors hover:bg-muted/50 cursor-pointer h-full">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Terminal className="h-6 w-6 text-primary" />
                  <CardTitle>CLI Reference</CardTitle>
                </div>
                <CardDescription>
                  Full CLI reference, commands, env vars, and examples
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-sm text-muted-foreground">
                  Browse Reference →
                </div>
              </CardContent>
            </Card>
          </Link>

          {/* Configuration (from repo docs) */}
          <Link href="/docs/configuration" className="block">
            <Card className="group relative overflow-hidden transition-colors hover:bg-muted/50 cursor-pointer h-full">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Settings className="h-6 w-6 text-primary" />
                  <CardTitle>Configuration</CardTitle>
                </div>
                <CardDescription>
                  Configure RunMat via YAML/JSON/TOML with env & CLI precedence
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-sm text-muted-foreground">
                  Open Guide →
                </div>
              </CardContent>
            </Card>
          </Link>

          {/* Built-in Functions */}
          <Link href="/docs/builtin-functions" className="block">
            <Card className="group relative overflow-hidden transition-colors hover:bg-muted/50 cursor-pointer h-full">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Code className="h-6 w-6 text-primary" />
                  <CardTitle>Built-in Functions</CardTitle>
                </div>
                <CardDescription>
                  Complete reference of all mathematical and plotting functions
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-sm text-muted-foreground">
                  Browse Functions →
                </div>
              </CardContent>
            </Card>
          </Link>

          {/* Architecture (from repo docs) */}
          <Link href="/docs/architecture" className="block">
            <Card className="group relative overflow-hidden transition-colors hover:bg-muted/50 cursor-pointer h-full">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <GitBranch className="h-6 w-6 text-primary" />
                  <CardTitle>Architecture</CardTitle>
                </div>
                <CardDescription>
                  Learn how RunMat is structured (interpreter, JIT, GC, runtime, plotting)
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-sm text-muted-foreground">
                  Open Document →
                </div>
              </CardContent>
            </Card>
          </Link>

          {/* Language Coverage (from repo docs) */}
          <Link href="/docs/language-coverage" className="block">
            <Card className="group relative overflow-hidden transition-colors hover:bg-muted/50 cursor-pointer h-full">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <BookOpen className="h-6 w-6 text-primary" />
                  <CardTitle>Language Coverage</CardTitle>
                </div>
                <CardDescription>
                  Track MATLAB syntax and feature support in RunMat
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-sm text-muted-foreground">
                  View Coverage →
                </div>
              </CardContent>
            </Card>
          </Link>

          {/* MATLAB Migration */}
          <Card className="group relative overflow-hidden transition-colors hover:bg-muted/50 opacity-60">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Zap className="h-6 w-6 text-primary" />
                <CardTitle>MATLAB Migration</CardTitle>
              </div>
              <CardDescription>
                Step-by-step guide to migrate your MATLAB projects to RunMat
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-sm text-muted-foreground">
                Coming Soon
              </div>
            </CardContent>
          </Card>

        </div>

        {/* Quick Links */}
        <div className="mt-16 rounded-lg border bg-muted/50 p-6">
          <h2 className="mb-4 text-xl font-semibold">Quick Links</h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Button variant="outline" size="sm" asChild>
              <Link href="/download">Download RunMat</Link>
            </Button>
            <Button variant="outline" size="sm" asChild>
              <Link href="/blog/introducing-runmat">Read Introduction</Link>
            </Button>
            <Button variant="outline" size="sm" asChild>
              <Link href="https://github.com/runmat-org/runmat" target="_blank">
                View on GitHub
              </Link>
            </Button>
            <Button variant="outline" size="sm" asChild>
              <Link href="/license">License (MIT)</Link>
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}