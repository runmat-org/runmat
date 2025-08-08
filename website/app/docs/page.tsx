import { Metadata } from "next";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { BookOpen, Zap, Code, Cpu, Settings, Terminal } from "lucide-react";

export const metadata: Metadata = {
  title: "RustMat Documentation - Complete Guides and API Reference",
  description: "Complete guides and API reference to get you started with RustMat. Learn installation, usage, MATLAB migration, and advanced features.",
  openGraph: {
    title: "RustMat Documentation - Complete Guides and API Reference",
    description: "Complete guides and API reference to get you started with RustMat.",
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
            RustMat Documentation
          </h1>
          <p className="mt-6 text-xl text-muted-foreground">
            Complete guides and API reference to get you started with RustMat
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
                  Quick start guide to install and run your first RustMat program
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
                  Deep dive into RustMat&apos;s architecture, JIT compilation, and performance
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-sm text-muted-foreground">
                  Learn More →
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

          {/* MATLAB Migration */}
          <Card className="group relative overflow-hidden transition-colors hover:bg-muted/50 opacity-60">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Zap className="h-6 w-6 text-primary" />
                <CardTitle>MATLAB Migration</CardTitle>
              </div>
              <CardDescription>
                Step-by-step guide to migrate your MATLAB projects to RustMat
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-sm text-muted-foreground">
                Coming Soon
              </div>
            </CardContent>
          </Card>

          {/* Configuration */}
          <Card className="group relative overflow-hidden transition-colors hover:bg-muted/50 opacity-60">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Settings className="h-6 w-6 text-primary" />
                <CardTitle>Configuration</CardTitle>
              </div>
              <CardDescription>
                Configure RustMat for optimal performance in your environment
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-sm text-muted-foreground">
                Coming Soon
              </div>
            </CardContent>
          </Card>

          {/* CLI Reference */}
          <Card className="group relative overflow-hidden transition-colors hover:bg-muted/50 opacity-60">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Terminal className="h-6 w-6 text-primary" />
                <CardTitle>CLI Reference</CardTitle>
              </div>
              <CardDescription>
                Complete command-line interface reference and examples
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
              <Link href="/download">Download RustMat</Link>
            </Button>
            <Button variant="outline" size="sm" asChild>
              <Link href="/blog/introducing-rustmat">Read Introduction</Link>
            </Button>
            <Button variant="outline" size="sm" asChild>
              <Link href="https://github.com/rustmat/rustmat" target="_blank">
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