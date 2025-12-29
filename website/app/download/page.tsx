import { Metadata } from "next";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  Package,
  Code,
  Zap,
  Cpu,
  ExternalLink
} from "lucide-react";
import { SiGithub } from "react-icons/si";
import { OSInstallCommand } from "@/components/OSInstallCommand";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Download RunMat",
  description: "Download RunMat for your platform. Fast, free, and open source MATLAB-compatible runtime with V8-inspired performance.",
};

export default function DownloadPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        
        {/* Header */}
        <div className="text-center mb-12">
          <Badge className="mb-4">Free Download</Badge>
          <h1 className="text-4xl lg:text-5xl font-bold mb-6 text-foreground">
            Download RunMat
          </h1>
          <p className="text-xl text-muted-foreground leading-relaxed max-w-3xl mx-auto">
            Get started with RunMat in seconds.
          </p>
        </div>

        {/* Quick Install */}
        <section className="mb-12">
          <Card className="bg-gradient-to-r from-blue-50 to-orange-50 dark:from-blue-900/20 dark:to-orange-900/20 border-blue-200 dark:border-blue-800">
            <CardContent className="p-4 text-center">
              <OSInstallCommand className="max-w-4xl mx-auto shadow-none border-none" />
            </CardContent>
          </Card>
        </section>

        {/* Alternative Installation Methods */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-8 text-center text-foreground">
            Alternative Installation Methods
          </h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            
            {/* Package Managers */}
            <Card className="relative">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Package className="h-5 w-5 mr-2 text-primary" />
                  Homebrew (macOS/Linux)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold text-foreground mb-2">One-liner install</h4>
                    <div className="bg-gray-200 dark:bg-gray-700 rounded-md p-3 font-mono text-sm text-foreground dark:text-foreground">
                      brew install runmat-org/tap/runmat
                    </div>
                  </div>
                  <div className="space-y-2">
                    <h4 className="font-semibold text-foreground">Tap then install</h4>
                    <div className="bg-gray-200 dark:bg-gray-700 rounded-md p-3 font-mono text-sm text-foreground dark:text-foreground">
                      brew tap runmat-org/tap
                    </div>
                    <div className="bg-gray-200 dark:bg-gray-700 rounded-md p-3 font-mono text-sm text-foreground dark:text-foreground">
                      brew install runmat
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Cargo Installation */}
            <Card className="relative border-primary/40">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Cpu className="h-5 w-5 mr-2 text-primary" />
                  Cargo (Rust)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold text-foreground mb-2">Install from crates.io</h4>
                    <div className="bg-gray-200 dark:bg-gray-700 rounded-md p-3 font-mono text-sm text-foreground dark:text-foreground">
                      cargo install runmat
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold text-foreground mb-2">From Source (Latest)</h4>
                    <div className="bg-gray-200 dark:bg-gray-700 rounded-md p-3 font-mono text-sm text-foreground dark:text-foreground">
                      cargo install --git https://github.com/runmat-org/runmat
                    </div>
                  </div>
                  
                  <p className="text-sm text-muted-foreground">
                    Requires Rust 1.70+ with LLVM support
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Development Setup */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-8 text-center text-foreground">
            Development Environment
          </h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* VS Code Extension */}
            <Card className="opacity-60 relative">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Code className="h-5 w-5 mr-2 text-gray-400" />
                  VS Code Extension
                  <span className="ml-auto bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300 px-2 py-1 rounded-full text-xs font-medium">
                    Coming Soon
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-500 dark:text-gray-400 mb-4">
                  Get syntax highlighting, IntelliSense, and integrated debugging for RunMat.
                </p>
                <Button className="w-full opacity-50 cursor-not-allowed" disabled>
                  <ExternalLink className="mr-2 h-4 w-4" />
                  Install Extension
                </Button>
              </CardContent>
            </Card>

            {/* IntelliJ IDEA Plugin */}
            <Card className="opacity-60 relative">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Code className="h-5 w-5 mr-2 text-gray-400" />
                  IntelliJ IDEA Plugin
                  <span className="ml-auto bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300 px-2 py-1 rounded-full text-xs font-medium">
                    Coming Soon
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-500 dark:text-gray-400 mb-4">
                  Get syntax highlighting, IntelliSense, and integrated debugging for RunMat.
                </p>
                <Button className="w-full opacity-50 cursor-not-allowed" disabled>
                  <ExternalLink className="mr-2 h-4 w-4" />
                  Install Extension
                </Button>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Next Steps */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-8 text-center text-foreground">
            Next Steps
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 border-green-200 dark:border-green-800">
              <CardContent className="p-6">
                <div className="flex items-center mb-4">
                  <Code className="h-6 w-6 mr-3 text-green-600" />
                  <h3 className="text-xl font-bold text-foreground">
                    Start Coding
                  </h3>
                </div>
                <p className="text-muted-foreground mb-4">
                  Jump into the interactive REPL or run your existing MATLAB scripts by typing <code>runmat</code> in your terminal.
                </p>
                <div className="bg-muted rounded-md p-3 font-mono text-sm text-muted-foreground mb-4">
                  runmat
                </div>
                <Button variant="outline" size="sm" asChild className="w-full">
                  <Link href="/docs/getting-started">
                    Getting Started Guide
                  </Link>
                </Button>
              </CardContent>
            </Card>
            <Card className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 border-purple-200 dark:border-purple-800">
              <CardContent className="p-6">
                <div className="flex items-center mb-4">
                  <Zap className="h-6 w-6 mr-3 text-purple-600" />
                  <h3 className="text-xl font-bold text-foreground">
                    Set up Jupyter Kernel
                  </h3>
                </div>
                <p className="text-muted-foreground mb-4">
                  Use RunMat as a Jupyter kernel for interactive notebooks with 150x speedup over GNU Octave.
                </p>
                <div className="bg-muted rounded-md p-3 font-mono text-sm text-muted-foreground mb-4">
                  runmat --install-kernel
                </div>
                <Button variant="outline" size="sm" asChild className="w-full">
                  <Link href="/docs/getting-started#jupyter-notebook-integration">
                    Jupyter Setup Guide
                  </Link>
                </Button>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Source Code */}
        <section className="mb-12">
          <Card className="bg-muted border-border">
            <CardHeader className="text-center">
              <CardTitle className="flex items-center justify-center text-foreground">
                <SiGithub className="h-6 w-6 mr-2" />
                Open Source
              </CardTitle>
            </CardHeader>
            <CardContent className="text-center">
              <p className="text-muted-foreground mb-6">
                RunMat is completely open source. Explore the code, contribute, or build it yourself.
              </p>
              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <Button asChild>
                            <Link href="https://github.com/runmat-org/runmat" target="_blank" rel="noopener noreferrer">
            <SiGithub className="mr-2 h-4 w-4" />
            View on GitHub
                  </Link>
                </Button>
                <Button variant="outline" asChild>
                  <Link href="https://crates.io/crates/runmat" target="_blank" rel="noopener noreferrer">
                    <Package className="mr-2 h-4 w-4" />
                    View on crates.io
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