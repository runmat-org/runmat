import { Metadata } from "next";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  Download, 
  Terminal, 
  FileText,
  ArrowRight,
  Zap,
  ExternalLink,
} from "lucide-react";
import Link from "next/link";
import { GettingStartedTabs } from "@/components/GettingStartedTabs";

export const metadata: Metadata = {
  title: "Getting Started | Docs",
  description: "Learn how to install and use RunMat, the modern MATLAB/Octave runtime. Try in the browser, install the CLI, or use Jupyter. Complete guide for researchers, engineers, and students.",
};

export default function GettingStartedPage() {
  return (
    <div className="min-h-screen bg-background">
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
            <Link href="/sandbox" target="_blank" rel="noopener noreferrer" className="inline-flex items-center">
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

        {/* Next Steps */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">
            Next Steps
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
                  Install the CLI to run scripts from your terminal and use local files. Need native GPU performance? Download RunMat for full-speed execution.
                </p>
                <Button variant="outline" className="w-full" asChild>
                  <Link href="/download" className="flex items-center justify-center">
                    Install RunMat / Download
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <FileText className="h-5 w-5 mr-2 text-blue-600" />
                  Learn the Fundamentals
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Dive deeper into RunMat&apos;s features and capabilities.
                </p>
                <Button variant="outline" className="w-full" asChild>
                  <Link href="/docs/how-it-works" className="flex items-center justify-center">
                    How RunMat Works
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Terminal className="h-5 w-5 mr-2 text-green-600 dark:text-green-400" />
                  Explore Examples
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
                  Understand the Design
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Learn why RunMat keeps a slim core and package-first model.
                </p>
                <Button variant="outline" className="w-full" asChild>
                  <Link href="/docs/design-philosophy" className="flex items-center justify-center">
                    Design Philosophy
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
                Need Help?
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