import { Metadata } from "next";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Download, 
  Terminal, 
  Play, 
  FileText,
  CheckCircle,
  ArrowRight,
  Clock,
  Users
} from "lucide-react";
import Link from "next/link";
import { OSInstallCommand } from "@/components/OSInstallCommand";

export const metadata: Metadata = {
  title: "Getting Started with RunMat",
  description: "Learn how to install and use RunMat, the modern MATLAB/Octave runtime. Complete guide for researchers, engineers, and students.",
};

export default function GettingStartedPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 md:px-6 py-16 md:py-4">
        
        {/* Header */}
        <div className="mb-12">
          <Badge variant="secondary" className="mb-4">Documentation</Badge>
          <h1 className="text-4xl lg:text-5xl font-bold mb-6">
            Getting Started with RunMat
          </h1>
          <p className="text-xl text-muted-foreground leading-relaxed">
            Get up and running with RunMat in minutes. This guide will walk you through installation, 
            basic usage, and your first interactive session.
          </p>
        </div>

        {/* Installation Section */}
        <section id="installation" className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">
            Installation
          </h2>
          
          <div className="space-y-6">
            {/* Quick Install */}
            <OSInstallCommand className="mb-6" />
            <Button asChild>
                    <Link href="/download">
                      <Download className="mr-2 h-4 w-4" />
                More Installation Options
                    </Link>
            </Button>
          </div>
        </section>

        {/* First Steps */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">
            Your First RunMat Session
          </h2>
          
          <div className="space-y-6">
            {/* Start REPL */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <span className="bg-blue-100 dark:bg-blue-900/30 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold text-blue-600 mr-3">1</span>
                  Start the Interactive REPL
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Open your terminal and start RunMat:
                </p>
                <div className="bg-gray-900 rounded-md p-4 font-mono text-sm text-white overflow-x-auto mb-4">
                  <div className="text-gray-400">$ runmat</div>
                  <div className="text-green-400 mt-1">RunMat v0.0.1 by Dystr (https://dystr.com)</div>
                  <div className="text-green-400">High-performance MATLAB/Octave runtime with JIT compilation and GC</div>
                  <div className="text-green-400 mt-1">JIT compiler: enabled (Cranelift optimization level: Speed)</div>
                  <div className="text-green-400">Garbage collector: &quot;default&quot;</div>
                  <div className="text-green-400">No snapshot loaded - standard library will be compiled on demand</div>
                  <div className="text-green-400">Type &apos;help&apos; for help, &apos;exit&apos; to quit, &apos;.info&apos; for system information</div>
                  <div className="mt-2 text-white">runmat&gt;</div>
                </div>
              </CardContent>
            </Card>

            {/* Basic Calculations */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <span className="bg-orange-100 dark:bg-orange-900/30 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold text-orange-600 mr-3">2</span>
                  Try Basic Calculations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Start with simple arithmetic and variables:
                </p>
                <div className="bg-gray-900 rounded-md p-4 font-mono text-sm text-white overflow-x-auto mb-4">
                  <div className="text-blue-400">runmat&gt;</div> <span>x = 5</span>
                  <div className="text-gray-400">ans = 5</div>
                  <div className="text-blue-400 mt-2">runmat&gt;</div> <span>y = 3.14</span>
                  <div className="text-gray-400">ans = 3.14</div>
                  <div className="text-blue-400 mt-2">runmat&gt;</div> <span>result = x * y + 2</span>
                  <div className="text-gray-400">ans = 17.7</div>
                </div>
              </CardContent>
            </Card>

            {/* Matrices */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <span className="bg-green-100 dark:bg-green-900/30 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold text-green-600 mr-3">3</span>
                  Work with Matrices
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Create and manipulate matrices using familiar MATLAB syntax:
                </p>
                <div className="bg-gray-900 rounded-md p-4 font-mono text-sm text-white overflow-x-auto mb-4">
                  <div className="text-blue-400">runmat&gt;</div> <span>A = [1, 2, 3; 4, 5, 6]</span>
                  <div className="text-gray-400">ans = [1 2 3; 4 5 6]</div>
                  <div className="text-blue-400 mt-2">runmat&gt;</div> <span>B = A * 2</span>
                  <div className="text-gray-400">ans = [2 4 6; 8 10 12]</div>
                  <div className="text-blue-400 mt-2">runmat&gt;</div> <span>C = A + B</span>
                  <div className="text-gray-400">ans = [3 6 9; 12 15 18]</div>
                </div>
              </CardContent>
            </Card>

            {/* Plotting */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <span className="bg-purple-100 dark:bg-purple-900/30 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold text-purple-600 mr-3">4</span>
                  Create Your First Plot
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Generate beautiful plots with GPU acceleration:
                </p>
                <div className="bg-gray-900 rounded-md p-4 font-mono text-sm text-white overflow-x-auto mb-4">
                  <div className="text-blue-400">runmat&gt;</div> <span>x = [0, 1, 2, 3, 4, 5]</span>
                  <div className="text-gray-400">ans = [0 1 2 3 4 5]</div>
                  <div className="text-blue-400 mt-1">runmat&gt;</div> <span>y = [0, 1, 4, 9, 16, 25]</span>
                  <div className="text-gray-400">ans = [0 1 4 9 16 25]</div>
                  <div className="text-blue-400 mt-1">runmat&gt;</div> <span>plot(x, y)</span>
                  <div className="text-green-400 mt-1">[Interactive plot window opens]</div>
                </div>
                <div className="flex items-center space-x-2 text-sm text-green-600 dark:text-green-400">
                  <CheckCircle className="h-4 w-4" />
                  <span>Interactive window with zoom, pan, and rotate controls</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Running Scripts */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">
            Running MATLAB Scripts
          </h2>
          
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Play className="h-5 w-5 mr-2 text-green-600" />
                Execute .m Files
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground mb-4">
                Run existing MATLAB/Octave scripts directly:
              </p>
              <div className="bg-gray-900 rounded-md p-4 font-mono text-sm text-white overflow-x-auto mb-4">
                <div className="text-gray-400"># Run a script file</div>
                <div>runmat script.m</div>
                <div className="mt-2 text-gray-400"># Run with specific options</div>
                <div>runmat run --jit-threshold 100 simulation.m</div>
              </div>
              <p className="text-sm text-muted-foreground">
                Most MATLAB and GNU Octave scripts will run without modification. Check our 
                <Link href="/docs/language-coverage" className="text-blue-600 dark:text-blue-400 underline"> compatibility guide</Link> for details.
              </p>
            </CardContent>
          </Card>
        </section>

        {/* Jupyter Integration */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">
            Jupyter Notebook Integration
          </h2>
          
          <div className="space-y-6">
            {/* Install Kernel */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <span className="bg-blue-100 dark:bg-blue-900/30 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold text-blue-600 mr-3">1</span>
                  Install RunMat as a Jupyter Kernel
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Make RunMat available as a kernel in Jupyter notebooks:
                </p>
                <div className="bg-gray-900 rounded-md p-4 font-mono text-sm text-white overflow-x-auto mb-4">
                  <div>runmat --install-kernel</div>
                  <div className="text-green-400 mt-1">RunMat Jupyter kernel installed successfully!</div>
                  <div className="text-green-400">Kernel directory: ~/.local/share/jupyter/kernels/runmat</div>
                </div>
                <div className="flex items-center space-x-2 text-sm text-green-600 dark:text-green-400">
                  <CheckCircle className="h-4 w-4" />
                  <span>One-time setup that works with existing Jupyter installations</span>
                </div>
              </CardContent>
            </Card>

            {/* Use in Jupyter */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <span className="bg-orange-100 dark:bg-orange-900/30 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold text-orange-600 mr-3">2</span>
                  Start Jupyter and Select RunMat
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Launch Jupyter and create notebooks with the RunMat kernel:
                </p>
                <div className="bg-gray-900 rounded-md p-4 font-mono text-sm text-white overflow-x-auto mb-4">
                  <div className="text-gray-400"># Start Jupyter Notebook</div>
                  <div>jupyter notebook</div>
                  <div className="mt-2 text-gray-400"># Or Jupyter Lab</div>
                  <div>jupyter lab</div>
                  <div className="mt-2 text-gray-400"># Then select &quot;RunMat&quot; when creating a new notebook</div>
                </div>
                <div className="flex items-center space-x-2 text-sm text-blue-600 dark:text-blue-400">
                  <CheckCircle className="h-4 w-4" />
                  <span>Full MATLAB syntax support with 150x faster execution than GNU Octave</span>
                </div>
              </CardContent>
            </Card>

            {/* Verify Installation */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <span className="bg-green-100 dark:bg-green-900/30 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold text-green-600 mr-3">3</span>
                  Verify Installation
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Check that the RunMat kernel is properly installed:
                </p>
                <div className="bg-gray-900 rounded-md p-4 font-mono text-sm text-white overflow-x-auto mb-4">
                  <div>jupyter kernelspec list</div>
                  <div className="text-gray-400 mt-1">Available kernels:</div>
                  <div className="text-gray-400">  python3    /usr/local/share/jupyter/kernels/python3</div>
                  <div className="text-green-400">  runmat    ~/.local/share/jupyter/kernels/runmat</div>
                </div>
                <p className="text-sm text-muted-foreground">
                  If you don&apos;t see RunMat listed, ensure Jupyter is installed and try running the install command again.
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Next Steps */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">
            Next Steps
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
                  <Link href="/docs/how-it-works">
                    How RunMat Works
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card className="opacity-60 transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Terminal className="h-5 w-5 mr-2 text-gray-400" />
                  <span className="text-gray-500 dark:text-gray-400">Explore Examples</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-500 dark:text-gray-400 mb-4">
                  See RunMat in action with real-world examples.
                </p>
                <Button variant="outline" className="w-full opacity-50 cursor-not-allowed" disabled>
                  Coming Soon
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