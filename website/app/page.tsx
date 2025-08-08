import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

import { OSInstallCommand } from "@/components/OSInstallCommand";
import Link from "next/link";

export default function HomePage() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Hero Section */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="flex flex-col items-center space-y-6 text-center">
          <Badge variant="secondary" className="rounded-lg px-3 py-1 text-sm">
            🚀 Open Source • MIT Licensed • Free Forever
          </Badge>
          <h1 className="font-heading text-3xl sm:text-5xl md:text-6xl lg:text-7xl text-center">
            <span className="gradient-brand">The Modern MATLAB Runtime</span>
          </h1>
          <p className="max-w-[42rem] leading-relaxed text-muted-foreground sm:text-xl sm:leading-8">
              A free, high-performance runtime for MATLAB code that runs your existing scripts faster and more reliably. 
            No license fees, no vendor lock-in, just blazing-fast numerical computing with beautiful visualizations.
          </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <Button size="lg" asChild className="h-12 px-8 text-lg font-semibold bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-xl border-0 transition-all duration-200 hover:scale-105 hover:shadow-2xl">
                <Link href="/download">
                  Download RunMat
                </Link>
              </Button>
              <Button variant="outline" size="lg" asChild className="h-12 px-8">
                <Link href="/docs/getting-started">Get Started</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="w-full py-16 md:py-24 lg:py-32 bg-muted/50">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
            <h2 className="font-heading text-3xl leading-[1.1] sm:text-4xl md:text-5xl lg:text-6xl">
              Why Use RunMat?
            </h2>
            <p className="max-w-[42rem] leading-relaxed text-muted-foreground sm:text-xl sm:leading-8">
              All the power of MATLAB without the price tag or limitations
            </p>
        </div>
        <div className="mx-auto grid justify-center gap-4 sm:grid-cols-2 md:max-w-[64rem] md:grid-cols-3">
          <Card>
            <CardHeader>
              <div className="mb-2 text-3xl">⚡</div>
              <CardTitle>Dramatically Faster</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>
                Advanced optimization technology that makes your MATLAB code run significantly 
                faster than alternatives like GNU Octave. Same syntax, better performance.
              </CardDescription>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <div className="mb-2 text-3xl">🛡️</div>
              <CardTitle>Rock Solid Reliability</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>
                Never lose your work to crashes or memory errors. Built with modern safety 
                technology to ensure your simulations and analyses complete successfully.
              </CardDescription>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <div className="mb-2 text-3xl">🎨</div>
              <CardTitle>Beautiful Plots</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>
                GPU-accelerated plotting with modern aesthetics. Interactive 2D/3D 
                visualizations that export to any format.
              </CardDescription>
            </CardContent>
          </Card>
        </div>
        </div>
      </section>

      {/* Code Example Section */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
          <h2 className="font-heading text-3xl leading-[1.1] sm:text-3xl md:text-6xl">
            Same MATLAB Syntax, Better Performance
          </h2>
          <p className="max-w-[42rem] leading-relaxed text-muted-foreground sm:text-xl sm:leading-8">
            Copy-paste your existing MATLAB code and watch it run faster and for free
          </p>
          </div>

          <div className="mx-auto max-w-7xl">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
              {/* Terminal/Shell on the left */}
              <div className="order-2 lg:order-1">
                <Card className="bg-gray-900 dark:bg-gray-900 bg-gray-100 border-gray-700 dark:border-gray-700 border-gray-300 shadow-2xl py-0">
                  {/* Terminal Header */}
                  <div className="flex items-center justify-between px-4 py-3 bg-gray-800 dark:bg-gray-800 bg-gray-200 rounded-t-lg border-b border-gray-700 dark:border-gray-700 border-b-gray-300">
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    </div>
                    <div className="w-12"></div>
                  </div>

                  {/* Terminal Content */}
                  <CardContent className="px-6 pt-4 pb-6 font-mono text-sm">
                    <div className="space-y-2 text-white dark:text-white text-gray-900">
                      {/* Initial command */}
                      <div className="flex">
                        <span className="text-gray-500 dark:text-gray-500 text-gray-600 dark:text-gray-500 dark:text-gray-500 text-gray-600 text-gray-600">$</span>
                        <span className="ml-2">runmat</span>
                      </div>

                      {/* RunMat header */}
                      <div className="text-cyan-400 dark:text-cyan-400 text-cyan-600">RunMat v0.1.0 by Dystr (https://dystr.com)</div>
                      <div className="text-cyan-400 dark:text-cyan-400 text-cyan-600">High-performance MATLAB/Octave runtime with JIT compilation</div>
                      <div className="text-cyan-400 dark:text-cyan-400 text-cyan-600">Type &apos;help&apos; for help, &apos;exit&apos; to quit</div>
                      <div className="h-4"></div>

                      {/* Interactive session */}
                      <div className="space-y-5">
                        {/* Generate data section */}
                        <div className="space-y-1">
                          <div className="text-cyan-300 dark:text-cyan-300 text-cyan-700 mb-2">% Generate data</div>
                          <div>
                            <div className="flex">
                              <span className="text-blue-400 dark:text-blue-400 text-blue-600">runmat&gt;</span>
                              <span className="ml-2">x = linspace(0, 4 * pi, 1000);</span>
                            </div>
                            <div className="text-gray-500 dark:text-gray-500 text-gray-600">ans = Num(1000x1 vector)</div>
                          </div>
                          <div>
                            <div className="flex">
                              <span className="text-blue-400 dark:text-blue-400 text-blue-600">runmat&gt;</span>
                              <span className="ml-2">y = sin(x) .* exp(-x/10);</span>
                            </div>
                            <div className="text-gray-500 dark:text-gray-500 text-gray-600">ans = Num(1000x1 vector)</div>
                          </div>
                        </div>

                        {/* Create plot section */}
                        <div className="space-y-1">
                          <div className="text-cyan-300 dark:text-cyan-300 text-cyan-700 mb-2">% Create beautiful plot</div>
                          <div>
                            <div className="flex">
                              <span className="text-blue-400 dark:text-blue-400 text-blue-600">runmat&gt;</span>
                              <span className="ml-2">plot(x, y);</span>
                            </div>
                            <div className="text-gray-500 dark:text-gray-500 text-gray-600">[Interactive plot window opened]</div>
                          </div>
                        </div>

                        {/* Matrix operations section */}
                        <div className="space-y-1">
                          <div className="text-cyan-300 dark:text-cyan-300 text-cyan-700 mb-2">% Matrix operations</div>
                          <div>
                            <div className="flex">
                              <span className="text-blue-400 dark:text-blue-400 text-blue-600">runmat&gt;</span>
                              <span className="ml-2">A = randn(1000, 1000);</span>
                            </div>
                            <div className="text-gray-500 dark:text-gray-500 text-gray-600">ans = Matrix(1000x1000 double)</div>
                          </div>
                          <div>
                            <div className="flex">
                              <span className="text-blue-400 dark:text-blue-400 text-blue-600">runmat&gt;</span>
                              <span className="ml-2">B = A * A&apos;;</span>
                            </div>
                            <div className="text-gray-500 dark:text-gray-500 text-gray-600">ans = Matrix(1000x1000 double)</div>
                          </div>
                          <div>
                            <div className="flex">
                              <span className="text-blue-400 dark:text-blue-400 text-blue-600">runmat&gt;</span>
                              <span className="ml-2">eigenvals = eig(B);</span>
                            </div>
                            <div className="text-gray-500 dark:text-gray-500 text-gray-600">ans = Num(1000x1 complex vector)</div>
                          </div>
                        </div>

                        {/* Cursor */}
                        <div className="flex mt-3">
                          <span className="text-blue-400 dark:text-blue-400 text-blue-600">runmat&gt;</span>
                          <span className="ml-2 animate-pulse">▊</span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* RunMat Plot Window */}
              <div className="order-1 lg:order-2">
                <div className="bg-gray-700 rounded-lg shadow-2xl overflow-hidden">
                  {/* macOS Window Header */}
                  <div className="bg-gray-600 px-4 py-3 flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    </div>
                    <div className="text-gray-200 text-sm font-medium">RunMat - Interactive Visualization</div>
                    <div className="w-16"></div>
                  </div>

                  {/* Window Content Area */}
                  <div className="overflow-hidden">
                    <img
                      src="/plot-example.jpg"
                      alt="RunMat interactive plot showing a damped sine wave with real-time visualization controls"
                      className="w-full h-auto"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Technical Details Section */}
      <section className="w-full py-16 md:py-24 lg:py-32 bg-muted/50">
        <div className="container mx-auto px-4 md:px-6">
        <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center mb-12">
          <h2 className="font-heading text-3xl leading-[1.1] sm:text-3xl md:text-6xl">
            Built with Modern Technology
          </h2>
          <p className="max-w-[42rem] leading-relaxed text-muted-foreground sm:text-xl sm:leading-8">
            For developers: RunMat leverages cutting-edge systems programming
          </p>
        </div>
        <div className="mx-auto grid justify-center gap-4 sm:grid-cols-2 md:max-w-[64rem]">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">V8-Inspired JIT Compilation</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription className="text-sm">
                Advanced tiered execution with Ignition interpreter and Turbine JIT compiler 
                using Cranelift for near-native performance on mathematical workloads.
              </CardDescription>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Memory Safety with Rust</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription className="text-sm">
                Zero-cost abstractions, guaranteed memory safety, and fearless concurrency 
                eliminate entire classes of bugs common in numerical computing.
              </CardDescription>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">GPU-Accelerated Graphics</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription className="text-sm">
                Modern rendering pipeline built on wgpu with WebGL/Metal/Vulkan backends 
                for interactive 60fps visualizations and scientific plotting.
              </CardDescription>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Instant Startup</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription className="text-sm">
                  Revolutionary snapshotting technology enables sub-5ms cold starts and 
                persistent workspace state across sessions.
              </CardDescription>
            </CardContent>
          </Card>
        </div>
        </div>
      </section>

      {/* Quick Install Section */}
      <section className="w-full py-16 md:py-24 lg:py-32 bg-muted/30">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center">
          <h2 className="font-heading text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
            Install in Seconds
          </h2>
          <p className="max-w-[42rem] leading-relaxed text-muted-foreground sm:text-xl sm:leading-8">
            Copy and paste one command to get started with RunMat
          </p>
          
          <OSInstallCommand className="w-full max-w-4xl" />
          
            <div className="flex flex-col sm:flex-row gap-4">
              <Button size="lg" asChild className="h-12 px-8 text-lg font-semibold bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-xl border-0 transition-all duration-200 hover:scale-105 hover:shadow-2xl">
                <Link href="/download">More Install Options</Link>
            </Button>
              <Button variant="outline" size="lg" asChild className="h-12 px-8">
              <Link href="/docs/getting-started">Get Started</Link>
            </Button>
          </div>
        </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="w-full py-16 md:py-24 lg:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-8 text-center">
          <h2 className="font-heading text-3xl leading-[1.1] sm:text-3xl md:text-6xl">
            Ready to Experience the Future?
          </h2>
          <p className="max-w-[42rem] leading-relaxed text-muted-foreground sm:text-xl sm:leading-8">
            Join researchers and engineers who&rsquo;ve made the switch to RunMat
          </p>
          <div className="flex gap-4">
            <Button size="lg" asChild>
              <Link href="/blog/introducing-runmat">Read the Story</Link>
            </Button>
            <Button variant="outline" size="lg" asChild>
              <Link href="https://github.com/runmat-org/runmat">View on GitHub</Link>
            </Button>
          </div>
        </div>
        </div>
      </section>
    </div>
  );
}