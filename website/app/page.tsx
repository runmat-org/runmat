import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CodeBlock } from "@/components/CodeBlock";
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
            A free, high-performance alternative to MATLAB that runs your existing code faster and more reliably. 
            No license fees, no vendor lock-in, just blazing-fast numerical computing with beautiful visualizations.
          </p>
            <div className="flex gap-4">
              <Button size="lg" asChild>
                <Link href="/download">Download RustMat</Link>
              </Button>
              <Button variant="outline" size="lg" asChild>
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
              Why Choose RustMat?
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
        <div className="mx-auto max-w-[58rem]">
          <Card>
            <CardContent className="p-6">
              <CodeBlock language="matlab">{`% Generate data
x = linspace(0, 4 * pi, 1000);
y = sin(x) .* exp(-x/10);

% Create beautiful plot
plot(x, y);

% Matrix operations
A = randn(1000, 1000);
B = A * A';
eigenvals = eig(B);`}</CodeBlock>
            </CardContent>
          </Card>
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
            For developers: RustMat leverages cutting-edge systems programming
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
            Copy and paste one command to get started with RustMat
          </p>
          
          <OSInstallCommand className="w-full max-w-4xl" />
          
          <div className="flex gap-4">
            <Button size="lg" asChild>
              <Link href="/download">More Install Options</Link>
            </Button>
            <Button variant="outline" size="lg" asChild>
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
            Join researchers and engineers who&rsquo;ve made the switch to RustMat
          </p>
          <div className="flex gap-4">
            <Button size="lg" asChild>
              <Link href="/blog/introducing-rustmat">Read the Story</Link>
            </Button>
            <Button variant="outline" size="lg" asChild>
              <Link href="https://github.com/rustmat/rustmat">View on GitHub</Link>
            </Button>
          </div>
        </div>
        </div>
      </section>
    </div>
  );
}