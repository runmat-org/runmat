import { Metadata } from "next";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { 
  ArrowRight, 
  Cpu, 
  Zap, 
  Database, 
  Monitor,
  GitBranch,
  Clock,
  Shield,
  BarChart3
} from "lucide-react";
import Link from "next/link";

export const metadata: Metadata = {
  title: "How RunMat Works - Architecture and Design",
  description: "Deep dive into RunMat's V8-inspired architecture, tiered execution, JIT compilation, garbage collection, and plotting system. A comprehensive technical guide.",
};

export default function HowItWorksPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto max-w-4xl px-4 md:px-6 py-16 md:py-24">
        
        {/* Header */}
        <div className="mb-12">
          <Badge variant="secondary" className="mb-4">Technical Deep Dive</Badge>
          <h1 className="text-4xl lg:text-5xl font-bold mb-6 text-foreground">
            How RunMat Works
          </h1>
          <p className="text-xl text-muted-foreground leading-relaxed">
            Traditional MATLAB interpreters can feel sluggish for compute-intensive workloads. RunMat takes a 
            different approach with a V8-inspired architecture that prioritizes performance without sacrificing 
            compatibility. Here&apos;s how we built a MATLAB engine that starts in 5ms and runs many workloads 
            significantly faster than existing implementations like GNU Octave.
          </p>
        </div>

        {/* The Problem with Traditional Scientific Computing */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">Why Octave and MATLAB are Slow</h2>
            <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-950/30 dark:to-orange-950/30 rounded-lg p-6 mb-8">
              <p className="text-lg leading-relaxed">
                <strong>Slow Startup:</strong> MATLAB takes 10+ seconds to start because it loads massive runtime environments, 
                initializes complex licensing systems, and builds symbol tables from scratch every time. Even with JIT compilation, 
                the startup overhead dominates short-running scripts.
              </p>
              <p className="text-lg leading-relaxed mt-4">
                <strong>Runtime Inefficiencies:</strong> Traditional interpreters execute code line-by-line, translating every 
                operation at runtime. A simple matrix multiplication gets parsed, analyzed, and dispatched to BLAS libraries — 
                every single time. Loop bodies get re-interpreted on each iteration.
              </p>
              <p className="text-lg leading-relaxed mt-4 text-muted-foreground">
                <strong>The result?</strong> You spend more time waiting for MATLAB to start than actually running your code, 
                especially for quick calculations and iterative development.
              </p>
            </div>
        </section>

        {/* The RunMat Solution */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">The RunMat Approach: Adaptive Compilation</h2>
          <p className="text-lg text-muted-foreground mb-8">
            RunMat takes a different approach inspired by modern JavaScript engines like V8. Instead of treating 
            every line of code the same, we use <strong>adaptive compilation</strong> that optimizes frequently-executed code paths.
          </p>

          <div className="grid md:grid-cols-2 gap-8 mb-8">
            <Card className="border-l-4 border-l-blue-500">
              <CardContent className="px-6 py-2">
                <div className="flex items-center mb-4">
                  <div className="bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded-full w-10 h-10 flex items-center justify-center text-lg font-bold mr-4">
                    <Zap className="h-5 w-5" />
                  </div>
                  <h3 className="text-xl font-semibold">Phase 1: Instant Execution</h3>
                </div>
                <p className="text-muted-foreground leading-relaxed">
                  Your code runs immediately through our <strong>Ignition interpreter</strong> — no compilation wait time. 
                  Good for interactive exploration, prototyping, and one-off calculations. While executing, we track 
                  which functions are called frequently.
                </p>
              </CardContent>
            </Card>
            <Card className="border-l-4 border-l-green-500">
              <CardContent className="px-6 py-2">
                <div className="flex items-center mb-4">
                  <div className="bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 rounded-full w-10 h-10 flex items-center justify-center text-lg font-bold mr-4">
                    <Cpu className="h-5 w-5" />
                  </div>
                  <h3 className="text-xl font-semibold">Phase 2: Machine Code</h3>
              </div>
                <p className="text-muted-foreground leading-relaxed">
                  Functions that run frequently get upgraded to native machine code via our <strong>Turbine JIT compiler</strong>. 
                  Same code, but now with optimizations like loop unrolling, function inlining, vectorization, and 
                  specialized paths for your specific data types and usage patterns.
                </p>
            </CardContent>
          </Card>
          </div>
          
          <div className="bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-950/30 dark:to-green-950/30 rounded-lg p-6">
            <div className="flex items-start">
              <BarChart3 className="h-6 w-6 text-blue-600 mr-3 mt-1 flex-shrink-0" />
              <div>
                <h4 className="text-lg font-semibold mb-3">Why This Approach Wins</h4>
          <p className="text-muted-foreground leading-relaxed">
                  Unlike traditional ahead-of-time compilers that optimize for average cases, our JIT compiler 
                  sees <em>exactly</em> how your code behaves at runtime. It knows your matrix sizes, your data types, 
                  your calling patterns — and optimizes accordingly. The 20% of your code that runs 80% of the time 
                  gets the full performance treatment.
                </p>
              </div>
            </div>
          </div>
        </section>



        {/* Performance Characteristics (moved up) */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">
            Performance Characteristics
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Startup Performance</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">GNU Octave</span>
                    <span className="font-semibold">914ms avg</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">RunMat</span>
                    <span className="font-semibold text-green-600">5ms avg</span>
                  </div>
                  <div className="flex justify-between border-t pt-2">
                    <span className="text-foreground font-medium">Speedup</span>
                    <span className="font-semibold text-green-600">182.93× faster</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Matrix Operations</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">GNU Octave</span>
                    <span className="font-semibold">822ms avg</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">RunMat</span>
                    <span className="font-semibold text-green-600">5ms avg</span>
                  </div>
                  <div className="flex justify-between border-t pt-2">
                    <span className="text-foreground font-medium">Speedup</span>
                    <span className="font-semibold text-green-600">164.40× faster</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Mathematical Functions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">GNU Octave</span>
                    <span className="font-semibold">868ms avg</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">RunMat (Interpreter)</span>
                    <span className="font-semibold text-green-600">5.7ms avg</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">RunMat (JIT)</span>
                    <span className="font-semibold text-green-600">5.3ms avg</span>
                  </div>
                  <div className="flex justify-between border-t pt-2">
                    <span className="text-foreground font-medium">Speedup</span>
                    <span className="font-semibold text-green-600">153-163x faster</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Control Flow</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">GNU Octave</span>
                    <span className="font-semibold">876ms avg</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">RunMat</span>
                    <span className="font-semibold text-green-600">5.7ms avg</span>
                  </div>
                  <div className="flex justify-between border-t pt-2">
                    <span className="text-foreground font-medium">Speedup</span>
                    <span className="font-semibold text-green-600">154.54x faster</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="mt-6 text-sm text-muted-foreground">
            Benchmarks run on Apple M2 Max (32GB RAM) under identical conditions. Full suite available in <code>/benchmarks</code>; reproduce with <code>./benchmarks/run_benchmarks.sh</code>.
          </div>
        </section>

        {/* Deep Dive: How the Magic Happens */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">Deep Dive: How the Magic Happens</h2>
          
          <div className="space-y-8">
            {/* Ignition Interpreter */}
            <div>
              <h3 className="text-2xl font-semibold mb-4 flex items-center">
                <Zap className="h-6 w-6 text-blue-600 mr-3" />
                The Ignition Interpreter: Speed from Day One
              </h3>
              <p className="text-lg text-muted-foreground mb-4">
                When you type <code>A = [1, 2; 3, 4]</code> and press enter, here&apos;s what happens in microseconds:
              </p>
              <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-6">
                <ol className="space-y-3 text-muted-foreground">
                  <li><strong>1. Lexical Analysis:</strong> Your MATLAB code gets tokenized into meaningful chunks</li>
                  <li><strong>2. Parsing:</strong> Tokens become an Abstract Syntax Tree (AST) representing the structure</li>
                  <li><strong>3. HIR Translation:</strong> The AST becomes High-level Intermediate Representation for optimization</li>
                  <li><strong>4. Bytecode Generation:</strong> HIR compiles to compact bytecode instructions</li>
                  <li><strong>5. Immediate Execution:</strong> Bytecode runs instantly while profiling counters track &quot;hot&quot; functions</li>
                </ol>
              </div>
              <p className="text-muted-foreground mt-4">
                This entire pipeline completes in under 5ms — faster than most editors can update their syntax highlighting.
              </p>
                </div>

            {/* Turbine JIT */}
                <div>
              <h3 className="text-2xl font-semibold mb-4 flex items-center">
                <Cpu className="h-6 w-6 text-green-600 mr-3" />
                The Turbine JIT: When Performance Really Matters
              </h3>
              <p className="text-lg text-muted-foreground mb-4">
                Once a function gets called enough times (our threshold is carefully tuned), Turbine kicks in:
              </p>
              <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-6">
                <ul className="space-y-3 text-muted-foreground">
                  <li><strong>• Type Specialization:</strong> Generate optimized code paths for your specific data types</li>
                  <li><strong>• Loop Optimization:</strong> Unroll tight loops and vectorize operations using SIMD instructions</li>
                  <li><strong>• Function Inlining:</strong> Eliminate call overhead for frequently-used builtins like <code>sin()</code> or <code>cos()</code></li>
                  <li><strong>• Memory Layout Optimization:</strong> Arrange data structures for cache-friendly access patterns</li>
                  <li><strong>• Dead Code Elimination:</strong> Remove branches that never execute in your specific use case</li>
                  </ul>
              </div>
              <p className="text-muted-foreground mt-4">
                The result? Machine code optimized for your specific usage patterns, with specialized paths 
                for your data types and calling conventions.
              </p>
            </div>
          </div>
        </section>

        {/* Memory: The Foundation of Speed */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground flex items-center">
            <Database className="h-8 w-8 text-purple-600 mr-3" />
            Memory: The Foundation of Speed
          </h2>
          <p className="text-lg text-muted-foreground mb-6">
            MATLAB&apos;s strength has always been matrices, so we designed our memory system from the ground up for numerical computing:
          </p>
          
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardContent className="p-6">
                <h4 className="text-lg font-semibold mb-3">Zero-Copy Arrays</h4>
                <p className="text-muted-foreground">
                  Arrays use column-major layout (just like MATLAB) and pass directly to BLAS/LAPACK libraries 
                  without any data copying. Large matrix operations can work directly with the underlying memory 
                  without unnecessary data movement.
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-6">
                <h4 className="text-lg font-semibold mb-3">Copy-on-Write Semantics</h4>
                <p className="text-muted-foreground">
                  When you write <code>B = A</code>, we don&apos;t copy the entire matrix. Instead, B shares A&apos;s memory 
                  until you modify one of them. This preserves MATLAB semantics while dramatically reducing memory usage.
                </p>
            </CardContent>
          </Card>
          </div>
        </section>

        {/* Garbage Collection */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">Garbage Collection: Smart Memory Management</h2>
          <p className="text-lg text-muted-foreground mb-6">
            Scientific computing creates lots of temporary arrays — intermediate results in calculations, 
            temporary matrices in algorithms, etc. Our garbage collector is designed with this pattern in mind:
          </p>
          
          <div className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-950/30 dark:to-blue-950/30 rounded-lg p-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="text-lg font-semibold mb-3">Generational Collection</h4>
                <p className="text-muted-foreground">
                  Young objects (like temporary calculation results) get collected quickly and frequently. 
                  Old objects (like your main data matrices) get touched rarely, minimizing overhead.
                  </p>
                </div>
              <div>
                <h4 className="text-lg font-semibold mb-3">Background Threads</h4>
                <p className="text-muted-foreground">
                  Memory cleanup happens in the background while your calculations continue. No more 
                  &quot;GC pause&quot; interruptions in the middle of your analysis.
                  </p>
                </div>
              </div>
          </div>
        </section>

        {/* Instant Startup */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground flex items-center">
            <Clock className="h-8 w-8 text-orange-600 mr-3" />
            Instant Startup: 2000x Faster Boot Times
          </h2>
          <p className="text-lg text-muted-foreground mb-6">
            Remember waiting 10+ seconds for MATLAB to start? We solved that with a combination of 
            <strong> snapshotting, lightweight runtime design, and fast compilation</strong>:
          </p>
          
          <div className="space-y-4">
                        <Card>
              <CardContent className="p-6">
                <h4 className="text-lg font-semibold mb-3">Lightweight Runtime Architecture</h4>
                <p className="text-muted-foreground">
                  Built from the ground up in Rust with minimal startup overhead. No massive Java runtimes, 
                  no complex licensing checks, no bloated legacy code. Just a lean, fast runtime that gets 
                  out of your way and lets you focus on your work.
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-6">
                <h4 className="text-lg font-semibold mb-3">Instant Compilation Pipeline</h4>
                <p className="text-muted-foreground">
                  Our compilation pipeline is designed for speed: fast lexing, efficient parsing, and immediate 
                  bytecode execution. Combined with pre-warmed snapshots of the standard library, we eliminate 
                  the cold-start penalty that plagues traditional environments.
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Modern Plotting */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground flex items-center">
            <Monitor className="h-8 w-8 text-green-600 mr-3" />
            Modern Plotting: Built for the GPU Era
          </h2>
          <p className="text-lg text-muted-foreground mb-6">
            Traditional MATLAB plotting is CPU-bound and struggles with large datasets. We rebuilt plotting 
            from scratch for the modern era:
          </p>
          
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardContent className="p-6">
                <h4 className="text-lg font-semibold mb-3">GPU-Accelerated Rendering</h4>
                <p className="text-muted-foreground">
                  Built on WebGPU (wgpu) with custom WGSL shaders. Designed to handle large datasets efficiently 
                  by leveraging GPU acceleration for rendering, with smooth interaction for scatter plots, 
                  line charts, and other visualizations.
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-6">
                <h4 className="text-lg font-semibold mb-3">Interactive by Default</h4>
                <p className="text-muted-foreground">
                  Zoom, pan, rotate — all built-in and responsive. Level-of-detail rendering means performance 
                  stays smooth even when you&apos;re exploring huge datasets. Because the best insights often come 
                  from interactive exploration.
                </p>
            </CardContent>
          </Card>
          </div>
        </section>

        {/* Complete Ecosystem */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground flex items-center">
            <GitBranch className="h-8 w-8 text-blue-600 mr-3" />
            A Complete Scientific Computing Ecosystem
          </h2>
          <p className="text-lg text-muted-foreground mb-6">
            RunMat isn&apos;t just a faster interpreter — it&apos;s a complete rethinking of the scientific computing experience:
          </p>
          
          <div className="space-y-6">
            <Card>
              <CardContent className="p-6">
                <h4 className="text-xl font-semibold mb-3">Lightning-Fast REPL</h4>
                <p className="text-muted-foreground mb-4">
                  No more waiting for calculations to complete. The interactive shell starts instantly, 
                  remembers your variables and functions between sessions, and gives you immediate feedback 
                  on syntax errors before you even press enter.
                </p>
                <div className="bg-slate-100 dark:bg-slate-800 rounded p-3 font-mono text-sm">
                  <span className="text-green-600">runmat&gt;</span> A = randn(1000, 1000); B = A * A&apos;; trace(B)<br/>
                  <span className="text-blue-600">ans =</span> 1000.0  <span className="text-gray-500">% Computed in 2ms</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <h4 className="text-xl font-semibold mb-3">First-Class Jupyter Integration</h4>
                <p className="text-muted-foreground mb-4">
                  Built for the notebook era. Install RunMat as a Jupyter kernel with one command, 
                  and enjoy the same performance in your favorite notebook environment. 
                  Rich display support for plots, matrices, and data structures.
                </p>
                <div className="bg-slate-100 dark:bg-slate-800 rounded p-3 font-mono text-sm">
                  <span className="text-green-600">$</span> runmat --install-kernel<br/>
                  <span className="text-gray-500">✓ RunMat Jupyter kernel installed successfully!</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <h4 className="text-xl font-semibold mb-3">Modern Cloud & Container Deployment</h4>
                <p className="text-muted-foreground mb-4">
                  Single static binary with zero dependencies makes deployment trivial. Run MATLAB code in Docker 
                  containers, Kubernetes clusters, or cloud functions. No complex runtime environments, no licensing 
                  servers — just copy the binary and go.
                </p>
                <div className="bg-slate-100 dark:bg-slate-800 rounded p-3 font-mono text-sm">
                  <span className="text-green-600">$</span> echo &quot;A = [1, 2; 3, 4]; det(A)&quot; | runmat<br/>
                  <span className="text-blue-600">ans =</span> -2.0
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* What's Next */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground flex items-center">
            <Shield className="h-8 w-8 text-purple-600 mr-3" />
            What&apos;s Next: The Future is Bright
          </h2>
          <p className="text-lg text-muted-foreground mb-6">
            We&apos;re just getting started. Here&apos;s what&apos;s coming to make scientific computing even better:
          </p>
          
          <div className="space-y-6">
            <Card className="border-l-4 border-l-orange-500">
              <CardContent className="p-6">
                <h4 className="text-xl font-semibold mb-3">Cross-Platform Deployment: Run Anywhere</h4>
                <p className="text-muted-foreground">
                  Compile your MATLAB code to run on any platform — from embedded microcontrollers 
                  and edge devices to web browsers via WebAssembly, mobile devices, and cloud infrastructure. 
                  Write once in MATLAB, deploy everywhere from IoT sensors to high-performance clusters.
                </p>
              </CardContent>
            </Card>

            <Card className="border-l-4 border-l-red-500">
              <CardContent className="p-6">
                <h4 className="text-xl font-semibold mb-3">GPU Compute: Massively Parallel Everything</h4>
                <p className="text-muted-foreground">
                  Why limit matrix operations to your CPU? Direct CUDA and ROCm integration will 
                  automatically offload heavy computations to your GPU, turning your graphics card 
                  into a scientific supercomputer - without having to purchase yet another license.
                </p>
              </CardContent>
            </Card>

            <Card className="border-l-4 border-l-green-500">
              <CardContent className="p-6">
                <h4 className="text-xl font-semibold mb-3">Enterprise Freedom: Break the License Prison</h4>
                <p className="text-muted-foreground">
                  No more per-seat licensing, no network license servers, no vendor audits. Deploy RunMat 
                  across unlimited machines, scale teams without budget explosions, and never face 
                  astronomical renewal costs again. True computational freedom for organizations of any size.
                </p>
              </CardContent>
            </Card>

          </div>
        </section>

        {/* Learn More */}
        <section>
          <Card className="bg-gradient-to-r from-blue-50 to-orange-50 dark:from-blue-900/20 dark:to-orange-900/20 border-blue-200 dark:border-blue-800">
            <CardContent className="p-6 text-center">
              <h3 className="text-lg font-semibold mb-4 text-foreground">
                Ready to Dive Deeper?
              </h3>
              <p className="text-muted-foreground mb-6">
                Explore RunMat&apos;s source code or try it yourself to see this architecture in action.
              </p>
              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <Button asChild>
                  <Link href="https://github.com/runmat-org/runmat" target="_blank" rel="noopener noreferrer">
                    View Source Code
                  </Link>
                </Button>
                <Button variant="outline" asChild>
                  <Link href="/docs/getting-started">
                    Get Started
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
                <Button variant="outline" asChild>
                  <Link href="/docs/builtin-functions">
                    API Reference
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