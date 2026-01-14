import { Metadata } from "next";
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
  title: "How RunMat Works | Docs",
  description: "RunMat implements the full MATLAB language grammar and core semantics with a modern, V8-inspired engine, a slim core, generational GC, and a package-first model.",
};

export default function HowItWorksPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 md:px-0 py-16 md:py-4">
        
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl lg:text-5xl font-bold mb-6 text-foreground">
            How RunMat Works
          </h1>
          <p className="text-xl text-muted-foreground leading-relaxed">
            RunMat is a GPU-accelerated runtime for MATLAB code that automatically optimizes your math across CPU and GPU—often outperforming hand-tuned PyTorch without any kernel programming. 
          </p>
          <p className="text-lg text-muted-foreground leading-relaxed mt-8">
            It implements the full language grammar and core semantics of the MATLAB programming language — arrays and indexing, control flow, functions and multiple returns, cells/structs, and
            classdef OOP — with a fast engine that can execute your code quickly on CPU or GPU. RunMat is a clean, fast runtime for the MATLAB language.
          </p>
        </div>

        {/* The Problem with Traditional Scientific Computing */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">Why existing approaches are slow</h2>
          <div className="bg-gradient-to-r from-amber-100 to-yellow-100 dark:from-amber-900/40 dark:to-yellow-900/40 rounded-lg p-6 mb-8 border border-amber-200 dark:border-amber-800">
              <p className="text-lg leading-relaxed">
              <strong>Traditional MATLAB:</strong> Slow startup, interpreted loops, no GPU support without expensive toolboxes.
            </p>
            <p className="text-lg leading-relaxed mt-4">
              <strong>Octave:</strong> Significantly slower than MATLAB, especially for loop-heavy code. It is also not a full implementation of the MATLAB language, and does not have a GPU layer.
              </p>
              <p className="text-lg leading-relaxed mt-4">
              <strong>PyTorch/TensorFlow:</strong> Requires rewriting code, explicit device management, kernel programming knowledge.
              </p>
              <p className="text-lg leading-relaxed mt-4 text-muted-foreground">
              <strong>The result:</strong> Scientists choose between ease (MATLAB) or speed (PyTorch)—not both.
              </p>
            </div>
        </section>

        {/* The RunMat Solution */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">RunMat: slim core + adaptive compilation</h2>
          <p className="text-lg text-muted-foreground mb-8">
            Inspired by modern JavaScript engines, RunMat starts instantly in a high-performance interpreter and
            upgrades hot code to native machine code. The core is intentionally small and predictable, so the engine
            can be ruthlessly optimized.
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
                  Perfect for interactive work, tests, and scripts. While executing, we profile which functions are hot.
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
                  Functions that run frequently get upgraded to native machine code via the <strong>Turbine JIT</strong>:
                  loop optimizations, type-specialized paths, and vectorization tailored to your data and call patterns.
                </p>
            </CardContent>
          </Card>
          </div>
          
          <div className="bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-950/30 dark:to-green-950/30 rounded-lg p-6">
            <div className="flex items-start">
              <BarChart3 className="h-6 w-6 text-blue-600 mr-3 mt-1 flex-shrink-0" />
              <div>
                <h4 className="text-lg font-semibold mb-3">Why this approach wins</h4>
                <p className="text-muted-foreground leading-relaxed">
                  The RunMat JIT sees <em>exactly</em> how your code behaves: matrix sizes, datatypes, calling patterns.
                  The 20% of your code that runs 80% of the time gets the full optimization budget.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Automatic GPU Acceleration */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">Automatic GPU Acceleration: Faster than Hand-Optimized Code</h2>
          <p className="text-lg leading-relaxed mb-6 text-muted-foreground">
            RunMat&apos;s fusion engine automatically:
          </p>
          <p className="text-sm text-muted-foreground mb-4">
            For a product overview, read the{" "}
            <Link href="/blog/runmat-accelerate-fastest-runtime-for-your-math" className="underline">
              RunMat Accelerate announcement
            </Link>
            .
          </p>
          <ul className="space-y-4 mb-6 text-lg leading-relaxed">
            <li className="flex items-start">
              <span className="text-green-600 mr-3 mt-1">•</span>
              <span>Detects GPU-friendly operations and routes them intelligently</span>
            </li>
            <li className="flex items-start">
              <span className="text-green-600 mr-3 mt-1">•</span>
              <span>Fuses operation chains into optimized kernels (like PyTorch&apos;s <code className="bg-muted px-1.5 py-0.5 rounded">torch.compile</code>, but automatic)</span>
            </li>
            <li className="flex items-start">
              <span className="text-green-600 mr-3 mt-1">•</span>
              <span>Manages memory placement between CPU/GPU for maximum throughput</span>
            </li>
            <li className="flex items-start">
              <span className="text-green-600 mr-3 mt-1">•</span>
              <span>Often beats hand-tuned PyTorch—see our <Link href="/benchmarks" className="text-blue-600 hover:underline">benchmarks</Link></span>
            </li>
          </ul>
          <p className="text-lg leading-relaxed font-medium">
            No device flags. No kernel code. Just math that runs at maximum speed.
          </p>
        </section>

        {/* Language and Semantics */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">Language compatibility and semantics</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-muted-foreground">
            <Card>
              <CardHeader><CardTitle>Full grammar</CardTitle></CardHeader>
              <CardContent>
                Expressions, statements, functions with multiple returns, indexing <code>() [] { }</code>, command-form,
                and classdef are all accepted by the parser.
              </CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle>Core semantics</CardTitle></CardHeader>
              <CardContent>
                Column-major arrays and slicing (<code>end</code>, colon ranges, logical masks); cells and structs; OOP
                dispatch and <code>subsref</code>/<code>subsasgn</code>; <code>try/catch</code>, <code>global</code>, <code>persistent</code>.
              </CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle>Slim builtins</CardTitle></CardHeader>
              <CardContent>
                The core ships a curated standard library. Everything else comes from packages — native (Rust)
                or source (MATLAB code) packages.
              </CardContent>
            </Card>
          </div>
        </section>

        {/* HIR & AST: acceleration and tooling */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">HIR & AST: foundation for acceleration and tooling</h2>
          <p className="text-lg text-muted-foreground mb-6">
            RunMat lowers source to a high-level, typed IR (HIR) with flow-sensitive inference. A stable AST→HIR pipeline
            gives the engine precise structure and types, which unlocks acceleration and first-class editor tooling.
          </p>
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardContent className="p-6">
                <h4 className="text-lg font-semibold mb-3">Seamless Accelerate</h4>
                <p className="text-muted-foreground">
                  Tensor/Matrix ops are explicit in HIR, so the planner can route work to CPU or GPU, fuse common
                  elementwise chains, and support reverse-mode autograd by default. This lets the planner
                  dispatch and manage memory and operations across GPU/CPU/TPU/etc without needing any user code changes.
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-6">
                <h4 className="text-lg font-semibold mb-3">Great IntelliSense (LSP)</h4>
                <p className="text-muted-foreground">
                  Typed nodes power hover types, signature help, go-to-definition across packages, property/field
                  completion from the class registry, and precise diagnostics. The same HIR backs the interpreter and JIT.
                </p>
              </CardContent>
            </Card>
          </div>
          <div className="bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-950/30 dark:to-green-950/30 rounded-lg p-6 mt-6">
            <div className="flex items-start">
              <BarChart3 className="h-6 w-6 text-blue-600 mr-3 mt-1 flex-shrink-0" />
              <div>
                <h4 className="text-lg font-semibold mb-3">Portable by design</h4>
                <p className="text-muted-foreground leading-relaxed">
                  Typed HIR lowers to Cranelift IR, which makes it trivial to target multiple architectures. That makes RunMat
                  platform-agnostic and lightweight: small static binaries, predictable performance on Linux/macOS/Windows or embedded devices,
                  and room for ahead-of-time or cached compilation where it helps startup.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-950/30 dark:to-green-950/30 rounded-lg p-6 mt-6">
            <div className="flex items-start">
              <BarChart3 className="h-6 w-6 text-blue-600 mr-3 mt-1 flex-shrink-0" />
              <div>
                <h4 className="text-lg font-semibold mb-3">Why this architecture is different</h4>
                <p className="text-muted-foreground leading-relaxed">
                  Octave&apos;s classic interpreter does not expose a typed IR, limiting deep optimization and IDE tooling.
                  MATLAB has powerful internal IRs but limited external LSP integration. RunMat&apos;s stable HIR/ABI and
                  Cranelift backend make accelerators and editors plug in cleanly across platforms.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Performance Characteristics */}
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
            Benchmarks run on Apple M2 Max (32GB RAM) under identical conditions. Full suite available in <code>/benchmarks</code>; reproduce with <code>./benchmarks/run_benchmarks.sh</code>. Note that we cannot benchmark RunMat against MathWorks&apos;s MATLAB and ecosystem because installing it requires agreeing to their strict and dense license agreement, which we will not.
          </div>
        </section>

        {/* How GPU Acceleration Works */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">How GPU Acceleration Works</h2>
          
          <div className="space-y-8">
            <div>
              <h3 className="text-2xl font-semibold mb-4">Stage 1: Capture the Math</h3>
              <p className="text-lg text-muted-foreground leading-relaxed">
                RunMat builds an acceleration graph of your operations, tracking shapes, dependencies, and operation types. This graph captures the mathematical intent without being tied to specific hardware, enabling intelligent optimization decisions.
              </p>
            </div>

            <div>
              <h3 className="text-2xl font-semibold mb-4">Stage 2: Intelligent Routing</h3>
              <p className="text-lg text-muted-foreground leading-relaxed">
                Auto-offload heuristics decide CPU vs GPU based on size and operation type. Small operations stay on CPU to avoid transfer overhead, while large computations automatically route to GPU for maximum throughput.
              </p>
            </div>

            <div>
              <h3 className="text-2xl font-semibold mb-4">Stage 3: Kernel Fusion</h3>
              <p className="text-lg text-muted-foreground leading-relaxed">
                Sequential operations compile into single GPU kernels, eliminating memory transfers between steps. Instead of launching ten separate kernels, RunMat fuses them into one optimized GPU program that runs end-to-end on device.
              </p>
            </div>

            <div>
              <h3 className="text-2xl font-semibold mb-4">Stage 4: Execution with Residency</h3>
              <p className="text-lg text-muted-foreground leading-relaxed">
                Data stays on GPU between operations, copying back only when needed. This residency-aware execution minimizes host↔device transfers, keeping your data where it can be processed fastest.
              </p>
            </div>
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
              <div className="bg-muted dark:bg-slate-900 rounded-lg p-6 border border-border dark:border-slate-800">
                <ol className="space-y-3 text-foreground dark:text-muted-foreground">
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
                Once a function gets called enough times (the JIT is configurable, with good defaults), Turbine kicks in:
              </p>
              <div className="bg-muted dark:bg-slate-900 rounded-lg p-6 border border-border dark:border-slate-800">
                <ul className="space-y-3 text-foreground dark:text-muted-foreground">
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
            MATLAB&apos;s strength has always been matrices (and tensors), so we designed our memory system from the ground up for numerical computing:
          </p>
          
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardContent className="p-6">
                <h4 className="text-lg font-semibold mb-3">Zero-Copy Arrays</h4>
                <p className="text-muted-foreground">
                  Arrays use column-major layout (just like MATLAB) and pass directly to BLAS/LAPACK libraries 
                  without any data copying. Large matrix/tensor operations can work directly with the underlying memory 
                  without unnecessary data movement.
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-6">
                <h4 className="text-lg font-semibold mb-3">Copy-on-Write Semantics</h4>
                <p className="text-muted-foreground">
                  When you write <code>B = A</code>, we don&apos;t copy the entire matrix/tensor. Instead, B shares A&apos;s memory 
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
                  Old objects (like your main data matrices) are scanned rarely via remembered sets, minimizing overhead.
                  </p>
                </div>
              <div>
                <h4 className="text-lg font-semibold mb-3">Short, predictable pauses</h4>
                <p className="text-muted-foreground">
                  Minor collections are stop-the-world but fast. Write barriers track old→young edges so minor GCs
                  only scan what changed, not the entire heap.
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
            Remember how MATLAB takes 10+ seconds to start? It&apos;s a symptom of a larger problem.
            RunMat solves runtime startup time using modern techniques like <strong>snapshotting, lightweight runtime design, and fast compilation</strong>.
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
            Modern Plotting (Pre-release): Built for the GPU Era
          </h2>
          <p className="text-lg text-muted-foreground mb-6">
            Traditional MATLAB plotting is CPU-bound and struggles with large datasets. RunMat&apos;s plotting layer is still in pre-release: simple 2D line and scatter plots render on the GPU today, while filled shapes, 3D surfaces, and richer controls are actively being built:
          </p>
          
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardContent className="p-6">
                <h4 className="text-lg font-semibold mb-3">GPU-Accelerated Rendering (Today)</h4>
                <p className="text-muted-foreground">
                  Built on WebGPU (wgpu) with custom WGSL shaders. Current builds handle simple 2D line and scatter datasets efficiently by keeping them on the GPU, and we&apos;re extending the renderer to handle filled shapes and larger plot families.
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-6">
                <h4 className="text-lg font-semibold mb-3">Interactive by Default (Work in Progress)</h4>
                <p className="text-muted-foreground">
                  Basic zoom and pan controls ship today; more advanced rotation, camera presets, and LOD tricks are on the roadmap so that larger datasets feel smooth once the richer plot types land.
                </p>
            </CardContent>
          </Card>
          </div>
        </section>

        {/* Packages and Ecosystem */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground flex items-center">
            <GitBranch className="h-8 w-8 text-blue-600 mr-3" />
            Packages: extend the runtime without bloating the core
          </h2>
          <p className="text-lg text-muted-foreground mb-6">
            RunMat keeps the core minimal. New functions, types, and accelerators ship as packages — native (Rust)
            for maximum performance, or source (MATLAB) for portability.
          </p>
          <p className="text-sm text-muted-foreground mb-4">
            See the draft <Link href="/docs/package-manager" className="underline">Package Manager design</Link> for how registries, semver, and native/source packages will work.
          </p>
          
          <div className="space-y-6">
            <Card>
              <CardContent className="p-6">
                <h4 className="text-xl font-semibold mb-3">Lightning-Fast REPL</h4>
                <p className="text-muted-foreground mb-4">
                  No more waiting for calculations to complete. The interactive shell starts instantly, 
                  remembers your variables and functions within your session, and provides fast syntax error
                  detection with clear, helpful error messages.
                </p>
                <div className="bg-slate-100 dark:bg-slate-800 rounded p-3 font-mono text-sm">
                  <span className="text-green-600">runmat&gt;</span> A = randn(1000, 1000); B = A * A&apos;;<br />
                  { /* Note: This is a placeholder for the actual computation time. Need to measure this after implementing tic / toc */}
                  <span className="text-blue-600">ans =</span> 1000.0  {false && <span className="text-gray-500">% Computed in 2us</span>}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <h4 className="text-xl font-semibold mb-3">First-class Jupyter integration</h4>
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
                <h4 className="text-xl font-semibold mb-3">Modern cloud & container deployment</h4>
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

            <Card className="border-l-4 border-l-green-500">
              <CardContent className="p-6">
                <h4 className="text-xl font-semibold mb-3">Enterprise Freedom: Break the License Prison</h4>
                <p className="text-muted-foreground">
                  No per-seat licensing, no network license servers, no vendor audits. Deploy RunMat 
                  across unlimited machines, scale teams without budget explosions, and never face 
                  astronomical renewal costs again. True computational freedom for organizations of any size.
                </p>
              </CardContent>
            </Card>

          </div>
        </section>

        {/* Contrast: MATLAB, Octave, RunMat */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">How RunMat differs</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <CardHeader><CardTitle>MATLAB</CardTitle></CardHeader>
              <CardContent className="text-muted-foreground">
                Proprietary, massive standard library, heavy startup, license-gated deployment.
                Incredible breadth; less flexible for custom runtime needs.
              </CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle>GNU Octave</CardTitle></CardHeader>
              <CardContent className="text-muted-foreground">
                Open-source and compatible with many scripts; classic interpreter architecture
                emphasizes portability over peak performance and startup speed.
              </CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle>RunMat</CardTitle></CardHeader>
              <CardContent className="text-muted-foreground">
                Open-source, modern engine with full language grammar and core semantics,
                fast startup, tiered execution, generational GC, and a package-first library model.
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
                  <Link href="/docs/reference/builtins">API Reference</Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </section>
      </div>
    </div>
  );
}