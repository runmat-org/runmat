import { Metadata } from "next";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  Calculator, 
  TrendingUp, 
  BarChart3, 
  GitBranch,
  Database,
  Settings,
  Users,
  Brain,
  Code,
  Target,
  GitPullRequest
} from "lucide-react";

export const metadata: Metadata = {
  title: "RunMat Builtin Functions: Rapid MATLAB Compatibility",
  description: "Discover RunMat's revolutionary approach to building MATLAB-compatible functions at unprecedented speed using modern tooling and community contributions.",
};

export default function BuiltinFunctionsPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto max-w-4xl px-4 md:px-6 py-16 md:py-24">
        
        {/* Header */}
        <div className="mb-12">
          <Badge variant="secondary" className="mb-4">Builtin Functions</Badge>
          <h1 className="text-4xl lg:text-5xl font-bold mb-6 text-foreground">
            Builtin Functions Reference
          </h1>
          <p className="text-xl text-muted-foreground leading-relaxed mb-6">
            RunMat&apos;s architecture is designed for rapid implementation of MATLAB-compatible functions. 
            Our macro-based builtin system, combined with modern tooling and community contributions, 
            enables efficient expansion of the function library toward full MATLAB compatibility.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
            <div className="bg-blue-50 dark:bg-blue-950/30 rounded-lg p-4">
              <div className="text-2xl font-bold text-blue-600">50+</div>
              <div className="text-sm text-muted-foreground">Functions Implemented</div>
        </div>
            <div className="bg-green-50 dark:bg-green-950/30 rounded-lg p-4">
              <div className="text-2xl font-bold text-green-600">~1000</div>
              <div className="text-sm text-muted-foreground">MATLAB Core Functions</div>
                </div>
            <div className="bg-purple-50 dark:bg-purple-950/30 rounded-lg p-4">
              <div className="text-2xl font-bold text-purple-600">Open</div>
              <div className="text-sm text-muted-foreground">Community Driven</div>
                </div>
                </div>
              </div>

        {/* Quick Reference */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-foreground">
            Quick Reference
          </h2>
          
          <p className="text-muted-foreground mb-6">
            Here&apos;s a quick overview of the most commonly used functions:
          </p>

          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Mathematical Constants</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <code className="bg-muted px-2 py-1 rounded">pi</code>
                    <span className="text-muted-foreground/70 ml-2">π constant</span>
                  </div>
                  <div>
                    <code className="bg-muted px-2 py-1 rounded">e</code>
                    <span className="text-muted-foreground/70 ml-2">Euler&apos;s number</span>
                  </div>
                  <div>
                    <code className="bg-muted px-2 py-1 rounded">inf</code>
                    <span className="text-muted-foreground/70 ml-2">Infinity</span>
                  </div>
                  <div>
                    <code className="bg-muted px-2 py-1 rounded">nan</code>
                    <span className="text-muted-foreground/70 ml-2">Not a Number</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Array Creation</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm font-mono">
                  <div><code>zeros(m, n)</code> <span className="text-muted-foreground/70 font-sans">- Create m×n matrix of zeros</span></div>
                  <div><code>ones(m, n)</code> <span className="text-muted-foreground/70 font-sans">- Create m×n matrix of ones</span></div>
                  <div><code>eye(n)</code> <span className="text-muted-foreground/70 font-sans">- Create n×n identity matrix</span></div>
                  <div><code>rand(m, n)</code> <span className="text-muted-foreground/70 font-sans">- Create m×n matrix of random numbers</span></div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Mathematical Functions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm font-mono">
                  <div className="space-y-2">
                    <div><code>sin(x), cos(x), tan(x)</code></div>
                    <div><code>asin(x), acos(x), atan(x)</code></div>
                    <div><code>exp(x), log(x), log10(x)</code></div>
                    <div><code>sqrt(x), abs(x), sign(x)</code></div>
                  </div>
                  <div className="space-y-2">
                    <div><code>round(x), floor(x), ceil(x)</code></div>
                    <div><code>min(x), max(x), sum(x)</code></div>
                    <div><code>mean(x), std(x), var(x)</code></div>
                    <div><code>real(x), imag(x), angle(x)</code></div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Plotting Functions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm font-mono">
                  <div><code>plot(x, y)</code> <span className="text-muted-foreground/70 font-sans">- 2D line plot</span></div>
                  <div><code>scatter(x, y)</code> <span className="text-muted-foreground/70 font-sans">- 2D scatter plot</span></div>
                  <div><code>bar(x, y)</code> <span className="text-muted-foreground/70 font-sans">- Bar chart</span></div>
                  <div><code>histogram(x)</code> <span className="text-muted-foreground/70 font-sans">- Histogram</span></div>
                  <div><code>scatter3(x, y, z)</code> <span className="text-muted-foreground/70 font-sans">- 3D scatter plot</span></div>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Macro-Based Architecture - Detailed Section */}
        <section className="mb-12">
          <div className="flex items-center mb-6">
            <Code className="h-6 w-6 mr-3 text-blue-600" />
            <h2 className="text-3xl font-bold text-foreground">
              Macro-Based Architecture
            </h2>
          </div>
          
          <div className="bg-gradient-to-r from-blue-50/50 to-slate-50/50 dark:from-blue-950/30 dark:to-slate-950/30 rounded-xl p-8 mb-8">
            <p className="text-lg text-muted-foreground leading-relaxed mb-6">
              RunMat&apos;s <code className="bg-muted px-2 py-1 rounded text-sm">runtime_builtin</code> macro 
              dramatically simplifies how MATLAB functions are implemented. What traditionally requires complex 
              registration code and manual type handling becomes a simple attribute on a Rust function, allowing
              us to implement new functions at a rapid pace.
            </p>
            
            <div className="space-y-8">
              <div>
                <h4 className="text-xl font-semibold mb-4 text-foreground">Simple Implementation</h4>
                <div className="bg-slate-100 dark:bg-slate-800 rounded-lg p-8 font-mono text-base leading-relaxed">
                  <div className="text-blue-600 mb-1">#[runtime_builtin(name = &quot;sin&quot;)]</div>
                  <div> <span className="text-gray-600 dark:text-gray-400">fn</span> <span className="text-green-600">sin_builtin</span>(x: <span className="text-orange-600">f64</span>) -&gt; <span className="text-purple-600">Result</span>&lt;<span className="text-orange-600">f64</span>, <span className="text-red-600">String</span>&gt;</div>
                  
                  <div className="mt-4 text-blue-600">#[runtime_builtin(name = &quot;sin&quot;)]</div>
                  <div> <span className="text-gray-600 dark:text-gray-400">fn</span> <span className="text-green-600">sin_matrix</span>(x: <span className="text-orange-600">Matrix</span>) -&gt; <span className="text-purple-600">Result</span>&lt;<span className="text-orange-600">Matrix</span>, <span className="text-red-600">String</span>&gt;</div>
                </div>
              </div>
              
              <div>
                <h4 className="text-xl font-semibold mb-4 text-foreground">Automatic Features</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 flex-shrink-0"></div>
                    <div>
                      <div className="font-semibold text-foreground">Function Overloading</div>
                      <div className="text-sm text-muted-foreground">Multiple implementations with the same name automatically dispatch based on argument types</div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-green-600 rounded-full mt-2 flex-shrink-0"></div>
                    <div>
                      <div className="font-semibold text-foreground">Error Propagation</div>
                      <div className="text-sm text-muted-foreground">Rust&apos;s Result type integrates seamlessly with MATLAB&apos;s error handling</div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-purple-600 rounded-full mt-2 flex-shrink-0"></div>
                    <div>
                      <div className="font-semibold text-foreground">Runtime Registration</div>
                      <div className="text-sm text-muted-foreground">Functions are automatically discovered and registered at startup</div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-orange-600 rounded-full mt-2 flex-shrink-0"></div>
                    <div>
                      <div className="font-semibold text-foreground">Type Safety</div>
                      <div className="text-sm text-muted-foreground">Compile-time guarantees prevent runtime type errors</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

                 {/* Compatibility Goals */}
         <section className="mb-12">
           <div className="flex items-center mb-6">
             <Target className="h-6 w-6 mr-3 text-orange-600" />
             <h2 className="text-3xl font-bold text-foreground">
               Compatibility Goals
             </h2>
           </div>
           
           <div className="bg-gradient-to-r from-orange-50/50 to-amber-50/50 dark:from-orange-950/30 dark:to-amber-950/30 rounded-xl p-8">
             <p className="text-lg text-muted-foreground leading-relaxed mb-8">
               Our roadmap targets full compatibility with MATLAB&apos;s core function set, with plans to 
               extend support to major toolboxes. The modular architecture enables parallel development 
               of different function categories, accelerating our path to comprehensive compatibility.
             </p>
             
             <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
               <div className="text-center">
                 <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center mx-auto mb-3">
                   <Calculator className="h-6 w-6 text-blue-600" />
                 </div>
                 <h4 className="font-semibold text-foreground mb-2">Core MATLAB Functions</h4>
                 <p className="text-sm text-muted-foreground">Essential mathematical, statistical, and array manipulation functions that form the foundation of MATLAB compatibility</p>
               </div>
               
               <div className="text-center">
                 <div className="w-12 h-12 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center mx-auto mb-3">
                   <TrendingUp className="h-6 w-6 text-green-600" />
                 </div>
                 <h4 className="font-semibold text-foreground mb-2">Signal Processing Toolbox</h4>
                 <p className="text-sm text-muted-foreground">Advanced signal analysis, filtering, and transformation functions for engineering and scientific applications</p>
               </div>
               
               <div className="text-center">
                 <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center mx-auto mb-3">
                   <Database className="h-6 w-6 text-purple-600" />
                 </div>
                 <h4 className="font-semibold text-foreground mb-2">Statistics & Machine Learning</h4>
                 <p className="text-sm text-muted-foreground">Comprehensive statistical analysis and machine learning algorithms for data science workflows</p>
               </div>
             </div>

             <div className="border-t border-orange-200 dark:border-orange-800 pt-6">
               <h4 className="text-lg font-semibold text-foreground mb-4">Additional Toolboxes on the Roadmap</h4>
               <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 text-sm">
                 <div className="flex items-center space-x-2">
                   <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                   <span className="text-muted-foreground">Control System Toolbox</span>
                 </div>
                 <div className="flex items-center space-x-2">
                   <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                   <span className="text-muted-foreground">Image Processing Toolbox</span>
                 </div>
                 <div className="flex items-center space-x-2">
                   <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                   <span className="text-muted-foreground">Optimization Toolbox</span>
                 </div>
                 <div className="flex items-center space-x-2">
                   <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                   <span className="text-muted-foreground">Symbolic Math Toolbox</span>
                 </div>
                 <div className="flex items-center space-x-2">
                   <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                   <span className="text-muted-foreground">Curve Fitting Toolbox</span>
                 </div>
                 <div className="flex items-center space-x-2">
                   <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                   <span className="text-muted-foreground">Deep Learning Toolbox</span>
                 </div>
                 <div className="flex items-center space-x-2">
                   <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                   <span className="text-muted-foreground">Parallel Computing Toolbox</span>
                 </div>
                 <div className="flex items-center space-x-2">
                   <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                   <span className="text-muted-foreground">Communications Toolbox</span>
                 </div>
                 <div className="flex items-center space-x-2">
                   <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                   <span className="text-muted-foreground">Financial Toolbox</span>
                 </div>
               </div>
               <p className="text-xs text-muted-foreground/80 mt-4 italic">
                 Community contributions welcome for any of these toolboxes. Implementation priorities will be guided by user demand and community interest.
               </p>
             </div>
           </div>
         </section>

        {/* Contribution Opportunities */}
        <section className="mb-12">
          <Card className="bg-gradient-to-r from-green-50/50 to-blue-50/50 dark:from-green-950/30 dark:to-blue-950/30 border-border">
            <CardHeader>
              <CardTitle className="flex items-center">
                <GitPullRequest className="h-5 w-5 mr-2 text-green-600" />
                Join the Compatibility Effort
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground mb-4">
                RunMat&apos;s builtin system is designed for community contributions. Help us achieve 
                full MATLAB compatibility by implementing missing functions and expanding toolbox support.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                <div>
                  <h4 className="font-semibold text-foreground mb-2">High-Impact Areas:</h4>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Signal processing functions (FFT, filters)</li>
                    <li>• Statistical analysis (regression, distributions)</li>
                    <li>• Linear algebra expansions (sparse matrices)</li>
                    <li>• Image processing operations</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground mb-2">What We Provide:</h4>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Clear implementation patterns</li>
                    <li>• Automated testing framework</li>
                    <li>• Performance benchmarking tools</li>
                    <li>• MATLAB compatibility validation</li>
                  </ul>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row gap-3">
                <div className="text-muted-foreground">
                  Browse existing patterns and contribute on GitHub. Pull requests and discussions are welcome.
                </div>
              </div>
            </CardContent>
          </Card>
        </section>
      </div>
    </div>
  );
}