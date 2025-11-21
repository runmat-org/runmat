import type { Metadata } from "next";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const fusionTopics = [
  {
    title: "Elementwise Chains",
    description: "RunMat collapses arithmetic and transcendental expressions into one shader with full broadcasting.",
    href: "/docs/fusion/elementwise",
  },
  {
    title: "Reductions",
    description: "sum, mean, and similar column/row reductions with omit-NaN handling and scaling rules.",
    href: "/docs/fusion/reduction",
  },
  {
    title: "Matmul Epilogues",
    description: "Keep matmul outputs on device for scale, bias, clamp, pow, and diagonal extraction epilogues.",
    href: "/docs/fusion/matmul-epilogue",
  },
  {
    title: "Centered Gram / Covariance",
    description: "Mean subtraction plus covariance / Gram construction for any tall matrix stays resident.",
    href: "/docs/fusion/centered-gram",
  },
  {
    title: "Power-Step Normalisation",
    description: "Fuse matmul plus vector normalisation stages for iterative solvers and eigensolvers.",
    href: "/docs/fusion/power-step-normalize",
  },
  {
    title: "Explained Variance",
    description: "Track diag(Q' * G * Q)-style diagnostics without leaving the GPU.",
    href: "/docs/fusion/explained-variance",
  },
  {
    title: "Image Normalisation",
    description: "Batch × H × W whitening, gain, and bias fusion for image-like tensors.",
    href: "/docs/fusion/image-normalize",
  },
];

const tocItems = [
  { id: "documents", title: "Documents in This Folder" },
  { id: "how-to-use", title: "How to Use These Docs" },
  { id: "why-groups", title: "Why These Fusion Groups Exist" },
  { id: "next-targets", title: "Next Fusion & Kernel Targets" },
];

export const metadata: Metadata = {
  title: "Fusion Guide",
  description:
    "Understand RunMat fusion groups, how workloads stay on the GPU, and where to dive deeper into each fusion topic.",
  alternates: { canonical: "/docs/fusion-guide" },
};

export default function FusionGuidePage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 md:px-0 py-16 md:py-4">
        <div className="grid lg:grid-cols-[minmax(0,1fr)_260px] gap-12">
          <div className="space-y-12">
            <header className="space-y-6">
              <div>
                <p className="text-sm uppercase tracking-wide text-primary font-semibold">Fusion Reference</p>
                <h1 className="text-4xl md:text-5xl font-bold text-foreground mt-2">Fusion Guide</h1>
              </div>
              <p className="text-lg md:text-xl text-muted-foreground leading-relaxed max-w-3xl">
                RunMat&apos;s acceleration layer recognises multiple flavours of fusible graphs and hands them to your GPU
                provider as single kernels.
              </p>
            </header>


            <section id="why-groups" className="space-y-4 scroll-mt-16">
              <p className="text-muted-foreground">
                RunMat fuses common patterns that show up across linear algebra, signal processing, imaging, and solver workloads into single GPU programs. Keeping them
                fused prevents redundant memory traffic and lets us re-use provider kernels to ship quickly.
              </p>
              <ul className="list-disc pl-6 space-y-3 marker:text-blue-500">
                <li className="text-muted-foreground">
                  <strong>Elementwise &amp; reductions:</strong> Collapse dozens of scalar operations into one dispatch and
                  prevent repeated reads/writes of the same tensor.
                </li>
                <li className="text-muted-foreground">
                  <strong>Matmul epilogues:</strong> Fusing scale, bias, and activation work avoids launching a second kernel
                  that touches the full matrix again and delivers RunMat&apos;s matmul + activation parity goals.
                </li>
                <li className="text-muted-foreground">
                  <strong>Covariance / Gram / power-step / explained-variance chains:</strong> Iterative factorizations spend
                  most of their time in repeated &quot;multiply, renormalize, measure&quot; loops. Treating each stage as a fusion kind
                  keeps eigensolvers and Krylov methods resident on the GPU.
                </li>
                <li className="text-muted-foreground">
                  <strong>Image normalisation:</strong> Imaging and sensor pipelines often start with per-frame whitening plus
                  gain/bias adjustments. Folding statistics and affine transforms into one kernel removes several launches per
                  frame.
                </li>
              </ul>
              <p className="text-muted-foreground">
                We prioritised these groups because they appear across domains, keep chatty host/device traffic off PCIe, and
                benefit greatly from fusion. We&apos;ll be adding more fusion groups in the future to cover more workloads.
              </p>
              <p className="text-muted-foreground">
                Have a new fusion flavour in mind? Open an issue or submit a pull request so we can explore it together.
              </p>
            </section>

            <section id="documents" className="space-y-4 scroll-mt-16">
              <h2 className="text-3xl font-bold text-foreground">RunMat Currently Fuses the Following Patterns</h2>

              <div className="mt-6 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
                {fusionTopics.map((topic) => (
                  <Link key={topic.href} href={topic.href} className="block h-full">
                    <Card className="group relative overflow-hidden transition-colors hover:bg-muted/50 cursor-pointer h-full">
                      <CardHeader>
                        <CardTitle>{topic.title}</CardTitle>
                        <CardDescription>{topic.description}</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="text-sm text-muted-foreground group-hover:text-foreground transition-colors">
                          Read Guide →
                        </div>
                      </CardContent>
                    </Card>
                  </Link>
                ))}
              </div>
            </section>

            <section id="how-to-use" className="space-y-4 scroll-mt-28">
              <h2 className="text-3xl font-bold text-foreground">How to Use These Docs</h2>
              <ol className="list-decimal pl-6 space-y-3 marker:text-blue-500">
                <li className="text-muted-foreground">
                  <strong>Looking for coverage:</strong> Start with the link that matches your math. Each page lists the exact
                  instruction patterns the fusion planner looks for and the operations that stay on device.
                </li>
                <li className="text-muted-foreground">
                  <strong>Investigating surprises:</strong> If a workload isn&apos;t fusing, cross-check the prerequisites
                  section (e.g. single-consumer chains for elementwise groups or constant epsilon for power steps).
                </li>
                <li className="text-muted-foreground">
                  <strong>Extending RunMat:</strong> Combine these docs with <code>docs/HOW_RUNMAT_FUSION_WORKS.md</code> to see
                  where to add new detection logic or builtin metadata.
                </li>
                <li className="text-muted-foreground">
                  <strong>Telemetry correlation:</strong> Provider telemetry reports <code>fusion_kind</code> labels. Match those
                  labels to the filenames above to understand what the GPU executed.
                </li>
              </ol>
            </section>

          </div>

          <aside className="hidden lg:block">
            <div className="sticky top-24">
              <div className="text-sm font-semibold text-foreground/90 mb-2">On this page</div>
              <ul className="text-sm space-y-2">
                {tocItems.map((item) => (
                  <li key={item.id}>
                    <a href={`#${item.id}`} className="text-muted-foreground hover:text-foreground">
                      {item.title}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}

