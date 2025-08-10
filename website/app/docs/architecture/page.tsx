import { Metadata } from 'next';
import { readFileSync } from 'fs';
import { join } from 'path';
import { notFound } from 'next/navigation';
import { Badge } from '@/components/ui/badge';
import { MarkdownRenderer } from '@/components/MarkdownRenderer';

export const dynamic = 'force-static';

export const metadata: Metadata = {
  title: 'RunMat Architecture',
  description: "Deep dive into RunMat's multi-crate architecture and V8-inspired execution model.",
  alternates: { canonical: '/docs/architecture' },
  openGraph: {
    title: 'RunMat Architecture',
    description: "Deep dive into RunMat's multi-crate architecture and V8-inspired execution model.",
    type: 'article',
    url: '/docs/architecture',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'RunMat Architecture',
    description: "Deep dive into RunMat's multi-crate architecture and V8-inspired execution model.",
  },
};

export default async function ArchitectureDocsPage() {
  let source = '';
  try {
    const filePath = join(process.cwd(), '..', 'docs', 'ARCHITECTURE.md');
    source = readFileSync(filePath, 'utf-8');
  } catch {
    notFound();
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto max-w-4xl px-4 md:px-6 py-16 md:py-24">
        <Badge variant="secondary" className="mb-4">Documentation</Badge>
        <h1 className="text-4xl lg:text-5xl font-bold mb-6 text-foreground">Architecture</h1>
        <p className="text-muted-foreground mb-6 text-lg">
          Understand RunMat&apos;s high-level design: tiered execution (Ignition interpreter ➝ Turbine JIT),
          generational GC, BLAS/LAPACK runtime, kernel, and plotting subsystems.
        </p>
        <MarkdownRenderer source={source} />
      </div>
    </div>
  );
}


