import { Metadata } from 'next';
import { readFileSync } from 'fs';
import { join } from 'path';
import { notFound } from 'next/navigation';
import { Badge } from '@/components/ui/badge';
import { MarkdownRenderer } from '@/components/MarkdownRenderer';

export const dynamic = 'force-static';

export const metadata: Metadata = {
  title: 'RunMat Language Coverage',
  description: 'Up-to-date coverage of MATLAB language features supported by RunMat.',
  alternates: { canonical: '/docs/language-coverage' },
  openGraph: {
    title: 'RunMat Language Coverage',
    description: 'Up-to-date coverage of MATLAB language features supported by RunMat.',
    type: 'article',
    url: '/docs/language-coverage',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'RunMat Language Coverage',
    description: 'Coverage of MATLAB language features supported by RunMat.',
  },
};

export default async function LanguageCoveragePage() {
  let source = '';
  try {
    const filePath = join(process.cwd(), '..', 'docs', 'LANGUAGE_COVERAGE.md');
    source = readFileSync(filePath, 'utf-8');
  } catch {
    notFound();
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto max-w-4xl px-4 md:px-6 py-16 md:py-24">
        <Badge variant="secondary" className="mb-4">Documentation</Badge>
        <h1 className="text-4xl lg:text-5xl font-bold mb-6 text-foreground">Language Coverage</h1>
        <p className="text-muted-foreground mb-6 text-lg">
          Track MATLAB syntax and feature support in RunMat. This page is generated from the
          repository's authoritative coverage document and updated regularly.
        </p>
        <MarkdownRenderer source={source} />
      </div>
    </div>
  );
}


