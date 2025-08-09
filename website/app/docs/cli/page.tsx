import { Metadata } from 'next';
import { readFileSync } from 'fs';
import { join } from 'path';
import { notFound } from 'next/navigation';
import { Badge } from '@/components/ui/badge';
import { MarkdownRenderer } from '@/components/MarkdownRenderer';

export const dynamic = 'force-static';

export const metadata: Metadata = {
  title: 'RunMat CLI Reference',
  description: 'Complete command-line interface for RunMat with commands, flags, env vars, and examples.',
  alternates: { canonical: '/docs/cli' },
  openGraph: {
    title: 'RunMat CLI Reference',
    description: 'Complete command-line interface for RunMat with commands, flags, env vars, and examples.',
    type: 'article',
    url: '/docs/cli',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'RunMat CLI Reference',
    description: 'Commands, flags, environment variables, and examples.',
  },
};

export default async function CliDocsPage() {
  let source = '';
  try {
    const filePath = join(process.cwd(), '..', 'docs', 'CLI.md');
    source = readFileSync(filePath, 'utf-8');
  } catch {
    notFound();
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto max-w-4xl px-4 md:px-6 py-16 md:py-24">
        <Badge variant="secondary" className="mb-4">Documentation</Badge>
        <h1 className="text-4xl lg:text-5xl font-bold mb-6 text-foreground">CLI Reference</h1>
        <p className="text-muted-foreground mb-6 text-lg">
          Commands, flags, environment variables, and examples for using RunMat across local shells,
          CI/CD pipelines, containers, and headless servers.
        </p>
        <MarkdownRenderer source={source} />
      </div>
    </div>
  );
}


