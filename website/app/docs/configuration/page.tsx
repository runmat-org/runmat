import { Metadata } from 'next';
import { readFileSync } from 'fs';
import { join } from 'path';
import { notFound } from 'next/navigation';
import { Badge } from '@/components/ui/badge';
import { MarkdownRenderer } from '@/components/MarkdownRenderer';

export const dynamic = 'force-static';

export const metadata: Metadata = {
  title: 'RunMat Configuration',
  description: 'Configure RunMat with YAML/JSON/TOML files, environment overrides, and clear precedence.',
  alternates: { canonical: '/docs/configuration' },
  openGraph: {
    title: 'RunMat Configuration',
    description: 'Configure RunMat with YAML/JSON/TOML files, environment overrides, and clear precedence.',
    type: 'article',
    url: '/docs/configuration',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'RunMat Configuration',
    description: 'All config options with examples and precedence.',
  },
};

export default async function ConfigurationDocsPage() {
  let source = '';
  try {
    const filePath = join(process.cwd(), '..', 'docs', 'CONFIG.md');
    source = readFileSync(filePath, 'utf-8');
  } catch {
    notFound();
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto max-w-4xl px-4 md:px-6 py-16 md:py-24">
        <Badge variant="secondary" className="mb-4">Documentation</Badge>
        <h1 className="text-4xl lg:text-5xl font-bold mb-6 text-foreground">Configuration</h1>
        <p className="text-muted-foreground mb-6 text-lg">
          Configure RunMat using YAML/JSON/TOML files, environment variables, and CLI overrides.
          Learn discovery paths, precedence rules, and real-world examples.
        </p>
        <MarkdownRenderer source={source} />
      </div>
    </div>
  );
}


