import React from 'react';

import { readFileSync, readdirSync, existsSync, statSync } from 'fs';
import { join } from 'path';
import matter from 'gray-matter';
import { MarkdownRenderer } from '@/components/MarkdownRenderer';
import { HeadingsNav } from '@/components/HeadingsNav';
import { notFound } from 'next/navigation';
import Link from 'next/link';

import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { BlogLayout } from '@/components/BlogLayout';
import NewsletterCta from '@/components/NewsletterCta';

interface Benchmark {
  slug: string;
  frontmatter: {
    title: string;
    description: string;
    date: string;
    authors?: Array<{ name: string; url?: string } | string>;
    readTime: string;
    tags: string[];
    excerpt: string;
    image?: string;
    imageAlt?: string;
    canonical?: string;
  };
  content: string;
  introParagraph: string;
  authors: { name: string; url?: string }[];
}

function normalizeAuthors(frontmatter: Benchmark['frontmatter']): { name: string; url?: string }[] {
  const out: { name: string; url?: string }[] = [];
  if (Array.isArray(frontmatter.authors)) {
    for (const entry of frontmatter.authors) {
      if (!entry) continue;
      if (typeof entry === 'string') {
        out.push({ name: entry });
      } else if (
        typeof entry === 'object' &&
        'name' in entry &&
        typeof (entry as { name?: unknown }).name === 'string'
      ) {
        const url = (entry as { url?: unknown }).url;
        out.push({ name: (entry as { name: string }).name, url: typeof url === 'string' ? url : undefined });
      }
    }
  }
  if (out.length === 0) {
    out.push({ name: 'RunMat Team' });
  }
  return out;
}

function extractTitleFromMarkdown(content: string): string {
  const lines = content.split('\n');
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('# ')) {
      return trimmed.substring(2).trim();
    }
  }
  return 'Untitled Benchmark';
}

function extractFirstParagraph(content: string): string {
  const lines = content.split('\n');
  let inParagraph = false;
  const paragraphLines: string[] = [];
  
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('#')) {
      if (inParagraph) break;
      continue;
    }
    if (!trimmed && !inParagraph) continue;
    if (trimmed) {
      inParagraph = true;
      paragraphLines.push(trimmed);
    } else if (inParagraph) {
      break;
    }
  }
  
  const paragraph = paragraphLines.join(' ');
  return paragraph || 'Performance benchmark comparing RunMat against alternatives.';
}

function truncateText(text: string, limit: number = 200): string {
  if (text.length <= limit) {
    return text;
  }
  return text.substring(0, limit).trimEnd() + '...';
}

function stripFirstHeadingAndParagraph(content: string): string {
  const lines = content.split('\n');
  let skipFirstHeading = true;
  let skipFirstParagraph = true;
  let inFirstParagraph = false;
  const result: string[] = [];
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();
    
    // Skip the first H1 heading
    if (skipFirstHeading && trimmed.startsWith('# ')) {
      skipFirstHeading = false;
      continue;
    }
    
    // Skip the first paragraph (non-empty lines after the heading until we hit a blank line or another heading)
    if (skipFirstParagraph) {
      if (!trimmed) {
        // Empty line - if we were in the first paragraph, we're done with it
        if (inFirstParagraph) {
          skipFirstParagraph = false;
          // Keep the empty line if it's not the first one
          if (result.length > 0 || i < lines.length - 1) {
            result.push(line);
          }
        } else {
          // Keep empty lines before the first paragraph
          result.push(line);
        }
        inFirstParagraph = false;
        continue;
      } else if (trimmed.startsWith('#')) {
        // Another heading - we're past the first paragraph
        skipFirstParagraph = false;
        result.push(line);
        continue;
      } else {
        // Content line - this is part of the first paragraph
        inFirstParagraph = true;
        continue;
      }
    }
    
    // After skipping the first heading and paragraph, keep everything else
    result.push(line);
  }
  
  return result.join('\n').trim();
}

function getBenchmark(slug: string): Benchmark | null {
  try {
    const filePath = join(process.cwd(), '..', 'benchmarks', slug, 'README.md');
    const fileContent = readFileSync(filePath, 'utf-8');
    const { data: frontmatter, content } = matter(fileContent);
    const firstParagraph = extractFirstParagraph(content);
    const introParagraph = (frontmatter as Record<string, unknown>)?.intro as string | undefined || firstParagraph;
    const safeIntroParagraph = introParagraph || 'Performance benchmark comparing RunMat against alternatives.';
    const excerpt = frontmatter.excerpt || frontmatter.description || truncateText(safeIntroParagraph);
    const description = frontmatter.description || excerpt;
    
    // Extract title from frontmatter or first heading
    const title = frontmatter.title || extractTitleFromMarkdown(content);
    
    // Strip the first H1 and first paragraph from content since BlogLayout displays them
    const strippedContent = stripFirstHeadingAndParagraph(content);
    
    // Get file modification date as fallback
    const stats = statSync(filePath);
    const defaultDate = stats.mtime.toISOString();
    const authors = normalizeAuthors(frontmatter as Benchmark['frontmatter']);
    
    return {
      slug,
      frontmatter: {
        title,
        description,
        date: frontmatter.date || defaultDate,
        authors: (frontmatter as Benchmark['frontmatter']).authors,
        readTime: frontmatter.readTime || '5 min read',
        tags: frontmatter.tags || [],
        excerpt,
        image: frontmatter.image,
        imageAlt: frontmatter.imageAlt,
        canonical: frontmatter.canonical,
      },
      content: strippedContent,
      introParagraph: safeIntroParagraph,
      authors,
    };
  } catch {
    return null;
  }
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const benchmark = getBenchmark(slug);
  if (!benchmark) return {};

  const authorNames = benchmark.authors.map(a => a.name);
  const img = benchmark.frontmatter.image;
  let imageUrl: string | undefined = undefined;
  if (img && typeof img === 'string') {
    if (img.startsWith('http://') || img.startsWith('https://')) {
      imageUrl = img;
    } else if (img.startsWith('/')) {
      const localPath = join(process.cwd(), 'public', img.replace(/^\//, ''));
      if (existsSync(localPath)) {
        imageUrl = img;
      }
    }
  }

  return {
    title: benchmark.frontmatter.title,
    description: benchmark.frontmatter.description,
    alternates: benchmark.frontmatter.canonical ? { canonical: benchmark.frontmatter.canonical } : undefined,
    openGraph: {
      title: benchmark.frontmatter.title,
      description: benchmark.frontmatter.description,
      type: 'article',
      publishedTime: benchmark.frontmatter.date,
      authors: authorNames,
      images: imageUrl ? [imageUrl] : undefined,
    },
    twitter: {
      card: 'summary_large_image',
      title: benchmark.frontmatter.title,
      description: benchmark.frontmatter.description,
      images: imageUrl ? [imageUrl] : undefined,
    },
  };
}

// Generate static params for all benchmarks
export async function generateStaticParams() {
  const benchmarksDirectory = join(process.cwd(), '..', 'benchmarks');
  try {
    const entries = readdirSync(benchmarksDirectory, { withFileTypes: true });
    return entries
      .filter(entry => entry.isDirectory() && entry.name !== '.harness' && entry.name !== 'wgpu_profile')
      .map(entry => ({
        slug: entry.name
      }));
  } catch (error) {
    console.warn('Could not read benchmarks directory:', error);
    return [];
  }
}

export default async function BenchmarkPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const benchmark = getBenchmark(slug);
  
  if (!benchmark) {
    notFound();
  }

  return (
    <BlogLayout
      title={benchmark.frontmatter.title}
      description={benchmark.introParagraph}
      descriptionPlacement="afterMeta"
      date={new Date(benchmark.frontmatter.date).toLocaleDateString()}
      readTime={benchmark.frontmatter.readTime}
      authors={benchmark.authors}
      tags={benchmark.frontmatter.tags}
      rightAside={<HeadingsNav source={benchmark.content} />}
      backLink={{ href: '/benchmarks', text: 'Back to Benchmarks' }}
    >
      <MarkdownRenderer source={benchmark.content} />
      
      {/* Newsletter CTA specific to benchmarks */}
      <div className="mt-16">
        <NewsletterCta
          title="Enjoyed this benchmark? Join the newsletter"
          description="Monthly updates on RunMat, Rust internals, and performance tips."
          align="center"
        />
      </div>

      <div className="mt-16 not-prose">
        <Card>
          <CardContent className="p-6 text-center">
            <h3 className="text-2xl font-semibold mb-3 sm:text-3xl">
              Ready to try RunMat?
            </h3>
            <p className="text-muted-foreground mb-4">
              Get started with the modern MATLAB runtime today.
            </p>
            <div className="flex flex-col sm:flex-row gap-3 justify-center">
              <Button variant="outline" asChild>
                <Link href="/download">Download RunMat</Link>
              </Button>
              <Button variant="outline" asChild>
                <Link href="/docs/getting-started">Get Started</Link>
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </BlogLayout>
  );
}



