import { Metadata } from "next";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { Calculator, TrendingUp, Hash, ArrowUpDown, BarChart3, Zap } from "lucide-react";
import { readFileSync, readdirSync, statSync } from 'fs';
import { join } from 'path';
import matter from 'gray-matter';
import type { LucideIcon } from "lucide-react";

export const metadata: Metadata = {
  title: "RunMat Benchmarks - Performance Comparisons",
  description: "Reproducible, cross-language benchmarks comparing RunMat against common alternatives for representative workloads.",
  openGraph: {
    title: "RunMat Benchmarks - Performance Comparisons",
    description: "Reproducible, cross-language benchmarks comparing RunMat against common alternatives.",
    type: "website",
  },
};

interface Benchmark {
  slug: string;
  title: string;
  description: string;
  date: string;
  readTime: string;
  author: string;
  tags: string[];
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

function extractDescriptionFromMarkdown(content: string): string {
  const lines = content.split('\n');
  let inParagraph = false;
  let description = '';
  
  for (const line of lines) {
    const trimmed = line.trim();
    // Skip frontmatter if present
    if (trimmed === '---') continue;
    // Skip headings
    if (trimmed.startsWith('#')) {
      if (inParagraph) break;
      continue;
    }
    // Skip empty lines at start
    if (!trimmed && !inParagraph) continue;
    // Collect first paragraph
    if (trimmed) {
      inParagraph = true;
      description += (description ? ' ' : '') + trimmed;
      if (description.length > 200) {
        description = description.substring(0, 200) + '...';
        break;
      }
    } else if (inParagraph) {
      break;
    }
  }
  
  return description || 'Performance benchmark comparing RunMat against alternatives.';
}

function getBenchmarkIcon(slug: string): LucideIcon {
  const iconMap: Record<string, LucideIcon> = {
    '4k-image-processing': BarChart3,
    'monte-carlo-analysis': TrendingUp,
    'pca': Calculator,
    'batched-nlms': Hash,
    'batched-iir-smoothing': ArrowUpDown,
  };
  return iconMap[slug] || Zap;
}

function getAllBenchmarks(): Benchmark[] {
  try {
    const benchmarksDir = join(process.cwd(), '..', 'benchmarks');
    const entries = readdirSync(benchmarksDir, { withFileTypes: true });
    
    const benchmarks = entries
      .filter(entry => entry.isDirectory() && entry.name !== '.harness' && entry.name !== 'wgpu_profile')
      .map(entry => {
        const slug = entry.name;
        const readmePath = join(benchmarksDir, slug, 'README.md');
        
        try {
          const fileContent = readFileSync(readmePath, 'utf-8');
          const { data: frontmatter, content } = matter(fileContent);
          
          // Extract title from frontmatter or first heading
          const title = frontmatter.title || extractTitleFromMarkdown(content);
          
          // Extract description from frontmatter or first paragraph
          const description = frontmatter.description || frontmatter.excerpt || extractDescriptionFromMarkdown(content);
          
          // Get file modification date as fallback
          const stats = statSync(readmePath);
          const defaultDate = stats.mtime.toISOString();
          
          return {
            slug,
            title,
            description,
            date: frontmatter.date || defaultDate,
            readTime: frontmatter.readTime || '5 min read',
            author: frontmatter.author || 'RunMat Team',
            tags: frontmatter.tags || []
          };
        } catch (error) {
          console.error(`Error reading benchmark ${slug}:`, error);
          return null;
        }
      })
      .filter((benchmark): benchmark is Benchmark => benchmark !== null);
    
    // Sort by date (newest first) or alphabetically if no date
    return benchmarks.sort((a, b) => {
      const dateA = new Date(a.date).getTime();
      const dateB = new Date(b.date).getTime();
      if (dateA !== dateB) {
        return dateB - dateA;
      }
      return a.title.localeCompare(b.title);
    });
  } catch (error) {
    console.error('Error reading benchmarks:', error);
    return [];
  }
}

export default function BenchmarksPage() {
  const benchmarks = getAllBenchmarks();

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-16 md:px-6 md:py-24">
        <div className="mx-auto max-w-[58rem] text-center">
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl">
            RunMat Benchmarks
          </h1>
          <p className="mt-6 text-base text-muted-foreground sm:text-lg">
            Reproducible, cross-language benchmarks comparing RunMat against common alternatives for representative workloads
          </p>
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {benchmarks.map((benchmark) => {
            const Icon = getBenchmarkIcon(benchmark.slug);
            return (
              <Link key={benchmark.slug} href={`/benchmarks/${benchmark.slug}`} className="block">
                <Card className="group overflow-hidden transition-all hover:shadow-lg cursor-pointer h-full flex flex-col">
                  <CardContent className="p-6 flex flex-col h-full">
                    {/* Icon and Title */}
                    <div className="flex items-start gap-3 mb-4">
                      <Icon className="h-6 w-6 text-foreground flex-shrink-0 mt-1" />
                      <h3 className="text-2xl font-semibold leading-tight sm:text-3xl flex-1">
                        {benchmark.title}
                      </h3>
                    </div>
                    
                    {/* Purple/Blue Gradient Area */}
                    <div className="w-full h-48 rounded-lg mb-4 bg-gradient-to-br from-purple-500 via-purple-600 to-blue-600 group-hover:from-purple-600 group-hover:via-purple-700 group-hover:to-blue-700 transition-all" />
                    
                    {/* Description */}
                    <p className="text-sm text-muted-foreground mb-4 flex-grow">
                      {benchmark.description}
                    </p>
                    
                    {/* View Benchmark Link */}
                    <div className="flex items-center text-sm text-foreground group-hover:text-primary transition-colors">
                      View Benchmark
                      <svg 
                        className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" 
                        fill="none" 
                        stroke="currentColor" 
                        viewBox="0 0 24 24"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </div>
                  </CardContent>
                </Card>
              </Link>
            );
          })}
        </div>

        {/* Call to Action */}
        <div className="mt-16 text-center">
          <Card className="inline-block">
            <CardContent className="p-6">
              <h3 className="text-2xl font-semibold mb-2 sm:text-3xl">
                Reproduce the Benchmarks
              </h3>
              <p className="text-muted-foreground mb-4">
                See the benchmarks directory in the RunMat repo for full source code and instructions
              </p>
              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <Button variant="outline" size="sm" asChild>
                  <Link href="https://github.com/runmat-org/runmat/tree/main/benchmarks" target="_blank">
                    View on GitHub
                  </Link>
                </Button>
                <Button variant="outline" size="sm" asChild>
                  <Link href="/download">
                    Download RunMat
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

