import { Metadata } from "next";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { readFileSync, readdirSync, statSync, existsSync } from 'fs';
import { join, extname } from 'path';
import matter from 'gray-matter';

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
  summary: string;
  imageUrl?: string;
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

function extractFirstImageUrl(content: string): string | undefined {
  const imageRegex = /!\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)/;
  const match = imageRegex.exec(content);
  return match ? match[1] : undefined;
}

function getMimeTypeFromExtension(extension: string): string | undefined {
  switch (extension.toLowerCase()) {
    case '.png':
      return 'image/png';
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg';
    case '.svg':
      return 'image/svg+xml';
    case '.webp':
      return 'image/webp';
    default:
      return undefined;
  }
}

function getAllBenchmarks(): Benchmark[] {
  try {
    const benchmarksDir = join(process.cwd(), '..', 'benchmarks');
    const entries = readdirSync(benchmarksDir, { withFileTypes: true });
    
    const benchmarks = entries
      .filter(entry => entry.isDirectory() && entry.name !== '.harness' && entry.name !== 'wgpu_profile')
      .map((entry): Benchmark | null => {
        const slug = entry.name;
        const readmePath = join(benchmarksDir, slug, 'README.md');
        
        try {
          const fileContent = readFileSync(readmePath, 'utf-8');
          const { data: frontmatter, content } = matter(fileContent);
          
          // Extract title from frontmatter or first heading
          const title = frontmatter.title || extractTitleFromMarkdown(content);
          
          // Extract description from frontmatter or first paragraph
          const rawDescription = frontmatter.description || frontmatter.excerpt || extractFirstParagraph(content);
          const description = rawDescription || 'Performance benchmark comparing RunMat against alternatives.';
          const summary = truncateText(description);

          const frontmatterImage = typeof frontmatter.image === 'string' ? frontmatter.image : undefined;
          const markdownImage = extractFirstImageUrl(content);
          const resolvedImagePath = frontmatterImage || markdownImage;

          let imageUrl: string | undefined;
          if (resolvedImagePath) {
            if (resolvedImagePath.startsWith('http://') || resolvedImagePath.startsWith('https://')) {
              imageUrl = resolvedImagePath;
            } else {
              const sanitizedPath = resolvedImagePath.replace(/^\.?\//, '');
              const absolutePath = join(benchmarksDir, slug, sanitizedPath);
              if (existsSync(absolutePath)) {
                const mimeType = getMimeTypeFromExtension(extname(absolutePath));
                if (mimeType) {
                  const fileBuffer = readFileSync(absolutePath);
                  const base64 = fileBuffer.toString('base64');
                  imageUrl = `data:${mimeType};base64,${base64}`;
                }
              }
            }
          }
          
          // Get file modification date as fallback
          const stats = statSync(readmePath);
          const defaultDate = stats.mtime.toISOString();
          
          return {
            slug,
            title,
            description,
            summary,
            imageUrl,
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
          {benchmarks.map((benchmark) => (
            <Link key={benchmark.slug} href={`/benchmarks/${benchmark.slug}`} className="block">
              <Card className="group overflow-hidden transition-all hover:shadow-lg cursor-pointer h-full flex flex-col">
                <CardContent className="p-6 flex flex-col h-full">
                  {/* Thumbnail */}
                  <div
                    className={`w-full h-48 rounded-lg mb-4 transition-all ${
                      benchmark.imageUrl
                        ? 'bg-muted/10'
                        : 'bg-gradient-to-br from-purple-500 via-purple-600 to-blue-600 group-hover:from-purple-600 group-hover:via-purple-700 group-hover:to-blue-700'
                    }`}
                    style={
                      benchmark.imageUrl
                        ? {
                            backgroundImage: `url(${benchmark.imageUrl})`,
                            backgroundSize: 'contain',
                            backgroundPosition: 'center',
                            backgroundRepeat: 'no-repeat',
                          }
                        : undefined
                    }
                  />

                  {/* Title */}
                  <h3 className="text-2xl font-semibold leading-tight sm:text-3xl mb-4">
                    {benchmark.title}
                  </h3>
                    
                  {/* Description */}
                  <p className="text-sm text-muted-foreground mb-4 flex-grow line-clamp-3">
                    {benchmark.summary}
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
          ))}
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

