import React, { use } from 'react';

import { readFileSync, readdirSync, existsSync } from 'fs';
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

interface AuthorInfo {
  name: string;
  url?: string;
}

interface BlogPost {
  slug: string;
  frontmatter: {
    title: string;
    description: string;
    date: string;
    author?: string;
    authors?: Array<{ name: string; url?: string } | string>;
    readTime: string;
    tags: string[];
    excerpt: string;
    image?: string;
    imageAlt?: string;
    canonical?: string;
    jsonLd?: unknown;
  };
  content: string;
  authors: AuthorInfo[];
}

function normalizeAuthors(frontmatter: BlogPost['frontmatter']): AuthorInfo[] {
  const result: AuthorInfo[] = [];

  if (Array.isArray(frontmatter.authors)) {
    for (const entry of frontmatter.authors) {
      if (!entry) continue;
      if (typeof entry === 'string') {
        result.push({ name: entry });
        continue;
      }
      if (typeof entry === 'object' && 'name' in entry && typeof entry.name === 'string') {
        result.push({
          name: entry.name,
          url: typeof entry.url === 'string' ? entry.url : undefined,
        });
      }
    }
  }

  if (result.length === 0 && typeof frontmatter.author === 'string') {
    result.push({ name: frontmatter.author });
  }

  if (result.length === 0) {
    result.push({ name: 'RunMat Team' });
  }

  return result;
}

function getBlogPost(slug: string): BlogPost | null {
  try {
    const filePath = join(process.cwd(), 'content/blog', `${slug}.md`);
    const fileContent = readFileSync(filePath, 'utf-8');
    const { data: frontmatter, content } = matter(fileContent);
    
    return {
      slug,
      frontmatter: frontmatter as BlogPost['frontmatter'],
      content,
      authors: normalizeAuthors(frontmatter as BlogPost['frontmatter']),
    };
  } catch {
    return null;
  }
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const post = getBlogPost(slug);
  if (!post) return {};

  const authorNames = post.authors.map(author => author.name);
  const img = post.frontmatter.image;
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
    title: post.frontmatter.title,
    description: post.frontmatter.description,
    alternates: post.frontmatter.canonical ? { canonical: post.frontmatter.canonical } : undefined,
    openGraph: {
      title: post.frontmatter.title,
      description: post.frontmatter.description,
      type: 'article',
      publishedTime: post.frontmatter.date,
      authors: authorNames,
      images: imageUrl ? [imageUrl] : undefined,
    },
    twitter: {
      card: 'summary_large_image',
      title: post.frontmatter.title,
      description: post.frontmatter.description,
      images: imageUrl ? [imageUrl] : undefined,
    },
  };
}

// Generate static params for all blog posts
export async function generateStaticParams() {
  const contentDirectory = join(process.cwd(), 'content/blog');
  try {
    const files = readdirSync(contentDirectory);
    return files
      .filter(file => file.endsWith('.md'))
      .map(file => ({
        slug: file.replace('.md', '')
      }));
  } catch (error) {
    console.warn('Could not read blog directory:', error);
    return [];
  }
}

export default function BlogPostPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = use(params);
  const post = getBlogPost(slug);
  
  if (!post) {
    notFound();
  }

  // Format date as MM/DD/YYYY without timezone conversion
  const formatDate = (dateString: string): string => {
    const [year, month, day] = dateString.split('-');
    return `${month}/${day}/${year}`;
  };

  const fallbackJsonLd = {
    "@context": "https://schema.org",
    "@type": "Article",
    headline: post.frontmatter.title,
    description: post.frontmatter.description,
    datePublished: post.frontmatter.date,
    dateModified: post.frontmatter.date,
    author: post.authors.map(author => ({
      "@type": "Person",
      name: author.name,
      ...(author.url ? { url: author.url } : {}),
    })),
    publisher: {
      "@type": "Organization",
      name: "RunMat",
    },
    ...(post.frontmatter.canonical ? { mainEntityOfPage: post.frontmatter.canonical } : {}),
    ...(post.frontmatter.image ? { image: post.frontmatter.image } : {}),
  };

  const jsonLdString = (() => {
    try {
      const data = post.frontmatter.jsonLd || fallbackJsonLd;
      return JSON.stringify(data);
    } catch {
      return undefined;
    }
  })();

  return (
    <BlogLayout
      title={post.frontmatter.title}
      description=""
      date={formatDate(post.frontmatter.date)}
      readTime={post.frontmatter.readTime}
      authors={post.authors}
      tags={post.frontmatter.tags}
      rightAside={<HeadingsNav source={post.content} />}
    >
      {jsonLdString && (
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: jsonLdString }}
        />
      )}
      <MarkdownRenderer source={post.content} />
      
      {/* Newsletter CTA specific to blog posts */}
      <div className="mt-16">
        <NewsletterCta
          title="Enjoyed this post? Join the newsletter"
          description="Monthly updates on RunMat, Rust internals, and performance tips."
          align="center"
        />
      </div>

      <div className="mt-16 not-prose">
        <Card>
          <CardContent className="p-6 text-center">
            <h3 className="text-lg font-semibold mb-3">
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