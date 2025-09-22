import React from 'react';

import { readFileSync, readdirSync } from 'fs';
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

interface BlogPost {
  slug: string;
  frontmatter: {
    title: string;
    description: string;
    date: string;
    author: string;
    readTime: string;
    tags: string[];
    excerpt: string;
  };
  content: string;
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
    };
  } catch {
    return null;
  }
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const post = getBlogPost(slug);
  if (!post) return {};

  return {
    title: post.frontmatter.title,
    description: post.frontmatter.description,
    openGraph: {
      title: post.frontmatter.title,
      description: post.frontmatter.description,
      type: 'article',
      publishedTime: post.frontmatter.date,
      authors: [post.frontmatter.author],
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

export default async function BlogPostPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const post = getBlogPost(slug);
  
  if (!post) {
    notFound();
  }

  return (
    <BlogLayout
      title={post.frontmatter.title}
      description={post.frontmatter.excerpt}
      date={new Date(post.frontmatter.date).toLocaleDateString()}
      readTime={post.frontmatter.readTime}
      author={post.frontmatter.author}
      tags={post.frontmatter.tags}
      rightAside={<HeadingsNav source={post.content} />}
    >
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
              <Button asChild>
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