import { readFileSync } from 'fs';
import { join } from 'path';
import matter from 'gray-matter';
import { MDXRemote } from 'next-mdx-remote/rsc';
import { notFound } from 'next/navigation';
import React from 'react';

import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { BlogLayout } from '@/components/BlogLayout';
import { CodeBlock } from '@/components/CodeBlock';
import { MermaidDiagram } from '@/components/MermaidDiagram';
import Link from 'next/link';

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

export default async function BlogPostPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const post = getBlogPost(slug);
  
  if (!post) {
    notFound();
  }

  const components = {
    // Headings with proper spacing and styling
    h1: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <h1 className="text-4xl font-bold mt-12 mb-6 text-foreground first:mt-0" {...props}>{children}</h1>
    ),
    h2: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <h2 className="text-3xl font-semibold mt-10 mb-5 text-foreground" {...props}>{children}</h2>
    ),
    h3: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <h3 className="text-2xl font-semibold mt-8 mb-4 text-foreground" {...props}>{children}</h3>
    ),
    h4: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <h4 className="text-xl font-semibold mt-6 mb-3 text-foreground" {...props}>{children}</h4>
    ),
    h5: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <h5 className="text-lg font-semibold mt-5 mb-2 text-foreground" {...props}>{children}</h5>
    ),
    h6: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <h6 className="text-base font-semibold mt-4 mb-2 text-foreground" {...props}>{children}</h6>
    ),

    // Paragraphs with proper spacing
    p: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => {
      // Check if children contains any block elements that shouldn't be in a p tag
      const hasBlockElements = React.Children.toArray(children).some(child => {
        if (React.isValidElement(child)) {
          const type = child.type;
          // Check for block elements that shouldn't be in paragraphs
          return typeof type === 'string' && ['pre', 'div', 'blockquote', 'ul', 'ol', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(type);
        }
        return false;
      });
      
      if (hasBlockElements) {
        // If it contains block elements, render as a div instead
        return <div className="my-6" {...props}>{children}</div>;
      }
      
      // Otherwise render as a normal paragraph
      return <p className="my-6 text-muted-foreground leading-relaxed" {...props}>{children}</p>;
    },

    // Lists with proper styling and spacing
    ul: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <ul className="my-6 ml-6 space-y-2 list-disc marker:text-blue-500" {...props}>{children}</ul>
    ),
    ol: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <ol className="my-6 ml-6 space-y-2 list-decimal marker:text-blue-500" {...props}>{children}</ol>
    ),
    li: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <li className="text-muted-foreground leading-relaxed pl-2" {...props}>{children}</li>
    ),

    // Blockquotes
    blockquote: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <blockquote className="my-6 pl-6 border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/20 py-4 pr-4 rounded-r-lg" {...props}>
        <div className="text-muted-foreground italic">{children}</div>
      </blockquote>
    ),

    // Horizontal rule
    hr: ({ ...props }: { [key: string]: unknown }) => (
      <hr className="my-12 border-t border-border" {...props} />
    ),

    // Links with proper styling
    a: ({ children, href, ...props }: { children: React.ReactNode; href?: string; [key: string]: unknown }) => (
      <a 
        href={href} 
        className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 underline underline-offset-4 transition-colors" 
        {...props}
      >
        {children}
      </a>
    ),

    // Strong and emphasis
    strong: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <strong className="font-semibold text-foreground" {...props}>{children}</strong>
    ),
    em: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <em className="italic" {...props}>{children}</em>
    ),

    // Tables
    table: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <div className="my-8 overflow-x-auto">
        <table className="w-full border-collapse border border-border rounded-lg" {...props}>{children}</table>
      </div>
    ),
    thead: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <thead className="bg-muted" {...props}>{children}</thead>
    ),
    tbody: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <tbody {...props}>{children}</tbody>
    ),
    tr: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <tr className="border-b border-border last:border-b-0" {...props}>{children}</tr>
    ),
    th: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <th className="border border-border px-4 py-3 text-left font-semibold text-foreground" {...props}>{children}</th>
    ),
    td: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
      <td className="border border-border px-4 py-3 text-muted-foreground" {...props}>{children}</td>
    ),

    // Code blocks and inline code
    pre: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => {
      // Extract the code content from the pre tag
      if (React.isValidElement(children) && children.type === 'code') {
        const codeProps = children.props as { className?: string; children: string };
        const language = codeProps.className?.replace(/language-/, '') || 'text';
        return (
          <div className="my-8">
            <CodeBlock language={language} {...props}>{codeProps.children}</CodeBlock>
          </div>
        );
      }
      // Fallback for non-code pre elements
      return <div className="my-8 bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-x-auto font-mono text-sm whitespace-pre">{children}</div>;
    },
    code: ({ children, className, ...props }: { children: string; className?: string; [key: string]: unknown }) => {
      // Only handle inline code here - block code is handled by pre
      if (!className) {
        return <code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded text-sm font-mono text-foreground" {...props}>{children}</code>;
      }
      // For block code, just return the basic element (pre will handle the styling)
      return <code className={className} {...props}>{children}</code>;
    },

    // Center and style Mermaid diagrams
    MermaidDiagram: ({ chart, ...props }: { chart: string; [key: string]: unknown }) => (
      <div className="my-12 flex justify-center">
        <div className="max-w-full overflow-x-auto">
          <MermaidDiagram chart={chart} {...props} />
        </div>
      </div>
    ),

    // Images with responsive styling
    img: ({ src, alt, ...props }: { src?: string; alt?: string; [key: string]: unknown }) => (
      <div className="my-8 text-center">
        <img 
          src={src} 
          alt={alt} 
          className="max-w-full h-auto rounded-lg shadow-lg mx-auto" 
          {...props} 
        />
        {alt && <p className="mt-2 text-sm text-muted-foreground italic">{alt}</p>}
      </div>
    ),
  };

  return (
    <BlogLayout
      title={post.frontmatter.title}
      description={post.frontmatter.excerpt}
      date={new Date(post.frontmatter.date).toLocaleDateString()}
      readTime={post.frontmatter.readTime}
      author={post.frontmatter.author}
      tags={post.frontmatter.tags}
    >
      <MDXRemote source={post.content} components={components} />
      
      <div className="mt-16 not-prose">
        <Card>
          <CardContent className="p-6 text-center">
            <h3 className="text-lg font-semibold mb-3">
              Ready to try RustMat?
            </h3>
            <p className="text-muted-foreground mb-4">
              Get started with the modern MATLAB runtime today.
            </p>
            <div className="flex flex-col sm:flex-row gap-3 justify-center">
              <Button asChild>
                <Link href="/download">Download RustMat</Link>
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