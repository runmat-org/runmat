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

type JsonLdPrimitive = string | number | boolean | null;
type JsonLdValue = JsonLdPrimitive | JsonLdObject | JsonLdValue[];
export type JsonLdObject = {
  "@context"?: string;
  "@type"?: string | string[];
  "@id"?: string;
  [key: string]: JsonLdValue | undefined;
};
type JsonLd = JsonLdObject | { "@graph": JsonLdObject[] };

interface BlogPost {
  slug: string;
  frontmatter: {
    title: string;
    description: string;
    date: string;
    slug?: string;
    author?: string;
    authors?: Array<{ name: string; url?: string } | string>;
    readTime: string;
    tags: string[];
    excerpt: string;
    image?: string;
    imageAlt?: string;
    canonical?: string;
    keywords?: string | string[];
    ogType?: string;
    ogTitle?: string;
    ogDescription?: string;
    twitterCard?: string;
    twitterTitle?: string;
    twitterDescription?: string;
    jsonLd?: JsonLd;
  };
  content: string;
  authors: AuthorInfo[];
}

const FRONTMATTER_KEYS = new Set([
  'title',
  'description',
  'date',
  'slug',
  'author',
  'authors',
  'readTime',
  'tags',
  'keywords',
  'excerpt',
  'image',
  'imageAlt',
  'canonical',
  'ogType',
  'ogTitle',
  'ogDescription',
  'twitterCard',
  'twitterTitle',
  'twitterDescription',
  'jsonLd',
]);

function isJsonLdValue(value: unknown): value is JsonLdValue {
  if (value === null) return true;
  const t = typeof value;
  if (t === 'string' || t === 'number' || t === 'boolean') return true;
  if (Array.isArray(value)) return value.every(isJsonLdValue);
  return isJsonLdObject(value);
}

function isJsonLdObject(value: unknown): value is JsonLdObject {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  return Object.values(value).every(isJsonLdValue);
}

function validateJsonLd(jsonLd: unknown, slug: string): JsonLd | undefined {
  if (jsonLd === undefined) return undefined;
  if (isJsonLdObject(jsonLd)) return jsonLd;
  if (
    typeof jsonLd === 'object' &&
    jsonLd !== null &&
    !Array.isArray(jsonLd) &&
    '@graph' in jsonLd &&
    Array.isArray((jsonLd as Record<string, unknown>)['@graph']) &&
    ((jsonLd as Record<string, unknown>)['@graph'] as unknown[]).every(isJsonLdObject)
  ) {
    return jsonLd as JsonLd;
  }
  throw new Error(
    `Invalid jsonLd in frontmatter for slug "${slug}". jsonLd must be an object or an object with @graph containing objects.`
  );
}

function assertString(val: unknown, field: string, slug: string): string {
  if (typeof val !== 'string' || val.trim() === '') {
    throw new Error(`Frontmatter field "${field}" must be a non-empty string in slug "${slug}".`);
  }
  return val;
}

function assertStringArray(val: unknown, field: string, slug: string): string[] {
  if (!Array.isArray(val) || val.some(v => typeof v !== 'string')) {
    throw new Error(`Frontmatter field "${field}" must be an array of strings in slug "${slug}".`);
  }
  return val as string[];
}

function validateFrontmatter(raw: Record<string, unknown>, slug: string): BlogPost['frontmatter'] {
  for (const key of Object.keys(raw)) {
    if (!FRONTMATTER_KEYS.has(key)) {
      throw new Error(`Unexpected frontmatter key "${key}" in slug "${slug}". Allowed keys: ${Array.from(FRONTMATTER_KEYS).join(', ')}`);
    }
  }

  const title = assertString(raw.title, 'title', slug);
  const description = assertString(raw.description, 'description', slug);
  const date = assertString(raw.date, 'date', slug);
  const readTime = assertString(raw.readTime, 'readTime', slug);
  const excerpt = assertString(raw.excerpt, 'excerpt', slug);
  const tags = assertStringArray(raw.tags, 'tags', slug);
  const fmSlug = raw.slug === undefined ? undefined : assertString(raw.slug, 'slug', slug);
  const keywords =
    raw.keywords === undefined
      ? undefined
      : typeof raw.keywords === 'string'
        ? raw.keywords
        : assertStringArray(raw.keywords, 'keywords', slug);
  const ogType = raw.ogType === undefined ? undefined : assertString(raw.ogType, 'ogType', slug);
  const ogTitle = raw.ogTitle === undefined ? undefined : assertString(raw.ogTitle, 'ogTitle', slug);
  const ogDescription = raw.ogDescription === undefined ? undefined : assertString(raw.ogDescription, 'ogDescription', slug);
  const twitterCard = raw.twitterCard === undefined ? undefined : assertString(raw.twitterCard, 'twitterCard', slug);
  const twitterTitle = raw.twitterTitle === undefined ? undefined : assertString(raw.twitterTitle, 'twitterTitle', slug);
  const twitterDescription =
    raw.twitterDescription === undefined ? undefined : assertString(raw.twitterDescription, 'twitterDescription', slug);

  const image = raw.image === undefined ? undefined : assertString(raw.image, 'image', slug);
  const imageAlt = raw.imageAlt === undefined ? undefined : assertString(raw.imageAlt, 'imageAlt', slug);
  const canonical = raw.canonical === undefined ? undefined : assertString(raw.canonical, 'canonical', slug);

  const author = raw.author === undefined ? undefined : assertString(raw.author, 'author', slug);
  const authors = raw.authors;
  if (authors !== undefined) {
    if (
      !Array.isArray(authors) ||
      authors.some(
        entry =>
          !entry ||
          (typeof entry !== 'string' &&
            !(
              typeof entry === 'object' &&
              'name' in entry &&
              typeof (entry as { name?: unknown }).name === 'string' &&
              ((entry as { url?: unknown }).url === undefined || typeof (entry as { url?: unknown }).url === 'string')
            ))
      )
    ) {
      throw new Error(`Frontmatter field "authors" must be an array of strings or { name, url? } objects in slug "${slug}".`);
    }
  }

  const jsonLd = validateJsonLd(raw.jsonLd, slug);

  return {
    title,
    description,
    date,
    author,
    slug: fmSlug,
    authors: authors as BlogPost['frontmatter']['authors'],
    readTime,
    tags,
    excerpt,
    image,
    imageAlt,
    canonical,
    keywords,
    ogType,
    ogTitle,
    ogDescription,
    twitterCard,
    twitterTitle,
    twitterDescription,
    jsonLd,
  };
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
  const filePath = join(process.cwd(), 'content/blog', `${slug}.md`);
  try {
    const fileContent = readFileSync(filePath, 'utf-8');
    const { data: rawFrontmatter, content } = matter(fileContent);
    const frontmatter = validateFrontmatter(rawFrontmatter as Record<string, unknown>, slug);

    return {
      slug,
      frontmatter,
      content,
      authors: normalizeAuthors(frontmatter),
    };
  } catch (error) {
    // Surface validation or read errors to fail the build
    console.error(`Error loading blog post "${slug}":`, error);
    throw error;
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

  const fallbackJsonLd: JsonLdObject = {
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