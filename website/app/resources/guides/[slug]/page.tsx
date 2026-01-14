import { Metadata } from "next";
import { notFound } from "next/navigation";
import { readFileSync, readdirSync } from "fs";
import { join } from "path";
import matter from "gray-matter";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { MarkdownRenderer } from "@/components/MarkdownRenderer";

type GuideFrontmatter = {
  title?: string;
  description?: string;
  date?: string;
  readTime?: string;
};

type Guide = {
  slug: string;
  frontmatter: GuideFrontmatter;
  content: string;
};

function getGuide(slug: string): Guide | null {
  const filePath = join(process.cwd(), "content", "resources", "guides", `${slug}.md`);
  try {
    const raw = readFileSync(filePath, "utf-8");
    const { data, content } = matter(raw);
    return { slug, frontmatter: data as GuideFrontmatter, content };
  } catch {
    return null;
  }
}

export function generateStaticParams() {
  const dir = join(process.cwd(), "content", "resources", "guides");
  try {
    const files = readdirSync(dir).filter((f) => f.endsWith(".md"));
    return files.map((file) => ({ slug: file.replace(/\.md$/, "") }));
  } catch {
    return [];
  }
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  const guide = getGuide(slug);
  if (!guide) return {};
  const title = guide.frontmatter.title || guide.slug;
  return {
    title,
    description: guide.frontmatter.description,
    openGraph: {
      title,
      description: guide.frontmatter.description,
      type: "article",
    },
  };
}

export default async function GuidePage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const guide = getGuide(slug);
  if (!guide) notFound();

  const { frontmatter, content } = guide;
  const title = frontmatter.title || guide.slug;

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-10 md:px-6 md:py-14">
        <Card className="border">
          <CardContent className="prose dark:prose-invert max-w-none p-6">
            <div className="mb-4 flex items-center gap-2 text-sm text-muted-foreground">
              <Badge variant="secondary">Guide</Badge>
              {frontmatter.date && <span>{new Date(frontmatter.date).toLocaleDateString()}</span>}
              {frontmatter.readTime && <span>• {frontmatter.readTime}</span>}
            </div>
            <h1 className="mb-4 text-3xl font-bold leading-tight">{title}</h1>
            {frontmatter.description && (
              <p className="text-muted-foreground mb-6">{frontmatter.description}</p>
            )}
            <MarkdownRenderer source={content} />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

