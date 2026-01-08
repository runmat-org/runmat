import { Metadata } from "next";
import { notFound } from "next/navigation";
import { join } from "path";
import { readFileSync, existsSync } from "fs";
import { docsTree, findNodeBySlug, findPathBySlug, flatten, DocsNode } from "@/content/docs";
import { DocsContentSwitch } from "@/components/DocsContentSwitch";
import { DocsArticleVisibility } from "@/components/DocsArticleVisibility";
import { MarkdownRenderer } from "@/components/MarkdownRenderer";
import { HeadingsNav } from "@/components/HeadingsNav";
import matter from "gray-matter";

// Polyfill URL.canParse for Node environments that don't support it yet (e.g., Node 18)
const _u = URL;
if (typeof _u.canParse !== "function") {
  _u.canParse = (input: string, base?: string) => {
    try {
      // eslint-disable-next-line no-new
      new URL(input, base);
      return true;
    } catch {
      return false;
    }
  };
}

export const dynamic = "force-static";

export async function generateStaticParams(): Promise<{ slug: string[] }[]> {
  // Pre-render all manifest-backed pages so they work on Vercel and locally
  return flatten(docsTree)
    .filter((n) => !!n.slug && !!n.file)
    .map((n) => ({ slug: n.slug! }));
}

type DocFrontmatter = {
  title?: string;
  description?: string;
  keywords?: string | string[];
  ogTitle?: string;
  ogDescription?: string;
  jsonLd?: unknown;
};

function coerceKeywords(val?: string | string[]) {
  if (!val) return undefined;
  if (Array.isArray(val)) return val.filter(Boolean);
  const trimmed = val.trim();
  return trimmed ? [trimmed] : undefined;
}

function isJsonLdObject(val: unknown): Record<string, unknown> | undefined {
  if (!val || typeof val !== "object" || Array.isArray(val)) return undefined;
  return val as Record<string, unknown>;
}

function readDocSource(file: string): { body: string; data: DocFrontmatter } | null {
  const repoRoot = findRepoRoot(process.cwd());
  const candidatePaths = Array.from(
    new Set([
      join(process.cwd(), file),
      join(process.cwd(), "..", file),
      join(process.cwd(), "..", "..", file),
      repoRoot ? join(repoRoot, file) : undefined,
    ].filter(Boolean) as string[])
  );

  for (const p of candidatePaths) {
    try {
      const raw = readFileSync(p, "utf-8");
      const parsed = matter(raw);
      return { body: parsed.content, data: parsed.data as DocFrontmatter };
    } catch {
      continue;
    }
  }
  return null;
}

export async function generateMetadata({ params }: { params: Promise<{ slug?: string[] }> }): Promise<Metadata> {
  const { slug = [] } = await params;
  const node = findNodeBySlug(slug);
  const baseTitle = node ? `${node.title} | Docs` : "Docs";
  const base: Metadata = { title: baseTitle };
  if (!node) return base;

  const seo = (node as DocsNode).seo;
  const parsed = node.file ? readDocSource(node.file) : null;
  const fm = parsed?.data || {};

  const title = fm.title || node.title || baseTitle;
  const description = fm.description || seo?.description;
  const keywords = coerceKeywords(fm.keywords) ?? seo?.keywords;
  const ogTitle = fm.ogTitle || seo?.ogTitle || title;
  const ogDescription = fm.ogDescription || seo?.ogDescription || description;

  return {
    title,
    description,
    keywords,
    openGraph: (ogTitle || ogDescription) ? { title: ogTitle, description: ogDescription } : undefined,
    twitter: (ogTitle || ogDescription) ? { card: 'summary_large_image', title: ogTitle, description: ogDescription } : undefined,
  };
}

export default async function DocPage({ params }: { params: Promise<{ slug?: string[] }> }) {
  const { slug = [] } = await params;
  let node = findNodeBySlug(slug);
  // Small fallback: allow known slug families even if omitted from the manifest
  if (!node) {
    const fallback = fallbackForSlug(slug);
    if (fallback) node = fallback;
  }
  if (!node || !node.file) notFound();
  const parsed = readDocSource(node.file);
  if (!parsed) notFound();
  const { body, data } = parsed;
  const jsonLdObj = isJsonLdObject(data.jsonLd);
  const jsonLdString = jsonLdObj ? JSON.stringify(jsonLdObj).replace(/<\//g, "<\\/") : undefined;
  const crumbs = findPathBySlug(slug) ?? [];
  return (
    <div className="grid lg:grid-cols-[minmax(0,1fr)_260px] gap-8">
      <article className="prose dark:prose-invert max-w-none scroll-smooth min-w-0">
        <nav className="text-sm text-muted-foreground mb-4">
          {crumbs
            .filter((c) => c.title)
            .map((c, i) => (
              <span key={i}>
                {i > 0 && <span className="mx-1">/</span>}
                {(() => {
                  const href = c.slug ? ["/docs", ...(c.slug || [])].join("/") : c.externalHref;
                  if (href) {
                    return (
                      <a href={href} className="hover:text-foreground">
                        {c.title}
                      </a>
                    );
                  }
                  return <span>{c.title}</span>;
                })()}
              </span>
            ))}
        </nav>
        {/* Title is already conveyed by breadcrumbs and in-page H1 within markdown, avoid duplication */}
        {jsonLdString && (
          <script
            type="application/ld+json"
            dangerouslySetInnerHTML={{ __html: jsonLdString }}
          />
        )}
        <DocsArticleVisibility>
          <MarkdownRenderer source={body} />
        </DocsArticleVisibility>
        {/* Client search/results overlay */}
        <DocsContentSwitch source={body} />
      </article>
      <HeadingsNav source={body} />
    </div>
  );
}

// Map well-known slug prefixes to repo files when not present in the manifest
function fallbackForSlug(slug: string[]) {
  if (slug.length === 2 && slug[0] === "ignition") {
    const map: Record<string, string> = {
      "compiler-pipeline": "crates/runmat-ignition/COMPILER_PIPELINE.md",
      "instr-set": "crates/runmat-ignition/INSTR_SET.md",
      "indexing-and-slicing": "crates/runmat-ignition/INDEXING_AND_SLICING.md",
      "error-model": "crates/runmat-ignition/ERROR_MODEL.md",
      "oop-semantics": "crates/runmat-ignition/OOP_SEMANTICS.md",
    };
    const file = map[slug[1]];
    if (file) return { title: toTitleCase(slug[1].replace(/-/g, " ")), slug, file };
  }
  return undefined;
}

function toTitleCase(s: string): string {
  return s.replace(/\b\w/g, (m) => m.toUpperCase());
}

function findRepoRoot(start: string): string | undefined {
  const markers = [
    "Cargo.toml", // Rust workspace
    "docs/ARCHITECTURE.md",
    ".git",
  ];
  let dir = start;
  for (let i = 0; i < 5; i++) {
    const hasMarker = markers.some((m) => existsSync(join(dir, m)));
    if (hasMarker) return dir;
    const parent = join(dir, "..");
    if (parent === dir) break;
    dir = parent;
  }
  return undefined;
}


