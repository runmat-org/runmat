import { Metadata } from "next";
import { notFound } from "next/navigation";
import { join } from "path";
import { readFileSync, existsSync } from "fs";
import { docsTree, findNodeBySlug, findPathBySlug, flatten, DocsNode } from "@/content/docs";
import { DocsContentSwitch } from "@/components/DocsContentSwitch";
import { DocsArticleVisibility } from "@/components/DocsArticleVisibility";
import { MarkdownRenderer } from "@/components/MarkdownRenderer";
import { PageToc } from "@/components/PageToc";

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

export async function generateMetadata({ params }: { params: Promise<{ slug?: string[] }> }): Promise<Metadata> {
  const { slug = [] } = await params;
  const node = findNodeBySlug(slug);
  const base: Metadata = { title: node ? `${node.title} | Docs` : "Docs" };
  if (!node) return base;
  const seo = (node as DocsNode).seo;
  return {
    ...base,
    description: seo?.description,
    keywords: seo?.keywords,
    openGraph: seo ? { title: seo.ogTitle ?? base.title as string, description: seo.ogDescription ?? seo.description } : undefined,
    twitter: seo ? { card: 'summary_large_image', title: seo.ogTitle ?? base.title as string, description: seo.ogDescription ?? seo.description } : undefined,
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
  // Files in the manifest are repo-root relative (e.g., "docs/..." or "crates/...").
  // Compute a robust set of candidate absolute paths, walking up to detect the repo root.
  const repoRoot = findRepoRoot(process.cwd());
  const candidatePaths = Array.from(
    new Set([
      join(process.cwd(), node.file),
      join(process.cwd(), "..", node.file),
      join(process.cwd(), "..", "..", node.file),
      repoRoot ? join(repoRoot, node.file) : undefined,
    ].filter(Boolean) as string[])
  );
  let source = "";
  for (const p of candidatePaths) {
    try {
      source = readFileSync(p, "utf-8");
      break;
    } catch {}
  }
  if (!source) notFound();
  const crumbs = findPathBySlug(slug) ?? [];
  
  return (
    <div className="grid lg:grid-cols-[minmax(0,1fr)_260px] gap-8">
      <article className="prose dark:prose-invert max-w-none scroll-smooth">
        <nav className="text-sm text-muted-foreground mb-4">
          {crumbs
            .filter((c) => c.title)
            .map((c, i) => (
              <span key={i}>
                {i > 0 && <span className="mx-1">/</span>}
                {c.slug ? (
                  <a href={["/docs", ...(c.slug || [])].join("/")} className="hover:text-foreground">
                    {c.title}
                  </a>
                ) : (
                  <span>{c.title}</span>
                )}
              </span>
            ))}
        </nav>
        {/* Title is already conveyed by breadcrumbs and in-page H1 within markdown, avoid duplication */}
        <DocsArticleVisibility>
          <MarkdownRenderer source={source} />
        </DocsArticleVisibility>
        {/* Client search/results overlay */}
        <DocsContentSwitch source={source} />
      </article>
      <PageToc source={source} />
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


