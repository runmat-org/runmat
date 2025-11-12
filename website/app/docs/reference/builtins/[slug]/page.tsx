import type { Metadata } from 'next';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { loadBuiltins } from '@/lib/builtins';
import fs from 'fs';
import path from 'path';
import { MarkdownRenderer } from '@/components/MarkdownRenderer';
import { slugifyHeading } from '@/lib/utils';
import { builtinMetadataForSlug } from './meta';

export const dynamic = 'force-static';

export async function generateStaticParams() {
  const builtins = loadBuiltins();
  return builtins.map(b => ({ slug: b.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  return builtinMetadataForSlug(slug);
}

export default async function BuiltinDetailPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const builtins = loadBuiltins();
  const b = builtins.find(x => x.slug === slug);
  if (!b) return notFound();
  let source = '';
  if (b.mdxPath) {
    try {
      const abs = path.join(process.cwd(), 'content', b.mdxPath);
      source = fs.readFileSync(abs, 'utf-8');
    } catch { }
  }
  if (!source) return notFound();
  const toc = extractHeadings(source);
  return (
    <>
      <div className="container mx-auto px-4 md:px-6 pt-8">
        <p className="my-6 text-muted-foreground leading-relaxed break-words">
          <Link href="/docs/matlab-function-reference" className="text-muted-foreground hover:text-foreground transition-colors">
            View all functions
          </Link>
        </p>
      </div>
      <div className="container mx-auto px-4 md:px-6 pb-8">
      <div className="grid gap-8 lg:grid-cols-[1fr_280px]">
        <article className="prose dark:prose-invert max-w-none prose-headings:font-semibold prose-h2:mt-8 prose-h2:mb-3 prose-h3:mt-6 prose-h3:mb-2 prose-pre:bg-muted prose-pre:border prose-pre:rounded-md prose-code:bg-muted prose-code:border prose-code:rounded-sm">
          <MarkdownRenderer source={source} />
        </article>
        {toc.length > 0 && (
          <aside className="hidden lg:block sticky top-24 h-max text-sm">
            <div className="mb-2 font-semibold text-foreground/80">On this page</div>
            <ul className="space-y-1 text-muted-foreground">
              {toc.map((t) => (
                <li key={t.id} className={t.depth === 3 ? 'pl-3' : ''}>
                  <a href={`#${t.id}`} className="hover:text-foreground">{t.text}</a>
                </li>
              ))}
            </ul>
          </aside>
        )}
      </div>
    </div>
    </>
  );
}

function extractHeadings(md: string): { id: string; text: string; depth: number }[] {
  const lines = md.split(/\r?\n/);
  const out: { id: string; text: string; depth: number }[] = [];
  let inFence = false;
  for (const raw of lines) {
    const line = raw.trim();
    if (/^```/.test(line)) { inFence = !inFence; continue; }
    if (inFence) continue;
    const m = /^(#{2,6})\s+(.+)$/.exec(line);
    if (!m) continue;
    const depth = m[1].length; // 2..6
    const text = m[2].replace(/`/g, '');
    const id = slugifyHeading(text);
    out.push({ id, text, depth });
  }
  return out;
}