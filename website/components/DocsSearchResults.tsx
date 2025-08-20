"use client";
import { useMemo } from "react";
import { useSearchParams } from "next/navigation";
import Fuse from "fuse.js";
import { docsTree, flatten, type DocsNode } from "@/content/docs";

type DocItem = { title: string; href: string; summary: string; keywords: string[] };

export function DocsSearchResults({ source }: { source: string }) {
  const params = useSearchParams();
  const q = (params.get('q') || '').trim().toLowerCase();
  const index: DocItem[] = useMemo(() => {
    const nodes: DocsNode[] = flatten(docsTree);
    const pages = nodes.filter((n) => n.externalHref || n.slug);
    return pages.map((n) => ({
      title: n.title,
      href: n.externalHref ?? ["/docs", ...(n.slug || [])].join("/"),
      summary: n.seo?.description || "",
      keywords: n.seo?.keywords || [],
    }));
  }, []);

  const results = useMemo(() => {
    if (!q) return [] as { title: string; href: string; summary: string }[];
    const fuse = new Fuse(index, {
      includeMatches: true,
      threshold: 0.35,
      ignoreLocation: true,
      keys: [
        { name: 'title', weight: 0.6 },
        { name: 'summary', weight: 0.35 },
        { name: 'keywords', weight: 0.05 },
      ],
    });
    const hits = fuse.search(q);
    const dedup: Record<string, boolean> = {};
    const out: { title: string; href: string; summary: string }[] = [];
    for (const h of hits) {
      const doc = h.item;
      if (dedup[doc.href]) continue;
      dedup[doc.href] = true;
      out.push({ title: doc.title, href: doc.href, summary: doc.summary });
      if (out.length >= 10) break;
    }
    return out;
  }, [q, index]);

  if (!q) return (
    <p className="text-muted-foreground">Type to search all docs…</p>
  );
  return (
    <div>
      <h1 className="text-2xl font-semibold mb-4">Search results for “{params.get('q') || ''}”</h1>
      {results.length === 0 ? (
        <p className="text-muted-foreground">No results found.</p>
      ) : (
        <ul className="space-y-6">
          {results.map((r, i) => (
            <li key={i} className="rounded border border-border p-4">
              <a href={r.href} className="text-lg font-semibold hover:underline">{r.title}</a>
              {r.summary && <p className="mt-2 text-sm text-muted-foreground">{r.summary}</p>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}


