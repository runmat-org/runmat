"use client";

import { useEffect, useMemo, useState } from 'react';
import Fuse from 'fuse.js';
import Link from 'next/link';
import type { Builtin } from '@/lib/builtins';

export default function BuiltinsExplorer({ builtins }: { builtins: Builtin[] }) {
  const [q, setQ] = useState('');
  const [hideInternal, setHideInternal] = useState(true);
  const [selectedCategories, setSelectedCategories] = useState<Set<string>>(new Set());
  const [page, setPage] = useState<number>(1);
  const [pageSize, setPageSize] = useState<number>(25);

  const categories = useMemo(() => {
    const set = new Set<string>();
    for (const b of builtins) for (const c of b.category) set.add(c);
    return [...set].sort((a,b)=>a.localeCompare(b));
  }, [builtins]);

  const fuse = useMemo(() => new Fuse(builtins, {
    includeScore: true,
    threshold: 0.35,
    keys: [
      { name: 'name', weight: 0.6 },
      { name: 'summary', weight: 0.2 },
      { name: 'category', weight: 0.15 },
      { name: 'keywords', weight: 0.05 },
    ],
  }), [builtins]);

  const results = useMemo(() => {
    let items = builtins.slice();
    if (q.trim()) {
      items = fuse.search(q).map(r => r.item);
    }
    if (hideInternal) items = items.filter(b => !b.internal);
    if (selectedCategories.size > 0) {
      items = items.filter(b => b.category.some(c => selectedCategories.has(c)));
    }
    return items.sort((a, b) => a.name.localeCompare(b.name));
  }, [builtins, fuse, q, hideInternal, selectedCategories]);

  // Reset to first page whenever filters/search/page size change
  useEffect(() => {
    setPage(1);
  }, [q, hideInternal, selectedCategories, pageSize]);

  const pageCount = Math.max(1, Math.ceil(results.length / pageSize));
  const currentPage = Math.min(page, pageCount);
  const startIdx = (currentPage - 1) * pageSize;
  const endIdx = Math.min(results.length, startIdx + pageSize);
  const pagedResults = results.slice(startIdx, endIdx);

  return (
    <div className="grid md:grid-cols-3 gap-6">
      <section className="md:col-span-2">
        <div className="mb-3 flex items-center gap-2">
          <input
            value={q}
            onChange={(e)=>setQ(e.target.value)}
            placeholder="Search builtins…"
            className="w-full text-sm rounded-md border border-border bg-muted/40 pr-3 pl-3 py-2 h-9 focus:outline-none focus:ring-2 focus:ring-primary/60 focus:border-primary placeholder:text-muted-foreground/80 shadow-sm"
          />
        </div>
        {/* Pagination header */}
        <div className="mb-2 flex items-center justify-between text-sm">
          <div className="text-muted-foreground">
            Showing <span className="font-medium">{results.length === 0 ? 0 : startIdx + 1}</span>–<span className="font-medium">{endIdx}</span> of <span className="font-medium">{results.length}</span>
          </div>
          <div className="flex items-center gap-2">
            <label htmlFor="page-size" className="text-muted-foreground">Per page</label>
            <select
              id="page-size"
              value={pageSize}
              onChange={(e)=>setPageSize(Number(e.target.value))}
              className="h-8 rounded-md border border-border bg-muted/40 px-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/60"
            >
              <option value={10}>10</option>
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
            <div className="ml-2 flex items-center gap-2">
              <button
                type="button"
                onClick={()=> setPage(p => Math.max(1, p - 1))}
                disabled={currentPage <= 1}
                className="h-8 px-3 rounded-md border border-border bg-muted/40 disabled:opacity-50 hover:bg-muted"
                aria-label="Previous page"
              >Prev</button>
              <button
                type="button"
                onClick={()=> setPage(p => Math.min(pageCount, p + 1))}
                disabled={currentPage >= pageCount}
                className="h-8 px-3 rounded-md border border-border bg-muted/40 disabled:opacity-50 hover:bg-muted"
                aria-label="Next page"
              >Next</button>
            </div>
          </div>
        </div>
        <ul className="divide-y divide-border rounded border border-border">
          {pagedResults.map((b) => (
            <li key={b.slug} className="p-3 hover:bg-muted/40">
              <div className="flex items-center justify-between gap-3">
                <Link href={`/docs/reference/builtins/${b.slug}`} className="font-medium hover:underline">{b.name}</Link>
              </div>
              {b.summary && <div className="text-sm text-muted-foreground mt-1">{b.summary}</div>}
            </li>
          ))}
          {results.length === 0 && (
            <li className="p-3 text-sm text-muted-foreground">No builtins match your filters.</li>
          )}
        </ul>
        {/* Pagination footer */}
        <div className="mt-3 flex items-center justify-between text-sm">
          <div className="text-muted-foreground">Page <span className="font-medium">{currentPage}</span> of <span className="font-medium">{pageCount}</span></div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={()=> setPage(p => Math.max(1, p - 1))}
              disabled={currentPage <= 1}
              className="h-8 px-3 rounded-md border border-border bg-muted/40 disabled:opacity-50 hover:bg-muted"
              aria-label="Previous page"
            >Prev</button>
            <button
              type="button"
              onClick={()=> setPage(p => Math.min(pageCount, p + 1))}
              disabled={currentPage >= pageCount}
              className="h-8 px-3 rounded-md border border-border bg-muted/40 disabled:opacity-50 hover:bg-muted"
              aria-label="Next page"
            >Next</button>
          </div>
        </div>
      </section>
      <aside>
        <h2 className="text-sm font-semibold mb-2">Filters</h2>
        <div className="space-y-4 text-sm">
          <div className="flex items-center gap-2">
            <input id="hide-internal" type="checkbox" className="accent-foreground" checked={hideInternal} onChange={(e)=>setHideInternal(e.target.checked)} />
            <label htmlFor="hide-internal">Hide internal (__)</label>
          </div>
          <div>
            <div className="text-muted-foreground mb-1">Categories</div>
            <ul className="max-h-64 overflow-auto pr-1 space-y-1">
              {categories.map(c => (
                <li key={c} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    className="accent-foreground"
                    checked={selectedCategories.has(c)}
                    onChange={(e)=>{
                      const next = new Set(selectedCategories);
                      if (e.target.checked) next.add(c); else next.delete(c);
                      setSelectedCategories(next);
                    }}
                  />
                  <span>{c}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </aside>
    </div>
  );
}
