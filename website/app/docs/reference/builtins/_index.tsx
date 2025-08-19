import { Metadata } from 'next';
import Link from 'next/link';
import { loadBuiltins, progress } from '@/lib/builtins';

export const dynamic = 'force-static';

export const metadata: Metadata = {
  title: 'RunMat Built-in Functions',
  description: 'Reference and status for MATLAB-compatible built-in functions in RunMat.'
};

export default function BuiltinsIndexPage() {
  const builtins = loadBuiltins();
  const p = progress(builtins, 1200);
  const byCategory = new Map<string, number>();
  for (const b of builtins) for (const c of b.category) byCategory.set(c, (byCategory.get(c) || 0) + 1);

  return (
    <div className="min-h-screen">
      <h1 className="text-3xl font-bold mb-2">Built-in Functions</h1>
      <p className="text-muted-foreground mb-4">MATLAB-compatible builtins implemented in RunMat.</p>
      <div className="mb-6">
        <div className="h-2 bg-muted rounded">
          <div className="h-2 bg-green-600 rounded" style={{ width: `${Math.round(p.pct * 100)}%` }} />
        </div>
        <div className="text-sm text-muted-foreground mt-1">{p.implemented} / {p.total} implemented</div>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        <section className="md:col-span-2">
          <ul className="divide-y divide-border rounded border border-border">
            {builtins
              .slice()
              .sort((a, b) => a.name.localeCompare(b.name))
              .map((b) => (
                <li key={b.slug} className="p-3 hover:bg-muted/40">
                  <div className="flex items-center justify-between gap-3">
                    <Link href={`/docs/reference/builtins/${b.slug}`} className="font-medium hover:underline">{b.name}</Link>
                    <span className={`text-[11px] px-2 py-1 rounded ${b.status === 'implemented' ? 'bg-green-200/20 text-green-600' : 'bg-amber-200/20 text-amber-600'}`}>{b.status}</span>
                  </div>
                  {b.summary && <div className="text-sm text-muted-foreground mt-1">{b.summary}</div>}
                </li>
              ))}
          </ul>
        </section>
        <aside>
          <h2 className="text-sm font-semibold mb-2">Categories</h2>
          <ul className="space-y-1 text-sm">
            {[...byCategory.entries()].sort((a,b)=>b[1]-a[1]).map(([c,n]) => (
              <li key={c} className="flex justify-between"><span>{c}</span><span className="text-muted-foreground">{n}</span></li>
            ))}
          </ul>
        </aside>
      </div>
    </div>
  );
}


