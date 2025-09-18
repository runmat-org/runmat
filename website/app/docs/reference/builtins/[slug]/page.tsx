import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { loadBuiltins } from '@/lib/builtins';

export const dynamic = 'force-static';

export async function generateStaticParams() {
  const builtins = loadBuiltins().filter(b => !b.internal);
  return builtins.map(b => ({ slug: b.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  const b = loadBuiltins().find(x => x.slug === slug);
  if (!b) return { title: 'Builtin | Docs' };
  return {
    title: `${b.name} | Docs`,
    description: `Runmat Matlab Runtime builtin function ${b.name}`,
    openGraph: { title: `${b.name} - RunMat`, description: `Runmat Matlab Runtime builtin function ${b.name}` },
    twitter: { card: 'summary', title: `${b.name} - RunMat`, description: `Runmat Matlab Runtime builtin function ${b.name}` },
  };
}

export default async function BuiltinPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const b = loadBuiltins().find(x => x.slug === slug);
  if (!b) notFound();
  return (
    <div className="min-h-screen">
      <h1 className="text-3xl font-bold mb-2">{b.name}</h1>
      {b.summary && <p className="text-muted-foreground mb-6 max-w-[70ch]">{b.summary}</p>}
      <div className="mb-6 flex items-center gap-2 flex-wrap">
        {b.category.map(c => (
          <span key={c} className="text-[11px] px-2 py-1 rounded bg-muted text-muted-foreground">{c}</span>
        ))}
      </div>

      <section className="mb-8" id="usage">
        <h2 className="text-xl font-semibold mb-2">Usage</h2>
        <ul className="space-y-2 text-sm">
          {b.signatures.map((s, i) => {
            const argList = s.in.map((name, idx) => s.inTypes && s.inTypes[idx] ? `${name}: ${s.inTypes[idx]}` : name).join(', ');
            const outList = s.out.map((name, idx) => s.outTypes && s.outTypes[idx] ? `${name}: ${s.outTypes[idx]}` : name).join(', ');
            return (
              <li key={i} className="rounded border border-border p-3">
                <code className="block">{b.name}({argList})</code>
                {s.out.length > 0 && <div className="text-muted-foreground">returns [{outList}]</div>}
                <div className="text-muted-foreground">nargin: {s.nargin.min}..{s.nargin.max} · nargout: {s.nargout.min}..{s.nargout.max}</div>
              </li>
            );
          })}
        </ul>
      </section>

      {b.examples && b.examples.length > 0 && (
        <section className="mb-8" id="examples">
          <h2 className="text-xl font-semibold mb-2">Examples</h2>
          <ul className="space-y-2 text-sm">
            {b.examples.map((ex, i) => (
              <li key={i} className="rounded border border-border p-3 bg-muted/30">
                {ex.title && <div className="font-medium mb-1">{ex.title}</div>}
                <pre className="whitespace-pre-wrap">{ex.code}</pre>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}


