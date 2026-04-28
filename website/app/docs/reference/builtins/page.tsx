import { Suspense } from 'react';
import Link from 'next/link';
import { loadBuiltins, type Builtin } from '@/lib/builtins';
import BuiltinsExplorer from '@/components/BuiltinsExplorer';
import { buildPageMetadata } from '@/lib/seo';

export const metadata = buildPageMetadata({
  title: 'Built-in Functions | Docs',
  description: 'Search and browse RunMat built-in functions by name, category, or keyword.',
  canonicalPath: '/docs/matlab-function-reference',
  ogType: 'website',
  ogImagePath: '/docs/reference/builtins/opengraph-image',
});

export default async function BuiltinsIndexPage() {
    const builtins = loadBuiltins();
    const publicBuiltins = builtins
        .filter(b => !b.internal)
        .sort((a, b) => a.name.localeCompare(b.name));
    const grouped = groupByInitial(publicBuiltins);
    return (
        <div className="container mx-auto px-4 md:px-6 py-8">
            <h1 className="text-2xl sm:text-3xl font-bold mb-2">Builtin Function Reference</h1>
            <p className="text-foreground text-[0.938rem] mb-6">
                Discover MATLAB-compatible builtins implemented in RunMat. Search by name, category, or keywords.
            </p>
            <Suspense fallback={<div>Loading...</div>}>
                <BuiltinsExplorer builtins={builtins} />
            </Suspense>
            <section aria-labelledby="az-index" className="mt-12 border-t border-border pt-8">
                <h2 id="az-index" className="text-xl font-semibold mb-2">All functions A–Z</h2>
                <p className="text-sm text-muted-foreground mb-4">
                    Complete alphabetical index of every public builtin ({publicBuiltins.length}).
                </p>
                <div className="space-y-6">
                    {grouped.map(({ letter, items }) => (
                        <div key={letter}>
                            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                                {letter}
                            </h3>
                            <ul className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-x-4 gap-y-1 text-sm">
                                {items.map(b => (
                                    <li key={b.slug}>
                                        <Link
                                            href={`/docs/reference/builtins/${b.slug}`}
                                            className="text-foreground hover:text-[hsl(var(--brand))] hover:underline underline-offset-4"
                                        >
                                            {b.name}
                                        </Link>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    ))}
                </div>
            </section>
        </div>
    );
}

function groupByInitial(items: Builtin[]): { letter: string; items: Builtin[] }[] {
    const map = new Map<string, Builtin[]>();
    for (const b of items) {
        const first = b.name.charAt(0).toUpperCase();
        const letter = /[A-Z]/.test(first) ? first : '#';
        const bucket = map.get(letter);
        if (bucket) bucket.push(b);
        else map.set(letter, [b]);
    }
    return [...map.entries()]
        .sort(([a], [b]) => (a === '#' ? 1 : b === '#' ? -1 : a.localeCompare(b)))
        .map(([letter, items]) => ({ letter, items }));
}