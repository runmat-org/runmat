import { Suspense } from 'react';
import { loadBuiltins } from '@/lib/builtins';
import BuiltinsExplorer from '@/components/BuiltinsExplorer';
import { buildPageMetadata } from '@/lib/seo';

export const metadata = buildPageMetadata({
  title: 'Built-in Functions | Docs',
  description: 'Search and browse RunMat built-in functions by name, category, or keyword.',
  canonicalPath: '/docs/reference/builtins',
  ogType: 'website',
  ogImagePath: '/docs/reference/builtins/opengraph-image',
});

export default async function BuiltinsIndexPage() {
    const builtins = loadBuiltins();
    return (
        <div className="container mx-auto px-4 md:px-6 py-8">
            <h1 className="text-3xl md:text-4xl font-bold mb-2">Builtin Function Reference</h1>
            <p className="text-muted-foreground mb-6">
                Discover MATLAB-compatible builtins implemented in RunMat. Search by name, category, or keywords.
            </p>
            <Suspense fallback={<div>Loading...</div>}>
                <BuiltinsExplorer builtins={builtins} />
            </Suspense>
        </div>
    );
}