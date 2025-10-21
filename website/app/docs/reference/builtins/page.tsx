import type { Metadata } from 'next';
import { loadBuiltins } from '@/lib/builtins';
import BuiltinsExplorer from '@/components/BuiltinsExplorer';

export const metadata: Metadata = {
    title: 'Builtin Function Reference',
    description: 'Search and browse RunMat builtin functions by name, category, or keyword.',
    alternates: { canonical: '/docs/reference/builtins' },
};

export default async function BuiltinsIndexPage() {
    const builtins = loadBuiltins();
    return (
        <div className="container mx-auto px-4 md:px-6 py-8">
            <h1 className="text-3xl md:text-4xl font-bold mb-2">Builtin Function Reference</h1>
            <p className="text-muted-foreground mb-6">
                Discover MATLAB-compatible builtins implemented in RunMat. Search by name, category, or keywords.
            </p>
            <BuiltinsExplorer builtins={builtins} />
        </div>
    );
}