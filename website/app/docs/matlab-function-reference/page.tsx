import { loadBuiltins } from '@/lib/builtins';
import ElementsOfMatlabGrid from '@/components/ElementsOfMatlabGrid';
import { buildPageMetadata } from '@/lib/seo';

export const metadata = buildPageMetadata({
  title: 'MATLAB Function Reference | Docs',
  description:
    'Clickable map of core MATLAB building blocks implemented in RunMat. Explore data types, array operations, math functions, control flow, I/O, and more.',
  canonicalPath: '/docs/matlab-function-reference',
  ogType: 'website',
  ogImagePath: '/docs/opengraph-image',
});

export default async function ElementsOfMatlabPage() {
  const builtins = loadBuiltins();
  return (
    <>
      <div className="mb-8">
        <h1 className="text-3xl md:text-4xl font-bold mb-2">MATLAB Function Reference</h1>
        <p className="text-muted-foreground">
    Explore MATLAB-compatible builtins (functions) implemented in RunMat. Search by name, category, or keywords.


        </p>
      </div>
      <ElementsOfMatlabGrid builtins={builtins} />
    </>
  );
}

