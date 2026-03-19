import { loadBuiltins } from '@/lib/builtins';
import ElementsOfMatlabGrid from '@/components/ElementsOfMatlabGrid';
import { buildPageMetadata } from '@/lib/seo';

export const metadata = buildPageMetadata({
  title: 'MATLAB Function Reference | Docs',
  description:
    'Browse 310+ MATLAB-compatible functions implemented in RunMat — free, open-source, and GPU-accelerated. Search by name, category, or keyword.',
  canonicalPath: '/docs/matlab-function-reference',
  ogType: 'website',
  ogImagePath: '/docs/opengraph-image',
});

function buildJsonLd(builtins: { name: string; slug: string; summary: string }[]) {
  const publicBuiltins = builtins.filter((b: { name: string }) => !(b as Record<string, unknown>).internal);
  return JSON.stringify({
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'CollectionPage',
        '@id': 'https://runmat.com/docs/matlab-function-reference',
        url: 'https://runmat.com/docs/matlab-function-reference',
        name: 'MATLAB Function Reference — RunMat',
        description: `Browse ${publicBuiltins.length}+ MATLAB-compatible functions implemented in RunMat — free, open-source, and GPU-accelerated.`,
        isPartOf: { '@id': 'https://runmat.com/#website' },
        publisher: { '@id': 'https://runmat.com/#organization' },
        mainEntity: { '@id': '#function-list' },
      },
      {
        '@type': 'BreadcrumbList',
        '@id': 'https://runmat.com/docs/matlab-function-reference#breadcrumbs',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Docs', item: 'https://runmat.com/docs' },
          { '@type': 'ListItem', position: 2, name: 'MATLAB Function Reference', item: 'https://runmat.com/docs/matlab-function-reference' },
        ],
      },
      {
        '@type': 'ItemList',
        '@id': '#function-list',
        name: 'RunMat MATLAB-Compatible Functions',
        numberOfItems: publicBuiltins.length,
        itemListElement: publicBuiltins.map((b, i) => ({
          '@type': 'ListItem',
          position: i + 1,
          name: b.name,
          url: `https://runmat.com/docs/reference/builtins/${b.slug}`,
          description: b.summary || undefined,
        })),
      },
    ],
  });
}

export default async function ElementsOfMatlabPage() {
  const builtins = loadBuiltins();
  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: buildJsonLd(builtins) }}
      />
      <div className="mb-8">
        <h1 className="text-3xl md:text-4xl font-bold mb-2">MATLAB Function Reference</h1>
        <p className="text-muted-foreground max-w-3xl">
          RunMat implements {builtins.filter(b => !(b as Record<string, unknown>).internal).length}+ MATLAB-compatible
          functions spanning math, linear algebra, statistics, string processing, file I/O, and
          more. Every function is free, open-source, and runs with automatic GPU acceleration on
          Apple, Nvidia, and AMD hardware — no license required.
        </p>
        <p className="text-muted-foreground mt-2 max-w-3xl">
          Search by name, category, or keyword, or browse the categories below. Each function page
          includes runnable examples you can try in the browser.
        </p>
      </div>
      <ElementsOfMatlabGrid builtins={builtins} />
    </>
  );
}

