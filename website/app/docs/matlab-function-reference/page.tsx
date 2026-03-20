import { loadBuiltins } from '@/lib/builtins';
import ElementsOfMatlabGrid from '@/components/ElementsOfMatlabGrid';
import { buildPageMetadata } from '@/lib/seo';

export const metadata = buildPageMetadata({
  title: 'Built-in Function Reference: MATLAB-Compatible & Beyond | Docs',
  description:
    'Browse 310+ built-in functions in RunMat. MATLAB-compatible math, linear algebra, I/O, and GPU-accelerated extensions. Free, open-source, no license required.',
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
        name: 'Built-in Function Reference | RunMat',
        description: `Browse ${publicBuiltins.length}+ built-in functions in RunMat. MATLAB-compatible and beyond. Free, open-source, GPU-accelerated.`,
        isPartOf: { '@id': 'https://runmat.com/#website' },
        publisher: { '@id': 'https://runmat.com/#organization' },
        mainEntity: { '@id': '#function-list' },
      },
      {
        '@type': 'BreadcrumbList',
        '@id': 'https://runmat.com/docs/matlab-function-reference#breadcrumbs',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Docs', item: 'https://runmat.com/docs' },
          { '@type': 'ListItem', position: 2, name: 'Built-in Function Reference', item: 'https://runmat.com/docs/matlab-function-reference' },
        ],
      },
      {
        '@type': 'ItemList',
        '@id': '#function-list',
        name: 'RunMat Built-in Functions',
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
        <h1 className="text-3xl md:text-4xl font-bold mb-2">Built-in Function Reference</h1>
        <p className="text-muted-foreground max-w-3xl">
          {builtins.filter(b => !(b as Record<string, unknown>).internal).length}+ built-in functions
          covering math, linear algebra, statistics, string processing, file I/O, and more.
          Most are MATLAB-compatible. The library keeps growing with RunMat-native additions.
          Free, open-source, GPU-accelerated on Apple, Nvidia, and AMD hardware.
        </p>
      </div>
      <ElementsOfMatlabGrid builtins={builtins} />
    </>
  );
}

