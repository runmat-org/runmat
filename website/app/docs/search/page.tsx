import { DocsSearchResults } from '@/components/DocsSearchResults';
import { buildPageMetadata } from '@/lib/seo';

export const dynamic = 'force-static';

export const metadata = buildPageMetadata({
  title: 'Search | Docs',
  description: 'Search the RunMat documentation with fuzzy matching and highlighted results.',
  canonicalPath: '/docs/search',
  ogType: 'website',
  ogImagePath: '/docs/opengraph-image',
});

export default function DocsSearchPage() {
  return (
    <div className="min-h-screen">
      <div className="text-sm text-muted-foreground mb-4">Documentation / Search</div>
      <DocsSearchResults source={''} />
    </div>
  );
}


