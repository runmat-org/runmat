import { Metadata } from 'next';
import { DocsSearchResults } from '@/components/DocsSearchResults';

export const dynamic = 'force-static';

export const metadata: Metadata = {
  title: 'Search | Docs',
  description: 'Search the RunMat documentation with fuzzy matching and highlighted results.',
};

export default function DocsSearchPage() {
  return (
    <div className="min-h-screen">
      <div className="text-sm text-muted-foreground mb-4">Documentation / Search</div>
      <DocsSearchResults source={''} />
    </div>
  );
}


