import { readFileSync } from 'fs';
import { join } from 'path';
import matter from 'gray-matter';
import { createOgResponse, OG_SIZE } from '@/lib/og';

export const size = { width: OG_SIZE.width, height: OG_SIZE.height };
export const contentType = 'image/png';
export const runtime = 'edge';
export const dynamic = 'force-static';

function getFrontmatter(slug: string): { title?: string; description?: string; image?: string } | null {
  try {
    const filePath = join(process.cwd(), 'content/blog', `${slug}.md`);
    const fileContent = readFileSync(filePath, 'utf-8');
    const { data } = matter(fileContent);
    return data as any;
  } catch {
    return null;
  }
}

export default function Image({ params }: { params: { slug: string } }) {
  const fm = getFrontmatter(params.slug);
  const title = fm?.title ?? 'RunMat Blog';
  const subtitle = fm?.description ?? 'Latest updates and insights from the RunMat team';
  const imageUrl = fm?.image && (fm.image.startsWith('http://') || fm.image.startsWith('https://') || fm.image.startsWith('/'))
    ? fm.image
    : undefined;
  return createOgResponse({ title, subtitle, imageUrl, siteTitle: 'Fast, Free, Modern MATLAB Runtime', siteUrl: 'runmat.org' });
}