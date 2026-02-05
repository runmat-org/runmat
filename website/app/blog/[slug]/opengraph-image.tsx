import { readFileSync } from 'fs';
import matter from 'gray-matter';
import { createOgResponse, OG_SIZE } from '@/lib/og';
import { resolveBlogFilePath } from '@/lib/blog';

export const size = { width: OG_SIZE.width, height: OG_SIZE.height };
export const contentType = 'image/png';
export const runtime = 'nodejs';
export const dynamic = 'force-static';

type BlogFrontmatter = { title?: string; description?: string; image?: string };

function getFrontmatter(slug: string): BlogFrontmatter | null {
  const filePath = resolveBlogFilePath(slug);
  if (!filePath) return null;

  try {
    const fileContent = readFileSync(filePath, 'utf-8');
    const { data } = matter(fileContent);
    const fm = data as Partial<BlogFrontmatter>;
    return {
      title: typeof fm.title === 'string' ? fm.title : undefined,
      description: typeof fm.description === 'string' ? fm.description : undefined,
      image: typeof fm.image === 'string' ? fm.image : undefined,
    };
  } catch {
    return null;
  }
}

export default async function Image({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const fm = getFrontmatter(slug);
  const title = fm?.title ?? 'RunMat Blog';
  const subtitle = fm?.description ?? 'Latest updates and insights from the RunMat team';
  const imageUrl = fm?.image && (fm.image.startsWith('http://') || fm.image.startsWith('https://') || fm.image.startsWith('/'))
    ? fm.image
    : undefined;
  return createOgResponse({ title, subtitle, imageUrl, siteTitle: 'Fast, Free, Modern MATLAB Runtime', siteUrl: 'runmat.com' });
}
