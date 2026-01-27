import { createOgResponse, OG_SIZE } from '@/lib/og';
import { builtinOgTitleSubtitle } from './meta';

export const size = { width: OG_SIZE.width, height: OG_SIZE.height };
export const contentType = 'image/png';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export default async function Image({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const { title, subtitle } = builtinOgTitleSubtitle(slug);
  return createOgResponse({ title, subtitle });
}