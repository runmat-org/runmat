import { createOgResponse, OG_SIZE } from '@/lib/og';
import { builtinOgTitleSubtitle } from './meta';

export const size = { width: OG_SIZE.width, height: OG_SIZE.height };
export const contentType = 'image/png';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export default function Image({ params }: { params: { slug: string } }) {
  const { slug } = params;
  const { title, subtitle } = builtinOgTitleSubtitle(slug);
  return createOgResponse({ title, subtitle });
}