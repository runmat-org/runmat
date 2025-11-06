import { createOgResponse, OG_SIZE } from '@/lib/og';

export const size = { width: OG_SIZE.width, height: OG_SIZE.height };
export const contentType = 'image/png';
export const dynamic = 'force-static';

export default function Image() {
  return createOgResponse({ 
    title: 'RunMat', 
    subtitle: 'Fast, Free, Modern MATLAB Runtime' 
  });
}