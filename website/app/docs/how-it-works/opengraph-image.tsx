import { createOgResponse, OG_SIZE } from '@/lib/og';

export const size = { width: OG_SIZE.width, height: OG_SIZE.height };
export const contentType = 'image/png';
export const dynamic = 'force-static';

export default function Image() {
  return createOgResponse({ 
    title: 'How It Works', 
    subtitle: 'Understanding RunMat\'s execution model and runtime' 
  });
}


