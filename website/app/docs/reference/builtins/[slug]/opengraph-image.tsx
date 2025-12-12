import { createOgResponse, OG_SIZE } from '@/lib/og';
import { builtinOgTitleSubtitle } from './meta';

export const size = { width: OG_SIZE.width, height: OG_SIZE.height };
export const contentType = 'image/png';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export default function Image({ params }: { params: { slug: string } }) {
  const { slug } = params;
  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/adc5a0ef-9c26-4e0c-abc4-335154a72020',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sessionId:'debug-session',runId:'pre-fix',hypothesisId:'H1',location:'docs/reference/builtins/[slug]/opengraph-image.tsx:Image',message:'OG image request received',data:{slug},timestamp:Date.now()})}).catch(()=>{});
  // #endregion
  const { title, subtitle } = builtinOgTitleSubtitle(slug);
  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/adc5a0ef-9c26-4e0c-abc4-335154a72020',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sessionId:'debug-session',runId:'pre-fix',hypothesisId:'H2',location:'docs/reference/builtins/[slug]/opengraph-image.tsx:Image',message:'Resolved OG title/subtitle',data:{slug,title,subtitle},timestamp:Date.now()})}).catch(()=>{});
  // #endregion
  return createOgResponse({ title, subtitle });
}