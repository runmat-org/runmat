import { ImageResponse } from 'next/og';
import type { Metadata } from 'next';
import { theme } from './theme';
import Logo from '@/components/Logo';

export const OG_SIZE = { width: 1200, height: 630 } as const;

type OgOptions = {
  title: string;
  subtitle?: string;
  imageUrl?: string; // if provided, render right-side image (1/3 width)
  siteTitle?: string; // footer tagline
  siteUrl?: string; // footer link text
};

// Use extracted CSS tokens from globals.css
const colors = {
  // Increase contrast for OG cards
  background: '#0b0b0c',
  card: '#0f1012',
  border: '#2a2b2f',
  foreground: '#ffffff',
  muted: '#b8bcc6',
  brandGradient: theme.dark.brandGradient,
};

export function renderOg({ title, subtitle, imageUrl, siteTitle = 'Fast, Free, Modern MATLAB Runtime', siteUrl = 'runmat.com' }: OgOptions) {
  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        backgroundColor: colors.background,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <div
        style={{
          width: 1100,
          height: 520,
          borderRadius: 24,
          background: colors.card,
          border: `2px solid ${colors.border}`,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: 48,
        }}
      >
        <div style={{ display: 'flex', gap: 24, flex: 1 }}>
          <div style={{ flex: imageUrl ? 2 : 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div
              style={{
                fontSize: 104,
                lineHeight: 1.05,
                fontWeight: 900,
                color: colors.foreground,
                letterSpacing: -1,
              }}
            >
              {title}
            </div>
            {subtitle ? (
              <div style={{ fontSize: 40, color: colors.muted, marginTop: 16 }}>{subtitle}</div>
            ) : null}
          </div>
          {imageUrl ? (
            <div style={{ flex: 1, height: '100%' }}>
              <img
                src={imageUrl}
                alt=""
                width={Math.round(1100 / 3)}
                height={424}
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  borderRadius: 16,
                  border: `1px solid ${colors.border}`,
                }}
              />
            </div>
          ) : null}
        </div>

        {/* Footer */}
        <div style={{ display: 'flex', justifyContent: 'flex-start', alignItems: 'center', marginTop: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 35 }}>
            <Logo height={50} />
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <div style={{ fontSize: 40, color: colors.foreground, fontWeight: 700 }}>{siteTitle}</div>
              <div style={{ fontSize: 32, color: colors.muted }}>{siteUrl}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function createOgResponse(opts: OgOptions) {
  return new ImageResponse(renderOg(opts), {
    width: OG_SIZE.width,
    height: OG_SIZE.height,
  });
}

function titleFromMetaTitle(t: Metadata['title']): string | undefined {
  if (!t) return undefined;
  if (typeof t === 'string') return t;
  // Structured title object
  // Prefer absolute > default
  if (typeof t === 'object') {
    // @ts-expect-error narrow structured title
    if (t.absolute && typeof t.absolute === 'string') return t.absolute as string;
    // @ts-expect-error narrow structured title
    if (t.default && typeof t.default === 'string') return t.default as string;
  }
  return undefined;
}

export function resolveOgFields(meta: Metadata): { title: string; subtitle: string } {
  const og = meta.openGraph;
  const ogTitle = typeof og?.title === 'string' ? og?.title : undefined;
  const ogDesc = typeof og?.description === 'string' ? og?.description : undefined;
  const title = ogTitle ?? titleFromMetaTitle(meta.title);
  const subtitle = ogDesc ?? (typeof meta.description === 'string' ? meta.description : undefined);
  if (!title || !subtitle) {
    throw new Error('OG image requires metadata.openGraph.title/description or metadata.title/description');
  }
  return { title, subtitle };
}
