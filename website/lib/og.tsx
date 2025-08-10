import { ImageResponse } from 'next/og';
import type { Metadata } from 'next';
import { theme } from './theme';

export const OG_SIZE = { width: 1200, height: 630 } as const;

type OgOptions = {
  title: string;
  subtitle?: string;
};

// Use extracted CSS tokens from globals.css
const colors = theme.dark;

export function renderOg({ title, subtitle }: OgOptions) {
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
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 8,
              background: colors.brandGradient,
              boxShadow: '0 6px 18px rgba(62,167,253,0.35)',
            }}
          />
          <div
            style={{
              fontSize: 28,
              fontWeight: 700,
              color: colors.foreground,
              letterSpacing: 1,
            }}
          >
            RunMat
          </div>
        </div>

        <div style={{ marginTop: 24 }}>
          <div
            style={{
              fontSize: 64,
              lineHeight: 1.1,
              fontWeight: 800,
              color: colors.foreground,
              backgroundImage: colors.brandGradient,
              WebkitBackgroundClip: 'text',
              backgroundClip: 'text',
              colorAdjust: 'exact',
              WebkitTextFillColor: 'transparent',
            }}
          >
            {title}
          </div>
          {subtitle ? (
            <div style={{ fontSize: 28, color: colors.muted, marginTop: 12 }}>{subtitle}</div>
          ) : null}
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 24 }}>
          <div style={{ fontSize: 22, color: colors.muted }}>Fast, Free, Modern MATLAB Runtime</div>
          <div style={{ fontSize: 22, color: colors.muted }}>runmat.org</div>
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


