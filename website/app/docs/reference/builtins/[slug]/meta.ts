import type { Metadata } from 'next';
import { loadBuiltins } from '@/lib/builtins';
import { resolveOgFields } from '@/lib/og';

export function builtinMetadataForSlug(slug: string): Metadata {
  const b = loadBuiltins().find(x => x.slug === slug);
  if (!b) return {};
  return {
    title: `${b.name} | Function Reference`,
    description: `RunMat / MATLAB Language Function documentation`,
    openGraph: {
      title: `${b.name}`,
      description: `MATLAB Language Function Documentation`,
      images: [{ url: `/docs/reference/builtins/${slug}/opengraph-image` }],
    },
    twitter: {
      card: 'summary_large_image',
      title: `${b.name}`,
      description: `MATLAB Language Function documentation`,
      images: [{ url: `/docs/reference/builtins/${slug}/opengraph-image` }],
    },
    alternates: { canonical: `/docs/reference/builtins/${slug}` },
  } satisfies Metadata;
}

export function builtinOgTitleSubtitle(slug: string): { title: string; subtitle: string } {
  const meta = builtinMetadataForSlug(slug);
  return resolveOgFields(meta);
}