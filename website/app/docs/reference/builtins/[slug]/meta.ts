import type { Metadata } from 'next';
import builtinsData from '@/content/builtins.json';
import type { Builtin } from '@/lib/builtins';
import { resolveOgFields } from '@/lib/og';
import { buildPageMetadata } from '@/lib/seo';

const BUILTINS: Builtin[] = builtinsData as Builtin[];
const BUILTIN_MAP: Map<string, Builtin> = new Map(BUILTINS.map(b => [b.slug, b]));
const FALLBACK_DESCRIPTION = 'RunMat / MATLAB Language Function documentation';

export function builtinMetadataForSlug(slug: string): Metadata {
    const builtin = BUILTIN_MAP.get(slug);
    if (!builtin) {
        const title = `${slug} | MATLAB Language Function Reference`;
        return buildPageMetadata({
          title,
          description: FALLBACK_DESCRIPTION,
          canonicalPath: `/docs/reference/builtins/${slug}`,
          ogType: 'article',
          ogImagePath: `/docs/reference/builtins/${slug}/opengraph-image`,
        });
    }

    const description = (
        builtin.description ||
        builtin.summary ||
        FALLBACK_DESCRIPTION
    ).trim();

    const title = `${builtin.name} | MATLAB Language Function Reference`;
    return buildPageMetadata({
      title,
      description,
      canonicalPath: `/docs/reference/builtins/${slug}`,
      ogType: 'article',
      ogImagePath: `/docs/reference/builtins/${slug}/opengraph-image`,
    });
}

export function builtinOgTitleSubtitle(slug: string): { title: string; subtitle: string } {
    const meta = builtinMetadataForSlug(slug);
    let result: { title: string; subtitle: string };
    try {
        result = resolveOgFields(meta);
    } catch {
        console.warn(
            '[builtins-og] Falling back to slug-based OG metadata',
            { slug, meta }
        );
        result = {
            title: typeof meta.title === 'string' ? meta.title : slug,
            subtitle: typeof meta.description === 'string' ? meta.description : FALLBACK_DESCRIPTION,
        };
    }
    return result;
}