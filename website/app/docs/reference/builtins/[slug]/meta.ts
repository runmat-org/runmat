import type { Metadata } from 'next';
import builtinsData from '@/content/builtins.json';
import type { Builtin } from '@/lib/builtins';
import { resolveOgFields } from '@/lib/og';

const BUILTINS: Builtin[] = builtinsData as Builtin[];
const BUILTIN_MAP: Map<string, Builtin> = new Map(BUILTINS.map(b => [b.slug, b]));
const FALLBACK_DESCRIPTION = 'RunMat / MATLAB Language Function documentation';

export function builtinMetadataForSlug(slug: string): Metadata {
    const builtin = BUILTIN_MAP.get(slug);
    if (!builtin) {
        const title = `${slug} | MATLAB Language Function Reference`;
        return {
            title,
            description: FALLBACK_DESCRIPTION,
            openGraph: {
                title: slug,
                description: FALLBACK_DESCRIPTION,
            },
            twitter: {
                card: 'summary_large_image',
                title: slug,
                description: FALLBACK_DESCRIPTION,
            },
            alternates: { canonical: `/docs/reference/builtins/${slug}` },
        } satisfies Metadata;
    }

    const description = (
        builtin.description ||
        builtin.summary ||
        FALLBACK_DESCRIPTION
    ).trim();

    return {
        title: `${builtin.name} | MATLAB Language Function Reference`,
        description,
        openGraph: {
            title: builtin.name,
            description,
        },
        twitter: {
            card: 'summary_large_image',
            title: builtin.name,
            description,
        },
        alternates: { canonical: `/docs/reference/builtins/${slug}` },
    } satisfies Metadata;
}

export function builtinOgTitleSubtitle(slug: string): { title: string; subtitle: string } {
    const meta = builtinMetadataForSlug(slug);
    let result: { title: string; subtitle: string };
    try {
        result = resolveOgFields(meta);
    } catch {
        // Guard against missing data to avoid 500s in OG route
        result = {
            title: typeof meta.title === 'string' ? meta.title : slug,
            subtitle: typeof meta.description === 'string' ? meta.description : FALLBACK_DESCRIPTION,
        };
    }
    return result;
}