import type { Metadata } from 'next';
import { getBuiltinDocBySlug } from '@/lib/builtins';
import { resolveOgFields } from '@/lib/og';

const FALLBACK_DESCRIPTION = 'RunMat / MATLAB Language Function documentation';

export function builtinMetadataForSlug(slug: string): Metadata {
    const builtin = getBuiltinDocBySlug(slug);
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

    const description = (builtin.description || builtin.summary || FALLBACK_DESCRIPTION).trim();

    return {
        title: { absolute: `${builtin.title} in MATLAB - runnable examples online + source | RunMat docs` },
        description,
        openGraph: {
            title: builtin.title,
            description,
        },
        twitter: {
            card: 'summary_large_image',
            title: builtin.title,
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
