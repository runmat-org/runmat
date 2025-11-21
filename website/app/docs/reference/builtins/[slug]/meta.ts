import type { Metadata } from 'next';
import { loadBuiltins } from '@/lib/builtins';
import { resolveOgFields } from '@/lib/og';

export function builtinMetadataForSlug(slug: string): Metadata {
    const builtin = loadBuiltins().find(x => x.slug === slug);
    if (!builtin) return {};

    const description = (
        builtin.description ||
        builtin.summary ||
        'RunMat / MATLAB Language Function documentation'
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
    return resolveOgFields(meta);
}