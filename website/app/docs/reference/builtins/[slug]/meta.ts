import type { Metadata } from 'next';
import { getBuiltinDocBySlug, formatCategoryLabel } from '@/lib/builtins';
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
        title: { absolute: `${builtin.title} in MATLAB: Free Online Compiler & Reference` },
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

export function builtinJsonLD(slug: string): string {
    const builtin = getBuiltinDocBySlug(slug);
    const jsonLd = {
        "@context": "https://schema.org",
        "@graph": [

            {
                "@type": "WebPage",
                "@id": `https://runmat.org/docs/reference/builtins/${builtin?.title}`,
                "url": `https://runmat.org/docs/reference/builtins/${builtin?.title}`,
                "name": `${builtin?.title} - RunMat Reference`,
                "isPartOf": {"@id": "https://runmat.org/#website"},
                "primaryImageOfPage": {
                    "@type": "ImageObject",
                    "url": "https://web.runmatstatic.com/runmat-sandbox-dark.png"
                },
                "breadcrumb": {"@id": "#breadcrumbs"},
                "mainEntity": {"@id": "#article"}
            },


            {
                "@type": "BreadcrumbList",
                "itemListElement": [
                    {
                        "@type": "ListItem",
                        "position": 1,
                        "name": "Docs",
                        "item": "https://runmat.org/docs"
                    },
                    {
                        "@type": "ListItem",
                        "position": 2,
                        "name": "Reference",
                        "item": "https://runmat.org/docs/matlab-function-reference"
                    },
                    {
                        "@type": "ListItem",
                        "position": 3,
                        "name": builtin?.title,
                        "item": `https://runmat.org/docs/reference/builtins/${builtin?.title}`
                    }
                ]
            },

            {
                "@type": "APIReference",
                "@id": `https://runmat.org/docs/reference/builtins/${builtin?.title}#article`,
                "headline": `${builtin?.title} in MATLAB — runnable examples + source (RunMat)`,
                "alternativeHeadline": `${builtin?.title} Function Reference`,
                "description": `Run ${builtin?.title} in your browser. Open-source, JIT-compiled MATLAB-compatible documentation for ${builtin?.description}.`,

                "proficiencyLevel": "Beginner",
                "programmingModel": "Procedural",
                "assemblyVersion": "RunMat 0.1.0",
                "datePublished": "2025-01-01",
                "dateModified": new Date().toISOString().split('T')[0],

                "programmingLanguage": {
                    "@type": "ComputerLanguage",
                    "name": "MATLAB",
                    "alternateName": "RunMat"
                },

                "author": {
                    "@type": "Organization",
                    "name": "RunMat",
                    "url": "https://runmat.org",
                    "alternateName": ["RunMat by Dystr", "Dystr"]
                },

                "about": {
                    "@type": "DefinedTerm",
                    "name": builtin?.title,
                    "description": builtin?.description,
                    "inDefinedTermSet": {
                        "@type": "DefinedTermSet",
                        "name": "RunMat Builtin Functions",
                        "url": "https://runmat.org/docs/matlab-function-reference"
                    }
                },

                "mainEntity": [
                    {"@id": "#source-code"},
                    {"@id": "#how-to-guide"},
                    {"@id": "#faq"}
                ]
            },


            {
                "@type": "SoftwareSourceCode",
                "@id": "#source-code",
                "name": `${builtin?.title}.rs`,
                "description": `The Rust implementation of the ${builtin?.title} function, compatible with all RunMat targets.`,
                "codeRepository": "https://github.com/runmat-org/runmat",
                "programmingLanguage": {"@type": "ComputerLanguage", "name": "Rust"},
                "codeSampleType": "full implementation",
                "targetProduct": {"@type": "SoftwareApplication", "name": "RunMat"},
                "license": "https://opensource.org/licenses/MIT",
                "runtimePlatform": [
                    "RunMat Web (WASM)",
                    "RunMat Desktop",
                    "RunMat CLI"
                ],


                "url": `https://github.com/runmat-org/runmat/blob/main/${builtin?.title}`
            },

            {
                "@type": "HowTo",
                "@id": "#how-to-guide",
                "name": `How to use ${builtin?.title} in RunMat`,
                "step": (builtin?.examples ?? []).map(example => ({
                    "@type": "HowToStep",
                    "name": example.description,
                    "url": `https://runmat.org/docs/reference/builtins/${builtin?.title}#${example.description.toLowerCase().replace(/\s+/g, '-').toLowerCase()}`,
                    "text": example.input,
                    "itemListElement": {
                        "@type": "HowToDirection",
                        "text": example.input
                    }
                }))
            },

            {
                "@type": "FAQPage",
                "@id": "#faq",
                "mainEntity": (builtin?.faqs ?? []).map(faq => ({
                    "@type": "Question",
                    "name": faq.question,
                    "acceptedAnswer": {
                        "@type": "Answer",
                        "text": faq.answer
                    }
                }))
            }
        ]
    };
    return JSON.stringify(jsonLd);
}
