import type { Metadata } from 'next';
import { getBuiltinDocBySlug } from '@/lib/builtins';
import { resolveOgFields } from '@/lib/og';
import { buildPageMetadata } from '@/lib/seo';

const FALLBACK_DESCRIPTION = 'RunMat / MATLAB Language Function documentation';

export function builtinMetadataForSlug(slug: string): Metadata {
    const builtin = getBuiltinDocBySlug(slug);
    if (!builtin) {
        const title = `${slug} — MATLAB Function Reference | Run Examples Live`;
        return buildPageMetadata({
          title,
          description: FALLBACK_DESCRIPTION,
          canonicalPath: `/docs/reference/builtins/${slug}`,
          ogType: 'article',
          ogImagePath: `/docs/reference/builtins/${slug}/opengraph-image`,
        });
    }

    const summary = (builtin.summary || '').trim();
    const description = summary
      ? `${summary} Try ${builtin.title}() live — edit code, see output instantly. No MATLAB license needed.`
      : (builtin.description || FALLBACK_DESCRIPTION).trim();

    const title = `${builtin.title} — MATLAB Function Reference | Run Examples Live`;
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

export function builtinJsonLD(slug: string): string {
    const builtin = getBuiltinDocBySlug(slug);
    const examples = builtin?.examples ?? [];
    const faqs = builtin?.faqs ?? [];

    const mainEntityRefs: Record<string, string>[] = [{"@id": "#source-code"}];
    if (examples.length > 0) mainEntityRefs.push({"@id": "#how-to-guide"});
    if (faqs.length > 0) mainEntityRefs.push({"@id": "#faq"});

    const graph: Record<string, unknown>[] = [
        {
            "@type": "WebPage",
            "@id": `https://runmat.com/docs/reference/builtins/${slug}`,
            "url": `https://runmat.com/docs/reference/builtins/${slug}`,
            "name": `${builtin?.title} - RunMat Reference`,
            "isPartOf": {"@id": "https://runmat.com/#website"},
            "author": {"@id": "https://runmat.com/#organization"},
            "publisher": {"@id": "https://runmat.com/#organization"},
            "primaryImageOfPage": {
                "@type": "ImageObject",
                "url": "https://web.runmatstatic.com/runmat-sandbox-dark.png"
            },
            "breadcrumb": {"@id": `https://runmat.com/docs/reference/builtins/${slug}#breadcrumbs`},
            "mainEntity": {"@id": "#article"}
        },
        {
            "@type": "BreadcrumbList",
            "@id": `https://runmat.com/docs/reference/builtins/${slug}#breadcrumbs`,
            "itemListElement": [
                {"@type": "ListItem", "position": 1, "name": "Docs", "item": "https://runmat.com/docs"},
                {"@type": "ListItem", "position": 2, "name": "Reference", "item": "https://runmat.com/docs/matlab-function-reference"},
                {"@type": "ListItem", "position": 3, "name": builtin?.title, "item": `https://runmat.com/docs/reference/builtins/${slug}`}
            ]
        },
        {
            "@type": "APIReference",
            "@id": `https://runmat.com/docs/reference/builtins/${slug}#article`,
            "headline": `${builtin?.title} in MATLAB — runnable examples + source (RunMat)`,
            "alternativeHeadline": `${builtin?.title} Function Reference`,
            "description": `Run ${builtin?.title} in your browser. Open-source, JIT-compiled MATLAB-compatible documentation for ${builtin?.description}.`,
            "proficiencyLevel": "Beginner",
            "programmingModel": "Procedural",
            "assemblyVersion": "RunMat 0.1.0",
            "dateModified": new Date().toISOString().split('T')[0],
            "programmingLanguage": {"@type": "ComputerLanguage", "name": "MATLAB", "alternateName": "RunMat"},
            "author": {"@id": "https://runmat.com/#organization"},
            "publisher": {"@id": "https://runmat.com/#organization"},
            "about": {
                "@type": "DefinedTerm",
                "name": builtin?.title,
                "description": builtin?.description,
                "inDefinedTermSet": {
                    "@type": "DefinedTermSet",
                    "name": "RunMat Builtin Functions",
                    "url": "https://runmat.com/docs/matlab-function-reference"
                }
            },
            "mainEntity": mainEntityRefs
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
            "runtimePlatform": ["RunMat Web (WASM)", "RunMat Desktop", "RunMat CLI"],
            "url": builtin?.source?.url
        },
    ];

    if (examples.length > 0) {
        graph.push({
            "@type": "HowTo",
            "@id": "#how-to-guide",
            "name": `How to use ${builtin?.title} in RunMat`,
            "step": examples.map(example => {
                const description = example.description?.trim() || 'Example';
                const anchor = description.toLowerCase().replace(/\s+/g, '-');
                return {
                    "@type": "HowToStep",
                    "name": description,
                    "url": `https://runmat.com/docs/reference/builtins/${slug}#${anchor}`,
                    "text": example.input,
                    "itemListElement": {"@type": "HowToDirection", "text": example.input}
                };
            })
        });
    }

    if (faqs.length > 0) {
        graph.push({
            "@type": "FAQPage",
            "@id": "#faq",
            "mainEntity": faqs.map(faq => ({
                "@type": "Question",
                "name": faq.question,
                "acceptedAnswer": {"@type": "Answer", "text": faq.answer}
            }))
        });
    }

    return JSON.stringify({"@context": "https://schema.org", "@graph": graph});
}
