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
    const description = builtin.seo_description
      ? builtin.seo_description
      : summary
        ? `${summary} Try ${builtin.title}() live — edit code, see output instantly. No MATLAB license needed.`
        : (builtin.description || FALLBACK_DESCRIPTION).trim();

    const title = builtin.seo_title || `${builtin.title} — MATLAB Function Reference | Run Examples Live`;
    const ogImagePath = builtin.hero_image || `/docs/reference/builtins/${slug}/opengraph-image`;
    const ogImageAlt = summary
      ? `${builtin.title} in RunMat — ${summary}`
      : `${builtin.title} — MATLAB function reference in RunMat`;
    return buildPageMetadata({
      title,
      description,
      canonicalPath: `/docs/reference/builtins/${slug}`,
      ogType: 'article',
      ogImagePath,
      ogImageAlt,
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
    const title = builtin?.title ?? slug;
    const examples = builtin?.examples ?? [];
    const faqs = builtin?.faqs ?? [];
    const heroImage = builtin?.hero_image || "https://web.runmatstatic.com/runmat-sandbox-dark.png";
    const sourceUrl = builtin?.source?.url?.trim();

    const pageUrl = `https://runmat.com/docs/reference/builtins/${slug}`;
    const articleId = `${pageUrl}#article`;
    const breadcrumbId = `${pageUrl}#breadcrumbs`;
    const sourceCodeId = `${pageUrl}#source-code`;
    const howToId = `${pageUrl}#how-to-guide`;
    const faqId = `${pageUrl}#faq`;

    const mainEntityRefs: Record<string, string>[] = [];
    if (sourceUrl) mainEntityRefs.push({ "@id": sourceCodeId });
    if (examples.length > 0) mainEntityRefs.push({ "@id": howToId });
    if (faqs.length > 0) mainEntityRefs.push({ "@id": faqId });

    const apiReference: Record<string, unknown> = {
        "@type": "APIReference",
        "@id": articleId,
        "headline": `${title} in MATLAB — runnable examples + source (RunMat)`,
        "description": `Run ${title} in your browser. Open-source, JIT-compiled MATLAB-compatible documentation for ${stripMarkdown(builtin?.description ?? '')}`.slice(0, 300),
        "image": heroImage,
        "programmingLanguage": {"@type": "ComputerLanguage", "name": "MATLAB", "alternateName": "RunMat"},
        "author": {"@id": "https://runmat.com/#organization"},
        "publisher": {"@id": "https://runmat.com/#organization"},
        "about": {
            "@type": "DefinedTerm",
            "name": title,
            "description": stripMarkdown(builtin?.description ?? ''),
            "inDefinedTermSet": {
                "@type": "DefinedTermSet",
                "name": "RunMat Builtin Functions",
                "url": "https://runmat.com/docs/matlab-function-reference"
            }
        },
    };
    if (mainEntityRefs.length > 0) apiReference.mainEntity = mainEntityRefs;

    const graph: Record<string, unknown>[] = [
        {
            "@type": "WebPage",
            "@id": pageUrl,
            "url": pageUrl,
            "name": `${title} - RunMat Reference`,
            "isPartOf": {"@id": "https://runmat.com/#website"},
            "author": {"@id": "https://runmat.com/#organization"},
            "publisher": {"@id": "https://runmat.com/#organization"},
            "primaryImageOfPage": {
                "@type": "ImageObject",
                "url": heroImage
            },
            "breadcrumb": {"@id": breadcrumbId},
            "mainEntity": {"@id": articleId}
        },
        {
            "@type": "BreadcrumbList",
            "@id": breadcrumbId,
            "itemListElement": [
                {"@type": "ListItem", "position": 1, "name": "Docs", "item": "https://runmat.com/docs"},
                {"@type": "ListItem", "position": 2, "name": "Reference", "item": "https://runmat.com/docs/matlab-function-reference"},
                {"@type": "ListItem", "position": 3, "name": title, "item": pageUrl}
            ]
        },
        apiReference,
    ];

    if (sourceUrl) {
        graph.push({
            "@type": "SoftwareSourceCode",
            "@id": sourceCodeId,
            "name": `${title}.rs`,
            "description": `The Rust implementation of the ${title} function, compatible with all RunMat targets.`,
            "codeRepository": "https://github.com/runmat-org/runmat",
            "programmingLanguage": {"@type": "ComputerLanguage", "name": "Rust"},
            "codeSampleType": "full implementation",
            "targetProduct": {"@type": "SoftwareApplication", "name": "RunMat"},
            "license": "https://opensource.org/licenses/MIT",
            "runtimePlatform": ["RunMat Web (WASM)", "RunMat Desktop", "RunMat CLI"],
            "url": sourceUrl,
        });
    }

    if (examples.length > 0) {
        graph.push({
            "@type": "HowTo",
            "@id": howToId,
            "name": `How to use ${title} in RunMat`,
            "description": `Runnable examples showing how to use ${title} in RunMat. Each example runs in the browser with no setup.`,
            "image": heroImage,
            "step": examples.map(example => {
                const description = example.description?.trim() || 'Example';
                const anchor = description.toLowerCase().replace(/\s+/g, '-');
                const stepImage = example.image_webp || example.image;
                const step: Record<string, unknown> = {
                    "@type": "HowToStep",
                    "name": description,
                    "url": `${pageUrl}#${anchor}`,
                    "text": example.input,
                };
                if (stepImage) step.image = stepImage;
                return step;
            })
        });
    }

    if (faqs.length > 0) {
        graph.push({
            "@type": "FAQPage",
            "@id": faqId,
            "mainEntity": faqs.map(faq => ({
                "@type": "Question",
                "name": faq.question,
                "acceptedAnswer": {"@type": "Answer", "text": markdownToHtml(faq.answer)}
            }))
        });
    }

    return JSON.stringify({"@context": "https://schema.org", "@graph": graph});
}

// Strip markdown formatting for use in plain-text schema.org fields
// (descriptions, about.description). Preserves content, removes backticks,
// code fences, and list bullets so search engines see clean prose.
function stripMarkdown(text: string): string {
    return text
        .replace(/```[\s\S]*?```/g, '') // remove fenced code blocks
        .replace(/`([^`]+)`/g, '$1')     // unwrap inline code
        .replace(/\*\*([^*]+)\*\*/g, '$1') // bold
        .replace(/\*([^*]+)\*/g, '$1')     // italic
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // links
        .replace(/^\s*[-*]\s+/gm, '')     // list bullets
        .replace(/\s+/g, ' ')              // collapse whitespace
        .trim();
}

// Minimal markdown-to-HTML conversion for FAQ answers in JSON-LD.
// Google's FAQPage spec supports HTML in answer text, but not raw markdown.
// Converts code fences, inline code, bold/italic, and paragraphs.
function markdownToHtml(text: string): string {
    let html = text;
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_m, _lang, code) => {
        const escaped = String(code)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        return `<pre><code>${escaped.trim()}</code></pre>`;
    });
    html = html.replace(/`([^`]+)`/g, (_m, code) => {
        const escaped = String(code)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        return `<code>${escaped}</code>`;
    });
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
    const paragraphs = html.split(/\n{2,}/).map(p => {
        const trimmed = p.trim();
        if (!trimmed) return '';
        if (trimmed.startsWith('<pre>') || trimmed.startsWith('<ul>')) return trimmed;
        return `<p>${trimmed.replace(/\n/g, ' ')}</p>`;
    }).filter(Boolean);
    return paragraphs.join('');
}
