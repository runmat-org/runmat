import type { Metadata } from 'next';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import {
  formatCategoryLabel,
  getBuiltinDocBySlug,
  getBuiltinMetadata,
  loadBuiltins,
  type BuiltinDocEntry,
  type BuiltinDocExample,
  type BuiltinDocLink,
  type BuiltinDocFAQ,
  type BuiltinDocJsonEncodeOption,
  type BuiltinDocValidation,
} from '@/lib/builtins';
import { BuiltinDocRenderer, type BuiltinDocBlock, type BuiltinDocInlineNode } from '@/components/BuiltinDocRenderer';
import { slugifyHeading } from '@/lib/utils';
import { builtinMetadataForSlug, builtinJsonLD } from './meta';
import { BuiltinMetadataChips } from '@/components/BuiltinMetadataChips';
import { getDisplayCategory } from '@/lib/builtin-utils';
import { categoryAnchorIdForRaw } from '@/lib/display-categories';
import { BuiltinsHeadingsNav } from '@/components/BuiltinsHeadingsNav';
import { SandboxCta } from '@/components/SandboxCta';


export const dynamic = 'force-static';
export const dynamicParams = false;

export async function generateStaticParams() {
  const builtins = loadBuiltins();
  return builtins.map(b => ({ slug: b.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  return builtinMetadataForSlug(slug);
}

export default async function BuiltinDetailPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const builtins = loadBuiltins();
  const b = builtins.find(x => x.slug === slug);
  if (!b) notFound();
  const doc = getBuiltinDocBySlug(slug);
  if (!doc) notFound();
  const siblings = builtins
    .filter(x => x.slug !== slug)
    .sort((a, c) => a.name.localeCompare(c.name));
  const blocks = renderBuiltinDocBlocks(doc, siblings);
  const toc = extractHeadingsFromBlocks(blocks);
  const metadata = getBuiltinMetadata(b);
  const allDisplayCategories = [...new Set(builtins.map(x => getDisplayCategory(x)))];
  const categoryAnchor = categoryAnchorIdForRaw(b.category[0] ?? '', allDisplayCategories);

  return (
    <>
    <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: builtinJsonLD(slug) }}
    />
      <div className="pt-8">
        <p className="mb-4 text-sm text-muted-foreground leading-relaxed break-words">
          <Link href="/docs/matlab-function-reference" className="inline-flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors">
            <span aria-hidden="true">&larr;</span> All functions
          </Link>
        </p>
        <BuiltinMetadataChips
          metadata={metadata}
          categoryAnchor={categoryAnchor}
          gpuSectionAnchor={doc.gpu_behavior && doc.gpu_behavior.length > 0 ? `#how-runmat-runs-${slug}-on-the-gpu` : undefined}
        />
      </div>
      <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_220px] pb-8">
        <div className="min-w-0">
          <article className="prose dark:prose-invert max-w-3xl min-w-0 prose-headings:font-semibold prose-h2:mt-8 prose-h2:mb-3 prose-h3:mt-6 prose-h3:mb-2 prose-pre:bg-muted prose-pre:border prose-pre:rounded-md prose-code:bg-muted prose-code:border prose-code:rounded-sm">
            <BuiltinDocRenderer blocks={blocks} />
          </article>
          <div className="mt-12 not-prose max-w-3xl">
            <SandboxCta source={`builtin-docs-${slug}`} />
          </div>
        </div>
        <BuiltinsHeadingsNav toc={toc} />
      </div>
    </>
  );
}

type TocEntry = { id: string; text: string; depth: number };

type FenceInfo = {
  language?: string;
  runnable?: boolean;
};

function renderBuiltinDocBlocks(doc: BuiltinDocEntry, allSiblings: { name: string; slug: string; category: string[] }[]): BuiltinDocBlock[] {
  const blocks: BuiltinDocBlock[] = [];
  const title = doc.title.trim();
  const descriptionHeader = doc.summary?.trim()
    ? `\`${title}\` — ${doc.summary.trim()}`
    : `\`${title}\``;
  const behaviorHeader = `How \`${title}\` works`;
  const examplesHeader = doc.examples && doc.examples.length === 1
    ? 'Example'
    : 'Examples';

  if (hasText(doc.description)) {
    blocks.push(createHeading(1, parseInline(descriptionHeader)));
    blocks.push(...parseMarkdownBlocks(doc.description, { splitListDashes: true }));
  }

  if (doc.syntax?.example?.input) {
    blocks.push(createHeading(2, parseInline(toSentenceCase('syntax'))));
    blocks.push(createCodeBlock(doc.syntax.example.input, { language: 'matlab' }));
    const syntaxList = renderListBlock(doc.syntax.points);
    if (syntaxList) blocks.push(syntaxList);
  }

  if (doc.behaviors && doc.behaviors.length > 0) {
    blocks.push(createHeading(2, parseInline(behaviorHeader)));
    const behaviorList = renderListBlock(doc.behaviors);
    if (behaviorList) blocks.push(behaviorList);
  }

  if (doc.options && doc.options.length > 0) {
    blocks.push(createHeading(2, parseInline(toSentenceCase('options'))));
    const optionsList = renderListBlock(doc.options);
    if (optionsList) blocks.push(optionsList);
  }

  if (doc.jsonencode_options && doc.jsonencode_options.length > 0) {
    blocks.push(createHeading(2, parseInline(toSentenceCase('jsonencode_options'))));
    blocks.push(renderJsonEncodeOptionsTable(doc.jsonencode_options));
  }

  if (doc.gpu_behavior && doc.gpu_behavior.length > 0) {
    blocks.push(createHeading(2, parseInline(`How RunMat runs \`${title}\` on the GPU`)));
    blocks.push(...linkKnownTerms(renderParagraphBlocks(doc.gpu_behavior)));
  }

  if (hasText(doc.gpu_residency)) {
    blocks.push(createHeading(2, parseInline('GPU memory and residency')));
    blocks.push(...linkKnownTerms(parseMarkdownBlocks(doc.gpu_residency)));
  }

  if (doc.examples && doc.examples.length > 0) {
    blocks.push(createHeading(2, parseInline(examplesHeader)));
    blocks.push(...renderExamples(doc.examples));
  }

  if (doc.validation) {
    blocks.push(...renderValidation(doc.validation, title));
  }

  if (doc.faqs && doc.faqs.length > 0) {
    blocks.push(createHeading(2, parseInline('FAQ')));
    blocks.push(...renderFaqs(doc.faqs));
  }

  {
    const isGuideLink = (url: string) => url.startsWith('/docs/') || url.startsWith('/blog/');
    const guideLinks = (doc.links ?? []).filter(l => l.url?.trim() && l.label?.trim() && isGuideLink(l.url.trim()));

    const rawCat = doc.category ?? '';
    const parentCat = getCategoryParent(rawCat);
    const groups = groupSiblingsBySubcategory(rawCat, allSiblings);

    if (groups.length > 0) {
      const parentLabel = formatCategoryPart(parentCat);
      blocks.push(createHeading(2, parseInline(`Related ${parentLabel} functions`)));

      const groupBlocks: BuiltinDocBlock[][] = [];
      for (const group of groups) {
        const col: BuiltinDocBlock[] = [];
        if (groups.length > 1 && group.label) {
          col.push(createHeading(4, parseInline(group.label)));
        }
        const linkNodes: BuiltinDocInlineNode[] = [];
        group.fns.forEach((fn, i) => {
          if (i > 0) linkNodes.push(textNode(' · '));
          linkNodes.push({ type: 'link', label: [textNode(fn.name)], href: `/docs/reference/builtins/${fn.slug}` });
        });
        col.push({ type: 'paragraph', content: linkNodes });
        groupBlocks.push(col);
      }

      const innerChildren: BuiltinDocBlock[] = [];
      if (groupBlocks.length > 1) {
        innerChildren.push({ type: 'columns', cols: groupBlocks });
      } else {
        for (const col of groupBlocks) innerChildren.push(...col);
      }

      blocks.push({ type: 'section', children: innerChildren, className: 'my-4 rounded-lg border border-border bg-muted/40 px-5 py-4' });
    }

    if (guideLinks.length > 0) {
      const guideHeading = rawCat === 'plotting' ? 'More plotting resources' : 'Related guides';
      blocks.push(createHeading(2, parseInline(guideHeading)));
      blocks.push({
        type: 'list',
        ordered: false,
        items: guideLinks.map(link => [
          { type: 'link' as const, label: parseInline(link.label.trim()), href: link.url.trim() },
        ]),
      });
    }
  }

  if (doc.source && hasText(doc.source.url)) {
    blocks.push(createHeading(2, parseInline('Open-source implementation')));
    blocks.push({
      type: 'paragraph',
      content: parseInline(
        `Unlike proprietary runtimes, every RunMat function is open-source. Read exactly how \`${title}\` works, line by line, in Rust.`
      ),
    });
    const sourceList = renderSourceSection(doc.source, title);
    if (sourceList) blocks.push(sourceList);
  }

  blocks.push(createHeading(2, parseInline('About RunMat')));
  blocks.push({
    type: 'paragraph',
    content: parseInline(
      'RunMat is an open-source runtime that executes MATLAB-syntax code — faster, on any GPU, with no license required.'
    ),
  });
  blocks.push({
    type: 'list',
    ordered: false,
    items: [
      parseInline(
        'Simulations that took hours now take minutes. RunMat automatically optimizes your math for GPU execution on Apple, Nvidia, and AMD hardware. No code changes needed.'
      ),
      parseInline(
        'Start running code in seconds. Open the browser sandbox or download a single binary. No license server, no IT ticket, no setup.'
      ),
      parseInline(
        'A full development environment. GPU-accelerated 2D and 3D plotting, automatic versioning on every save, and a browser IDE you can share with a link.'
      ),
    ],
  });
  blocks.push({
    type: 'paragraph',
    content: [
      { type: 'link', label: [textNode('Getting started')], href: '/docs/getting-started' },
      textNode(' · '),
      { type: 'link', label: [textNode('Benchmarks')], href: '/benchmarks' },
      textNode(' · '),
      { type: 'link', label: [textNode('Pricing')], href: '/pricing' },
    ],
  });

  return blocks;
}

function renderExamples(examples: BuiltinDocExample[]): BuiltinDocBlock[] {
  const blocks: BuiltinDocBlock[] = [];
  for (const example of examples) {
    const description = example.description?.trim();
    if (description) {
      blocks.push(createHeading(3, parseInline(description)));
    }
    blocks.push(createCodeBlock(example.input, { language: 'matlab', runnable: true }));
    if (hasText(example.output)) {
      blocks.push({ type: 'paragraph', content: [textNode('Expected output:')] });
      blocks.push(createCodeBlock(example.output, { language: 'matlab' }));
    }
    const exampleImage = example.image_webp || example.image;
    if (exampleImage) {
      blocks.push({
        type: 'image',
        src: exampleImage,
        alt: description || 'Example output',
        caption: 'Expected output:',
      });
    }
  }
  return blocks;
}

function renderValidation(validation: BuiltinDocValidation, title: string): BuiltinDocBlock[] {
  const blocks: BuiltinDocBlock[] = [];
  blocks.push(createHeading(2, parseInline(`How RunMat validates \`${title}\``)));
  if (hasText(validation.summary)) {
    blocks.push(...parseMarkdownBlocks(validation.summary));
  }
  const listItems: BuiltinDocInlineNode[][] = [];
  if (validation.implementation?.url && validation.implementation?.label) {
    listItems.push([
      textNode('Implementation: '),
      {
        type: 'link',
        label: [textNode(validation.implementation.label)],
        href: validation.implementation.url,
      },
    ]);
  }
  if (validation.parity_test?.url && validation.parity_test?.label) {
    listItems.push([
      textNode('Parity test: '),
      {
        type: 'link',
        label: [textNode(validation.parity_test.label)],
        href: validation.parity_test.url,
      },
    ]);
  }
  if (hasText(validation.tolerance)) {
    listItems.push([textNode(`Tolerance: ${validation.tolerance}`)]);
  }
  if (listItems.length > 0) {
    blocks.push({ type: 'list', ordered: false, items: listItems });
  }
  if (hasText(validation.limitations)) {
    blocks.push(...parseMarkdownBlocks(validation.limitations));
  }
  blocks.push({
    type: 'paragraph',
    content: [
      textNode('See '),
      { type: 'link', label: [textNode('Correctness & Trust')], href: '/docs/correctness' },
      textNode(' for the full methodology and coverage table.'),
    ],
  });
  return blocks;
}

function renderFaqs(faqs: BuiltinDocFAQ[]): BuiltinDocBlock[] {
  const blocks: BuiltinDocBlock[] = [];
  for (const faq of faqs) {
    blocks.push(createHeading(3, parseInline(faq.question)));
    blocks.push(...parseMarkdownBlocks(faq.answer));
  }
  return blocks;
}

function renderSourceSection(source: BuiltinDocLink, title?: string): BuiltinDocBlock | null {
  const url = resolveSourceUrl(source.url);
  const label = title ? `View ${title}.rs on GitHub` : (source.label?.trim() || url);
  const items: BuiltinDocInlineNode[][] = [];
  if (url) {
    items.push([
      { type: 'link', label: [textNode(label)], href: url },
    ]);
  }
  items.push([
    { type: 'link', label: [textNode('Learn how the runtime works')], href: '/docs/architecture' },
  ]);
  items.push([
    textNode('Found a bug? '),
    { type: 'link', label: [textNode('Open an issue')], href: 'https://github.com/runmat-org/runmat/issues/new' },
    textNode(' with a minimal reproduction.'),
  ]);
  if (items.length === 0) return null;
  return { type: 'list', ordered: false, items };
}

function renderJsonEncodeOptionsTable(options: BuiltinDocJsonEncodeOption[]): BuiltinDocBlock {
  return {
    type: 'table',
    headers: [
      [textNode('Name')],
      [textNode('Type')],
      [textNode('Default')],
      [textNode('Description')],
    ],
    rows: options.map(option => [
      [inlineCodeOrText(option.name)],
      parseInline(option.type),
      [inlineCodeOrText(option.default)],
      parseInline(option.description),
    ]),
  };
}

function renderListBlock(items: string[] | undefined): BuiltinDocBlock | null {
  if (!items || items.length === 0) return null;
  return {
    type: 'list',
    ordered: false,
    items: items
      .map((item) => item.trim())
      .filter(Boolean)
      .map((item) => parseInline(item)),
  };
}

function renderParagraphBlocks(items: string[]): BuiltinDocBlock[] {
  const paragraphs: string[] = [];
  for (const item of items) {
    const lines = splitMarkdown(item).map((line) => line.trim()).filter(Boolean);
    paragraphs.push(...lines);
  }
  return paragraphs.map((paragraph) => ({
    type: 'paragraph',
    content: parseInline(paragraph),
  }));
}

function createCodeBlock(code: string, info: FenceInfo): BuiltinDocBlock {
  const lines = splitMarkdown(code);
  return {
    type: 'code',
    language: info.language ?? 'plaintext',
    runnable: info.runnable,
    content: lines.join('\n'),
  };
}

function parseMarkdownBlocks(text: string, options?: { splitListDashes?: boolean }): BuiltinDocBlock[] {
  const lines = splitMarkdown(text);
  if (lines.length === 0) return [];

  const blocks: BuiltinDocBlock[] = [];
  let paragraphLines: string[] = [];
  let listItems: string[] = [];
  let inFence = false;
  let fenceLang = '';
  let fenceLines: string[] = [];

  const flushParagraph = () => {
    if (paragraphLines.length === 0) return;
    const paragraphText = paragraphLines.join(' ').trim();
    if (paragraphText) {
      blocks.push({ type: 'paragraph', content: parseInline(paragraphText) });
    }
    paragraphLines = [];
  };

  const flushList = () => {
    if (listItems.length === 0) return;
    blocks.push({
      type: 'list',
      ordered: false,
      items: listItems.map((item) => parseInline(item)),
    });
    listItems = [];
  };

  const flushFence = () => {
    const info = normalizeFenceInfo(fenceLang);
    blocks.push({
      type: 'code',
      language: info.language ?? 'plaintext',
      runnable: info.runnable,
      content: fenceLines.join('\n'),
    });
    fenceLines = [];
    fenceLang = '';
  };

  for (const rawLine of lines) {
    const line = rawLine.replace(/\s+$/, '');
    const trimmed = line.trim();

    if (trimmed.startsWith('```')) {
      if (inFence) {
        flushFence();
        inFence = false;
      } else {
        flushParagraph();
        flushList();
        inFence = true;
        fenceLang = trimmed.slice(3).trim();
      }
      continue;
    }

    if (inFence) {
      fenceLines.push(line);
      continue;
    }

    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    const headingMatch = /^(#{1,6})\s+(.+)$/.exec(trimmed);
    if (headingMatch) {
      flushParagraph();
      flushList();
      const level = headingMatch[1].length as 1 | 2 | 3 | 4 | 5 | 6;
      blocks.push(createHeading(level, parseInline(headingMatch[2])));
      continue;
    }

    if (trimmed.startsWith('- ')) {
      flushParagraph();
      const content = trimmed.slice(2);
      listItems.push(...splitListItemContent(content, options?.splitListDashes));
      continue;
    }

    paragraphLines.push(trimmed);
  }

  if (inFence) flushFence();
  flushParagraph();
  flushList();

  return blocks;
}

function splitListItemContent(content: string, shouldSplit?: boolean): string[] {
  const trimmed = content.trim();
  if (!shouldSplit) return [trimmed];
  const parts = trimmed
    .split(' - ')
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length > 1) return parts;
  return [trimmed];
}

function renderSeeAlsoLinks(links: BuiltinDocLink[]): BuiltinDocInlineNode[] {
  const nodes: BuiltinDocInlineNode[] = [];
  const entries = links
    .map((link) => {
      const url = link.url?.trim();
      const label = link.label?.trim() || url;
      if (!url || !label) return null;
      return { url, label };
    })
    .filter((entry): entry is { url: string; label: string } => Boolean(entry));

  entries.forEach((entry, index) => {
    if (index > 0) nodes.push(textNode(', '));
    nodes.push({ type: 'link', label: parseInline(entry.label), href: entry.url });
  });

  return nodes;
}

function resolveSourceUrl(url: string): string {
  const trimmed = url.trim();
  if (!trimmed) return '';
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  if (trimmed.startsWith('/')) return trimmed;
  return `https://github.com/runmat-org/runmat/blob/main/${trimmed}`;
}

function toSentenceCase(value: string): string {
  const spaced = value.replace(/_/g, ' ');
  return spaced.charAt(0).toUpperCase() + spaced.slice(1).toLowerCase();
}

function toGPUCase(value: string): string {
  const parts = value.split('_');
  return parts.map(p => (p.toLowerCase() === 'gpu') ? p.toUpperCase() : p.toLowerCase()).join(' ');
}

function hasText(value?: string | null): value is string {
  return typeof value === 'string' && value.trim().length > 0;
}



const PLOTTING_SUBGROUPS: Record<string, string> = {
  plot: '2D Charts', bar: '2D Charts', scatter: '2D Charts', histogram: '2D Charts',
  hist: '2D Charts', pie: '2D Charts', stairs: '2D Charts', stem: '2D Charts',
  area: '2D Charts', errorbar: '2D Charts', loglog: '2D Charts',
  semilogx: '2D Charts', semilogy: '2D Charts',
  surf: '3D & Surface', surfc: '3D & Surface', mesh: '3D & Surface',
  meshc: '3D & Surface', plot3: '3D & Surface', scatter3: '3D & Surface',
  contour: '3D & Surface', contourf: '3D & Surface', quiver: '3D & Surface',
  image: 'Images', imagesc: 'Images',
  axis: 'Axes & Layout', subplot: 'Axes & Layout', grid: 'Axes & Layout',
  box: 'Axes & Layout', view: 'Axes & Layout', sgtitle: 'Axes & Layout', zlabel: 'Axes & Layout',
  colormap: 'Appearance', colorbar: 'Appearance', shading: 'Appearance', legend: 'Appearance',
  get: 'Handle Access', set: 'Handle Access',
};

type RelatedGroup = { label: string | null; fns: { name: string; slug: string }[] };

function getCategoryParent(rawCat: string): string {
  const parts = rawCat.split('/');
  if (parts.length <= 1) return rawCat;
  return parts.slice(0, -1).join('/');
}

function getCategorySub(rawCat: string): string {
  const parts = rawCat.split('/');
  return parts.length > 1 ? parts[parts.length - 1] : '';
}

function formatCategoryPart(part: string): string {
  const last = part.includes('/') ? part.split('/').pop()! : part;
  return last
    .split(/[-_]/)
    .map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(' ');
}

function groupSiblingsBySubcategory(
  rawCat: string,
  allSiblings: { name: string; slug: string; category: string[] }[],
): RelatedGroup[] {
  const parent = getCategoryParent(rawCat);
  const ownSub = getCategorySub(rawCat);

  if (rawCat === 'plotting') {
    const plotFns = allSiblings.filter(s => s.category[0] === 'plotting');
    const grouped = new Map<string, { name: string; slug: string }[]>();
    for (const fn of plotFns) {
      const sub = PLOTTING_SUBGROUPS[fn.slug] ?? 'Other';
      if (!grouped.has(sub)) grouped.set(sub, []);
      grouped.get(sub)!.push(fn);
    }
    const order = ['2D Charts', '3D & Surface', 'Images', 'Axes & Layout', 'Appearance', 'Handle Access', 'Other'];
    return order
      .filter(label => grouped.has(label))
      .map(label => ({ label, fns: grouped.get(label)! }));
  }

  const siblings = allSiblings.filter(s => {
    const sCat = s.category[0] ?? '';
    return sCat.startsWith(parent + '/') || sCat === parent;
  });

  if (siblings.length === 0) return [];

  const grouped = new Map<string, { name: string; slug: string }[]>();
  for (const fn of siblings) {
    const fnCat = fn.category[0] ?? '';
    const sub = getCategorySub(fnCat) || parent;
    if (!grouped.has(sub)) grouped.set(sub, []);
    grouped.get(sub)!.push(fn);
  }

  if (grouped.size <= 1) {
    const fns = [...grouped.values()][0] ?? [];
    return fns.length > 0 ? [{ label: null, fns }] : [];
  }

  const groups: RelatedGroup[] = [];
  if (ownSub && grouped.has(ownSub)) {
    groups.push({ label: formatCategoryPart(ownSub), fns: grouped.get(ownSub)! });
  }
  for (const [sub, fns] of grouped) {
    if (sub === ownSub) continue;
    groups.push({ label: formatCategoryPart(sub), fns });
  }
  return groups;
}

function extractHeadingsFromBlocks(blocks: BuiltinDocBlock[]): TocEntry[] {
  const out: TocEntry[] = [];
  for (const block of blocks) {
    if (block.type === 'section') {
      out.push(...extractHeadingsFromBlocks(block.children));
      continue;
    }
    if (block.type === 'columns') {
      for (const col of block.cols) out.push(...extractHeadingsFromBlocks(col));
      continue;
    }
    if (block.type !== 'heading') continue;
    if (block.level < 2 || block.level > 6) continue;
    const text = inlineToPlainText(block.text).replace(/`/g, '');
    const id = block.id ?? slugifyHeading(text);
    out.push({ id, text, depth: block.level });
  }
  return out;
}

function splitMarkdown(text: string): string[] {
  if (!text || !text.trim()) return [];
  return text.trim().split(/\r?\n/);
}

function normalizeFenceInfo(language: string): FenceInfo {
  if (!language) return { language: 'plaintext' };
  const parts = language.split(':').map((part) => part.trim()).filter(Boolean);
  const base = parts[0] ?? 'plaintext';
  return {
    language: base,
    runnable: parts.includes('runnable'),
  };
}

function parseInline(text: string): BuiltinDocInlineNode[] {
  if (!text) return [];
  const nodes: BuiltinDocInlineNode[] = [];
  let index = 0;

  while (index < text.length) {
    const char = text[index];
    if (char === '`') {
      const end = text.indexOf('`', index + 1);
      if (end !== -1) {
        const value = text.slice(index + 1, end);
        nodes.push({ type: 'code', value });
        index = end + 1;
        continue;
      }
    }

    if (char === '*' && text[index + 1] === '*') {
      const end = text.indexOf('**', index + 2);
      if (end !== -1 && end > index + 2) {
        const inner = text.slice(index + 2, end);
        nodes.push({ type: 'strong', content: parseInline(inner) });
        index = end + 2;
        continue;
      }
    }

    if (char === '[') {
      const closeBracket = text.indexOf(']', index + 1);
      const openParen = closeBracket !== -1 ? text.indexOf('(', closeBracket + 1) : -1;
      const closeParen = openParen !== -1 ? text.indexOf(')', openParen + 1) : -1;
      if (closeBracket !== -1 && openParen === closeBracket + 1 && closeParen !== -1) {
        const label = text.slice(index + 1, closeBracket);
        const href = text.slice(openParen + 1, closeParen).trim();
        nodes.push({ type: 'link', label: parseInline(label), href });
        index = closeParen + 1;
        continue;
      }
    }

    const nextBacktick = text.indexOf('`', index);
    const nextLink = text.indexOf('[', index);
    const nextBold = text.indexOf('**', index);
    let nextIndex = text.length;
    if (nextBacktick !== -1) nextIndex = Math.min(nextIndex, nextBacktick);
    if (nextLink !== -1) nextIndex = Math.min(nextIndex, nextLink);
    if (nextBold !== -1) nextIndex = Math.min(nextIndex, nextBold);
    if (nextIndex === index) {
      nodes.push(textNode(text[index]));
      index += 1;
      continue;
    }
    const value = text.slice(index, nextIndex);
    if (value) nodes.push(textNode(value));
    index = nextIndex;
  }

  return nodes;
}

function inlineToPlainText(nodes: BuiltinDocInlineNode[]): string {
  return nodes.map((node) => {
    if (node.type === 'text' || node.type === 'code') return node.value;
    if (node.type === 'link') return inlineToPlainText(node.label);
    if (node.type === 'strong') return inlineToPlainText(node.content);
    return '';
  }).join('');
}

function createHeading(level: 1 | 2 | 3 | 4 | 5 | 6, text: BuiltinDocInlineNode[]): BuiltinDocBlock {
  const id = slugifyHeading(inlineToPlainText(text).replace(/`/g, ''));
  return { type: 'heading', level, text, id };
}

function textNode(value: string): BuiltinDocInlineNode {
  return { type: 'text', value };
}

function inlineCodeOrText(value: string): BuiltinDocInlineNode {
  const trimmed = String(value ?? '').trim();
  if (!trimmed) return textNode('');
  return { type: 'code', value: trimmed };
}

const GPU_TERM_LINKS: { term: string; href: string }[] = [
  { term: 'GPU acceleration', href: '/docs/accelerate/fusion-intro' },
  { term: 'fusion',           href: '/docs/accelerate/fusion-intro' },
  { term: 'residency',        href: '/docs/accelerate/gpu-behavior' },
  { term: 'gpuArray',         href: '/docs/reference/builtins/gpuarray' },
];

function linkKnownTerms(blocks: BuiltinDocBlock[]): BuiltinDocBlock[] {
  const linked = new Set<string>();
  return blocks.map((block) => {
    if (block.type === 'section') return { ...block, children: linkKnownTerms(block.children) };
    if (block.type === 'columns') return { ...block, cols: block.cols.map(col => linkKnownTerms(col)) };
    if (block.type !== 'paragraph') return block;
    return { ...block, content: linkInlineNodes(block.content, linked) };
  });
}

function linkInlineNodes(
  nodes: BuiltinDocInlineNode[],
  linked: Set<string>,
): BuiltinDocInlineNode[] {
  const result: BuiltinDocInlineNode[] = [];
  for (const node of nodes) {
    if (node.type !== 'text') {
      result.push(node);
      continue;
    }
    let remaining = node.value;
    while (remaining) {
      let earliest: { idx: number; term: string; href: string } | null = null;
      for (const { term, href } of GPU_TERM_LINKS) {
        if (linked.has(term)) continue;
        const idx = remaining.indexOf(term);
        if (idx !== -1 && (!earliest || idx < earliest.idx)) {
          earliest = { idx, term, href };
        }
      }
      if (!earliest) {
        result.push(textNode(remaining));
        break;
      }
      if (earliest.idx > 0) {
        result.push(textNode(remaining.slice(0, earliest.idx)));
      }
      result.push({
        type: 'link',
        label: [textNode(earliest.term)],
        href: earliest.href,
      });
      linked.add(earliest.term);
      remaining = remaining.slice(earliest.idx + earliest.term.length);
    }
  }
  return result;
}
