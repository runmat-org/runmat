import type { Metadata } from 'next';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import {
  getBuiltinDocBySlug,
  getBuiltinMetadata,
  loadBuiltins,
  type BuiltinDocEntry,
  type BuiltinDocExample,
  type BuiltinDocLink,
  type BuiltinDocFAQ,
  type BuiltinDocJsonEncodeOption,
} from '@/lib/builtins';
import { BuiltinDocRenderer, type BuiltinDocBlock, type BuiltinDocInlineNode } from '@/components/BuiltinDocRenderer';
import { slugifyHeading } from '@/lib/utils';
import { builtinMetadataForSlug, builtinJsonLD } from './meta';
import { BuiltinMetadataChips } from '@/components/BuiltinMetadataChips';

export const dynamic = 'force-static';

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
  if (!b) return notFound();
  const doc = getBuiltinDocBySlug(slug);
  if (!doc) return notFound();
  const blocks = renderBuiltinDocBlocks(doc);
  const toc = extractHeadingsFromBlocks(blocks);
  const metadata = getBuiltinMetadata(b);

  return (
    <>
    <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: builtinJsonLD(slug) }}
    />
      <div className="container mx-auto px-4 md:px-6 pt-8">
        <p className="mb-4 text-muted-foreground leading-relaxed break-words">
          <Link href="/docs/matlab-function-reference" className="text-muted-foreground hover:text-foreground transition-colors">
            View all functions
          </Link>
        </p>
        <BuiltinMetadataChips metadata={metadata} />
      </div>
      <div className="container mx-auto px-4 md:px-6 pb-8">
      <div className="grid gap-8 lg:grid-cols-[1fr_280px]">
        <article className="prose dark:prose-invert max-w-none prose-headings:font-semibold prose-h2:mt-8 prose-h2:mb-3 prose-h3:mt-6 prose-h3:mb-2 prose-pre:bg-muted prose-pre:border prose-pre:rounded-md prose-code:bg-muted prose-code:border prose-code:rounded-sm">
          <BuiltinDocRenderer blocks={blocks} />
        </article>
        {toc.length > 0 && (
          <aside className="hidden lg:block sticky top-24 h-max text-sm">
            <div className="mb-2 font-semibold text-foreground/80">On this page</div>
            <ul className="space-y-1 text-muted-foreground">
              {toc.map((t) => (
                <li key={t.id} className={t.depth === 3 ? 'pl-3' : ''}>
                  <a href={`#${t.id}`} className="hover:text-foreground">{t.text}</a>
                </li>
              ))}
            </ul>
          </aside>
        )}
      </div>
    </div>
    </>
  );
}

type TocEntry = { id: string; text: string; depth: number };

type FenceInfo = {
  language?: string;
  runnable?: boolean;
};

function renderBuiltinDocBlocks(doc: BuiltinDocEntry): BuiltinDocBlock[] {
  const blocks: BuiltinDocBlock[] = [];
  const title = doc.title.trim();
  const descriptionHeader = doc.summary?.trim()
    ? `\`${title}\` — ${doc.summary.trim()}`
    : `\`${title}\``;
  const behaviorHeader = `How does the \`${title}\` function behave in MATLAB / RunMat?`;
  const examplesHeader = doc.examples && doc.examples.length === 1
    ? `Example of using \`${title}\` in MATLAB / RunMat`
    : `Examples of using \`${title}\` in MATLAB / RunMat`;

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
    blocks.push(createHeading(2, parseInline(toGPUCase('gpu_behavior'))));
    blocks.push(...renderParagraphBlocks(doc.gpu_behavior));
  }

  if (hasText(doc.gpu_residency)) {
    blocks.push(createHeading(2, parseInline(toGPUCase('gpu_residency'))));
    blocks.push(...parseMarkdownBlocks(doc.gpu_residency));
  }

  if (doc.examples && doc.examples.length > 0) {
    blocks.push(createHeading(2, parseInline(examplesHeader)));
    blocks.push(...renderExamples(doc.examples));
  }

  if (doc.faqs && doc.faqs.length > 0) {
    blocks.push(createHeading(2, parseInline('FAQ')));
    blocks.push(...renderFaqs(doc.faqs));
  }

  if (doc.links && doc.links.length > 0) {
    const linkInline = renderSeeAlsoLinks(doc.links);
    if (linkInline.length > 0) {
      blocks.push(createHeading(2, parseInline('See also')));
      blocks.push({ type: 'paragraph', content: linkInline });
    }
  }

  if (doc.source && hasText(doc.source.url)) {
    blocks.push(createHeading(2, parseInline('Source & Feedback')));
    const sourceList = renderSourceSection(doc.source);
    if (sourceList) blocks.push(sourceList);
  }

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
  }
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

function renderSourceSection(source: BuiltinDocLink): BuiltinDocBlock | null {
  const url = resolveSourceUrl(source.url);
  const label = source.label && source.label.trim() ? source.label.trim() : url;
  const items: BuiltinDocInlineNode[][] = [];
  if (url) {
    items.push([
      textNode('Source code: '),
      { type: 'link', label: [textNode(label)], href: url },
    ]);
  }
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

function extractHeadingsFromBlocks(blocks: BuiltinDocBlock[]): TocEntry[] {
  const out: TocEntry[] = [];
  for (const block of blocks) {
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
    let nextIndex = text.length;
    if (nextBacktick !== -1) nextIndex = Math.min(nextIndex, nextBacktick);
    if (nextLink !== -1) nextIndex = Math.min(nextIndex, nextLink);
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
