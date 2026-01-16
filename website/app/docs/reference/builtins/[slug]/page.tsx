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
import { MarkdownRenderer } from '@/components/MarkdownRenderer';
import { slugifyHeading } from '@/lib/utils';
import { builtinMetadataForSlug } from './meta';
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
  const source = renderBuiltinDoc(doc);
  const toc = extractHeadings(source);
  const metadata = getBuiltinMetadata(b);
  return (
    <>
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
          <MarkdownRenderer source={source} />
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

function renderBuiltinDoc(doc: BuiltinDocEntry): string {
  const lines: string[] = [];
  const title = doc.title.trim();
  const descriptionHeader = `# What does the \`${title}\` function do in MATLAB / RunMat?`;
  const behaviorHeader = `How does the \`${title}\` function behave in MATLAB / RunMat?`;
  const examplesHeader = doc.examples && doc.examples.length === 1
    ? `Example of using \`${title}\` in MATLAB / RunMat`
    : `Examples of using \`${title}\` in MATLAB / RunMat`;

  if (hasText(doc.description)) {
    lines.push(descriptionHeader);
    lines.push('');
    lines.push(...renderDescriptionBlock(doc.description));
  }

  if (doc.syntax?.example?.input) {
    pushSection(lines, toSentenceCase('syntax'), [
      ...renderCodeBlock(doc.syntax.example.input),
      ...renderList(doc.syntax.points, true),
    ]);
  }

  if (doc.behaviors && doc.behaviors.length > 0) {
    pushSection(lines, behaviorHeader, renderList(doc.behaviors, true));
  }

  if (doc.options && doc.options.length > 0) {
    pushSection(lines, toSentenceCase('options'), renderList(doc.options, true));
  }

  if (doc.jsonencode_options && doc.jsonencode_options.length > 0) {
    pushSection(lines, toSentenceCase('jsonencode_options'), renderJsonEncodeOptions(doc.jsonencode_options));
  }

  if (doc.gpu_behavior && doc.gpu_behavior.length > 0) {
    pushSection(lines, toGPUCase('gpu_behavior'), renderParagraphs(doc.gpu_behavior));
  }

  if (hasText(doc.gpu_residency)) {
    pushSection(lines, toGPUCase('gpu_residency'), renderMarkdownBlock(doc.gpu_residency));
  }

  if (doc.examples && doc.examples.length > 0) {
    const exampleLines: string[] = [];
    for (const example of doc.examples) {
      if (exampleLines.length > 0) exampleLines.push('');
      exampleLines.push(...renderExample(example));
    }
    pushSection(lines, examplesHeader, exampleLines);
  }

  if (doc.faqs && doc.faqs.length > 0) {
    pushSection(lines, 'FAQ', renderFaqs(doc.faqs));
  }

  if (doc.links && doc.links.length > 0) {
    const formattedLinks = formatLinks(doc.links);
    if (formattedLinks) {
      pushSection(lines, 'See also', [formattedLinks]);
    }
  }

  if (doc.source && hasText(doc.source.url)) {
    pushSection(lines, 'Source & Feedback', renderSourceSection(doc.source));
  }

  return lines.join('\n');
}

function renderExample(example: BuiltinDocExample): string[] {
  const lines: string[] = [];
  const description = example.description?.trim();
  if (description) {
    lines.push(`### ${escapeMarkdownText(description)}`);
    lines.push('');
  }
  lines.push(...renderCodeBlock(example.input));
  if (hasText(example.output)) {
    lines.push('');
    lines.push('Expected output:');
    lines.push('');
    lines.push(...renderCodeBlock(example.output));
  }
  return lines;
}

function renderFaqs(faqs: BuiltinDocFAQ[]): string[] {
  const lines: string[] = [];
  for (const faq of faqs) {
    if (lines.length > 0) lines.push('');
    lines.push(`### ${escapeMarkdownText(faq.question)}`);
    lines.push('');
    lines.push(...renderMarkdownBlock(faq.answer));
  }
  return lines;
}

function renderSourceSection(source: BuiltinDocLink): string[] {
  const lines: string[] = [];
  const url = resolveSourceUrl(source.url);
  const label = source.label && source.label.trim() ? source.label.trim() : url;
  if (url) {
    lines.push(`- Source code: [${label}](${url})`);
  }
  lines.push('- Found a bug? [Open an issue](https://github.com/runmat-org/runmat/issues/new) with a minimal reproduction.');
  return lines;
}

function renderJsonEncodeOptions(options: BuiltinDocJsonEncodeOption[]): string[] {
  const lines: string[] = [];
  lines.push('| Name | Type | Default | Description |');
  lines.push('| ---- | ---- | ------- | ----------- |');
  for (const option of options) {
    const name = wrapInlineCode(option.name);
    const defaultValue = wrapInlineCode(option.default);
    lines.push(`| ${escapeTableCell(name)} | ${escapeTableCell(option.type)} | ${escapeTableCell(defaultValue)} | ${escapeTableCell(option.description)} |`);
  }
  return lines;
}

function renderList(items: string[] | undefined, prefixDash = false): string[] {
  if (!items || items.length === 0) return [];
  if (!prefixDash) {
    return items
      .map((item) => escapeMarkdownText(item.trim()))
      .filter(Boolean);
  }
  return items
    .map((item) => escapeMarkdownText(item.trim()))
    .filter(Boolean)
    .map((item) => `- ${item}`);
}

function renderParagraphs(items: string[]): string[] {
  const paragraphs: string[] = [];
  for (const item of items) {
    const lines = splitMarkdown(item).map((line) => line.trim()).filter(Boolean);
    paragraphs.push(...lines);
  }
  const out: string[] = [];
  for (const paragraph of paragraphs) {
    if (out.length > 0) out.push('');
    out.push(escapeMarkdownText(paragraph));
  }
  return out;
}

function renderCodeBlock(code: string): string[] {
  const lines: string[] = [];
  lines.push('```matlab');
  lines.push(...splitMarkdown(code));
  lines.push('```');
  return lines;
}

function renderMarkdownBlock(text: string): string[] {
  return splitMarkdown(text).map((line) => escapeMarkdownText(line));
}

function renderDescriptionBlock(text: string): string[] {
  const rawLines = splitMarkdown(text);
  const out: string[] = [];
  let inFence = false;
  for (const rawLine of rawLines) {
    const line = rawLine.replace(/\s+$/, '');
    const trimmed = line.trim();
    if (trimmed.startsWith('```')) {
      inFence = !inFence;
      out.push(line);
      continue;
    }
    if (!inFence && line.startsWith('- ')) {
      const content = line.slice(2);
      const parts = content.split(' - ').map((part) => part.trim()).filter(Boolean);
      if (parts.length > 1) {
        for (const part of parts) {
          out.push(`- ${part}`);
        }
        continue;
      }
    }
    out.push(line);
  }
  return out.map((line) => escapeMarkdownText(line));
}

function splitMarkdown(text: string): string[] {
  return text.trim().split(/\r?\n/);
}

function pushMarkdown(lines: string[], text: string): void {
  const block = renderMarkdownBlock(text);
  if (block.length === 0) return;
  lines.push(...block);
}

function pushSection(lines: string[], header: string, body: string[]): void {
  if (!body || body.length === 0) return;
  if (lines.length > 0) lines.push('');
  lines.push(`## ${header}`);
  lines.push('');
  lines.push(...body);
}

function formatLinks(links: BuiltinDocLink[]): string {
  return links
    .map((link) => {
      const url = link.url?.trim();
      const label = link.label?.trim() || url;
      if (!url || !label) return '';
      return `[${label}](${url})`;
    })
    .filter(Boolean)
    .join(', ');
}

function resolveSourceUrl(url: string): string {
  const trimmed = url.trim();
  if (!trimmed) return '';
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  if (trimmed.startsWith('/')) return trimmed;
  return `https://github.com/runmat-org/runmat/blob/main/${trimmed}`;
}

function escapeTableCell(value: string): string {
  return String(value ?? '').replace(/\|/g, '\\|');
}

function wrapInlineCode(value: string): string {
  const trimmed = String(value ?? '').trim();
  if (!trimmed) return '';
  if (trimmed.startsWith('`') && trimmed.endsWith('`')) return trimmed;
  return `\`${trimmed}\``;
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

function escapeMarkdownText(value: string): string {
  let result = '';
  let inCode = false;
  for (let i = 0; i < value.length; i += 1) {
    const char = value[i];
    if (char === '`') {
      inCode = !inCode;
      result += char;
      continue;
    }
    if (!inCode && (char === '<' || char === '>')) {
      result += char === '<' ? '&lt;' : '&gt;';
      continue;
    }
    result += char;
  }
  return result;
}

function extractHeadings(md: string): { id: string; text: string; depth: number }[] {
  const lines = md.split(/\r?\n/);
  const out: { id: string; text: string; depth: number }[] = [];
  let inFence = false;
  for (const raw of lines) {
    const line = raw.trim();
    if (/^```/.test(line)) { inFence = !inFence; continue; }
    if (inFence) continue;
    const m = /^(#{2,6})\s+(.+)$/.exec(line);
    if (!m) continue;
    const depth = m[1].length; // 2..6
    const text = m[2].replace(/`/g, '');
    const id = slugifyHeading(text);
    out.push({ id, text, depth });
  }
  return out;
}
