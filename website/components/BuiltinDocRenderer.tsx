import React from 'react';
import { slugifyHeading } from '@/lib/utils';
import { HeadingAnchor } from '@/components/HeadingAnchor';
import { TryInBrowserButton } from '@/components/TryInBrowserButton';
import { MermaidDiagram } from '@/components/MermaidDiagram';
import Prism from 'prismjs';
import 'prismjs/components/prism-matlab';

export type BuiltinDocInlineNode =
  | { type: 'text'; value: string }
  | { type: 'code'; value: string }
  | { type: 'link'; label: BuiltinDocInlineNode[]; href: string }
  | { type: 'strong'; content: BuiltinDocInlineNode[] };

export type BuiltinDocBlock =
  | { type: 'heading'; level: 1 | 2 | 3 | 4 | 5 | 6; text: BuiltinDocInlineNode[]; id?: string }
  | { type: 'paragraph'; content: BuiltinDocInlineNode[] }
  | { type: 'list'; ordered: boolean; items: BuiltinDocInlineNode[][] }
  | { type: 'code'; language?: string; runnable?: boolean; content: string }
  | { type: 'table'; headers: BuiltinDocInlineNode[][]; rows: BuiltinDocInlineNode[][][] }
  | { type: 'image'; src: string; alt: string; caption?: string }
  | { type: 'link-grid'; items: { label: string; href: string; thumbnail?: string }[] }
  | { type: 'divider' }
  | { type: 'section'; children: BuiltinDocBlock[]; className?: string }
  | { type: 'columns'; cols: BuiltinDocBlock[][]; className?: string }
  | { type: 'faq'; items: { question: BuiltinDocInlineNode[]; answerBlocks: BuiltinDocBlock[] }[] };

type BuiltinDocRendererProps = {
  blocks: BuiltinDocBlock[];
};

export function BuiltinDocRenderer({ blocks }: BuiltinDocRendererProps) {
  return (
    <>
      {blocks.map((block, index) => (
        <React.Fragment key={`${block.type}-${index}`}>
          {renderBlock(block)}
        </React.Fragment>
      ))}
    </>
  );
}

function renderBlock(block: BuiltinDocBlock): React.ReactNode {
  switch (block.type) {
    case 'heading':
      return renderHeading(block);
    case 'paragraph':
      return (
        <p className="my-4 text-foreground text-[0.938rem] leading-relaxed break-words">
          {renderInlineNodes(block.content)}
        </p>
      );
    case 'list':
      return block.ordered ? (
        <ol className="my-4 ml-6 space-y-1.5 list-decimal marker:text-muted-foreground break-words">
          {block.items.map((item, index) => (
            <li key={index} className="text-foreground text-[0.938rem] leading-relaxed pl-2 break-words">
              {renderInlineNodes(item)}
            </li>
          ))}
        </ol>
      ) : (
        <ul className="my-4 ml-6 space-y-1.5 list-disc marker:text-muted-foreground break-words">
          {block.items.map((item, index) => (
            <li key={index} className="text-foreground text-[0.938rem] leading-relaxed pl-2 break-words">
              {renderInlineNodes(item)}
            </li>
          ))}
        </ul>
      );
    case 'code':
      return renderCodeBlock(block);
    case 'table':
      return (
        <div className="my-8 mx-0 overflow-x-auto max-w-full">
          <table className="w-full caption-bottom text-sm table-mobile-wrap">
            <thead className="[&_tr]:border-b">
              <tr className="border-b border-border transition-colors">
                {block.headers.map((header, index) => (
                  <th
                    key={index}
                    className="px-4 py-3 text-left align-middle text-sm font-medium text-muted-foreground"
                  >
                    {renderInlineNodes(header)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="[&_tr:last-child]:border-0">
              {block.rows.map((row, rowIndex) => (
                <tr key={rowIndex} className="border-b border-border transition-colors">
                  {row.map((cell, cellIndex) => (
                    <td
                      key={cellIndex}
                      className="px-4 py-3 align-middle text-sm text-foreground"
                    >
                      {renderInlineNodes(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    case 'image':
      return (
        <figure className="my-8">
          {block.caption && (
            <figcaption className="mb-2 text-sm text-muted-foreground">
              {block.caption}
            </figcaption>
          )}
          <img
            src={block.src}
            alt={block.alt}
            loading="lazy"
            className="rounded-lg border border-border w-full max-w-2xl"
          />
        </figure>
      );
    case 'link-grid':
      return (
        <div className="my-8 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
          {block.items.map((item, index) => (
            <a
              key={index}
              href={item.href}
              className="group flex flex-col items-center gap-2 rounded-lg border border-border p-3 hover:border-blue-500/50 hover:bg-muted/50 transition-colors"
            >
              {item.thumbnail && (
                <img
                  src={item.thumbnail}
                  alt={`${item.label} preview`}
                  loading="lazy"
                  className="rounded w-full h-auto aspect-[5/3] object-cover"
                />
              )}
              <span className="text-sm font-mono text-foreground group-hover:text-blue-500 transition-colors">
                {item.label}
              </span>
            </a>
          ))}
        </div>
      );
    case 'divider':
      return <hr className="my-12 border-t border-border" />;
    case 'section':
      return (
        <div className={block.className}>
          {block.children.map((child, i) => (
            <React.Fragment key={`${child.type}-${i}`}>
              {renderBlock(child)}
            </React.Fragment>
          ))}
        </div>
      );
    case 'columns':
      return (
        <div className={block.className ?? 'grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-1'}>
          {block.cols.map((col, colIdx) => (
            <div key={colIdx}>
              {col.map((child, i) => (
                <React.Fragment key={`${child.type}-${i}`}>
                  {renderBlock(child)}
                </React.Fragment>
              ))}
            </div>
          ))}
        </div>
      );
    case 'faq':
      return (
        <div className="grid max-w-5xl gap-4 my-4">
          {block.items.map((item, i) => (
            <details
              key={i}
              className="group self-start rounded-xl border border-border/60 bg-card shadow-sm"
            >
              <summary className="flex cursor-pointer list-none items-center justify-between px-6 py-4 text-foreground">
                <span className="text-sm font-medium">{renderInlineNodes(item.question)}</span>
                <span className="text-muted-foreground transition-transform duration-200 group-open:rotate-180 ml-2 shrink-0">
                  ⌄
                </span>
              </summary>
              <div className="px-6 pb-4 text-sm text-foreground leading-relaxed">
                {item.answerBlocks.map((child, j) => (
                  <React.Fragment key={`${child.type}-${j}`}>
                    {renderBlock(child)}
                  </React.Fragment>
                ))}
              </div>
            </details>
          ))}
        </div>
      );
    default:
      return null;
  }
}

function renderHeading(block: Extract<BuiltinDocBlock, { type: 'heading' }>) {
  const text = inlineToPlainText(block.text).replace(/`/g, '');
  const id = block.id ?? slugifyHeading(text);
  const headingClasses: Record<number, string> = {
    1: 'text-xl sm:text-2xl font-bold mt-10 mb-4 text-foreground first:mt-0 break-words',
    2: 'group scroll-mt-24 text-lg sm:text-xl font-semibold mt-8 mb-3 text-foreground break-words',
    3: 'group scroll-mt-24 text-base sm:text-lg font-semibold mt-6 mb-2 text-foreground break-words',
    4: 'group scroll-mt-24 text-[0.8125rem] sm:text-sm font-semibold mt-5 mb-1.5 text-muted-foreground uppercase tracking-wide break-words',
    5: 'group scroll-mt-24 text-sm font-semibold mt-4 mb-1.5 text-foreground break-words',
    6: 'group scroll-mt-24 text-sm font-semibold mt-3 mb-1.5 text-foreground break-words',
  };

  const Tag = (`h${block.level}` as React.ElementType);
  const className = headingClasses[block.level];

  if (block.level === 1) {
    return (
      <Tag className={className}>
        {renderInlineNodes(block.text)}
      </Tag>
    );
  }

  return (
    <Tag id={id} className={className}>
      {renderInlineNodes(block.text)}
      <HeadingAnchor id={id} />
    </Tag>
  );
}

function renderCodeBlock(block: Extract<BuiltinDocBlock, { type: 'code' }>) {
  const language = block.language ?? 'plaintext';
  if (language === 'mermaid') {
    return (
      <div className="my-8 w-full">
        <MermaidDiagram chart={block.content.trim()} />
      </div>
    );
  }

  const code = block.content.trim();
  const highlighted = highlightCode(block.content, language);
  return (
    <div className="my-8 relative group">
      {block.runnable && (
        <div className="absolute top-3 right-3 z-10">
          <TryInBrowserButton
            code={code}
            size="sm"
            className="bg-code-surface/80 backdrop-blur-sm"
            source="builtin-doc-code-block"
          />
        </div>
      )}
      <pre className="markdown-pre m-0 max-w-full overflow-x-auto">
        <code
          className={`markdown-code language-${language}`}
          dangerouslySetInnerHTML={{ __html: highlighted }}
        />
      </pre>
    </div>
  );
}

function renderInlineNodes(nodes: BuiltinDocInlineNode[]): React.ReactNode[] {
  return nodes.map((node, index) => {
    switch (node.type) {
      case 'text':
        return <React.Fragment key={index}>{node.value}</React.Fragment>;
      case 'code':
        return (
          <code key={index} className="markdown-inline-code">
            {node.value}
          </code>
        );
      case 'link':
        return (
          <a
            key={index}
            href={node.href}
            className="text-[hsl(var(--brand))] hover:text-[hsl(var(--brand))]/80 underline underline-offset-4 transition-colors"
          >
            {renderInlineNodes(node.label)}
          </a>
        );
      case 'strong':
        return (
          <strong key={index} className="font-semibold">
            {renderInlineNodes(node.content)}
          </strong>
        );
      default:
        return null;
    }
  });
}

function inlineToPlainText(nodes: BuiltinDocInlineNode[]): string {
  return nodes.map((node) => {
    if (node.type === 'text' || node.type === 'code') return node.value;
    if (node.type === 'link') return inlineToPlainText(node.label);
    if (node.type === 'strong') return inlineToPlainText(node.content);
    return '';
  }).join('');
}

function highlightCode(code: string, language: string): string {
  const grammar = Prism.languages[language as keyof typeof Prism.languages];
  if (!grammar) {
    return escapeHtml(code);
  }
  return Prism.highlight(code, grammar, language);
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
