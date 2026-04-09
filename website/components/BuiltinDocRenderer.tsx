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
  | { type: 'link'; label: BuiltinDocInlineNode[]; href: string };

export type BuiltinDocBlock =
  | { type: 'heading'; level: 1 | 2 | 3 | 4 | 5 | 6; text: BuiltinDocInlineNode[]; id?: string }
  | { type: 'paragraph'; content: BuiltinDocInlineNode[] }
  | { type: 'list'; ordered: boolean; items: BuiltinDocInlineNode[][] }
  | { type: 'code'; language?: string; runnable?: boolean; content: string }
  | { type: 'table'; headers: BuiltinDocInlineNode[][]; rows: BuiltinDocInlineNode[][][] }
  | { type: 'divider' };

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
    case 'divider':
      return <hr className="my-12 border-t border-border" />;
    default:
      return null;
  }
}

function renderHeading(block: Extract<BuiltinDocBlock, { type: 'heading' }>) {
  const text = inlineToPlainText(block.text).replace(/`/g, '');
  const id = block.id ?? slugifyHeading(text);
  const headingClasses: Record<number, string> = {
    1: 'text-2xl sm:text-3xl font-bold mt-10 mb-4 text-foreground first:mt-0 break-words',
    2: 'group scroll-mt-24 text-lg sm:text-xl font-semibold mt-8 mb-3 text-foreground break-words',
    3: 'group scroll-mt-24 text-base sm:text-lg font-semibold mt-6 mb-2 text-foreground break-words',
    4: 'group scroll-mt-24 text-sm sm:text-base font-semibold mt-5 mb-2 text-foreground break-words',
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
      default:
        return null;
    }
  });
}

function inlineToPlainText(nodes: BuiltinDocInlineNode[]): string {
  return nodes.map((node) => {
    if (node.type === 'text' || node.type === 'code') return node.value;
    if (node.type === 'link') return inlineToPlainText(node.label);
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
