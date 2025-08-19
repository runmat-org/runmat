import React from 'react';
import { MDXRemote } from 'next-mdx-remote/rsc';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { MermaidDiagram } from '@/components/MermaidDiagram';
import { slugifyHeading } from '@/lib/utils';
import { HeadingAnchor } from '@/components/HeadingAnchor';

type MDXComponent = (props: { children?: React.ReactNode } & Record<string, unknown>) => React.ReactElement | null;
type MarkdownRendererComponents = Record<string, MDXComponent | ((props: { children: React.ReactNode } & Record<string, unknown>) => React.ReactElement | null)>;

type MarkdownRendererProps = {
  source: string;
  components?: MarkdownRendererComponents;
};

// Server component to render Markdown/MDX content with shared components
export async function MarkdownRenderer({ source, components = {} }: MarkdownRendererProps) {
  function toPlainText(node: React.ReactNode): string {
    if (node == null) return "";
    if (typeof node === 'string' || typeof node === 'number') return String(node);
    if (Array.isArray(node)) return node.map(toPlainText).join(' ');
    if (React.isValidElement(node)) {
      // recursively extract from element children (handles <code> etc.)
      // Fix: TypeScript error if node.props is not guaranteed to have 'children'
      // Use type assertion to access 'children' safely
      const props = (node as React.ReactElement).props as { children?: React.ReactNode };
      return toPlainText(props.children);
    }
    return "";
  }

  const defaultComponents: MarkdownRendererComponents = {
    h1: ({ children, ...props }: { children: React.ReactNode }) => (
      <h1 className="text-4xl font-bold mt-12 mb-6 text-foreground first:mt-0" {...props}>{children}</h1>
    ),
    h2: ({ children, ...props }: { children: React.ReactNode }) => {
      const text = toPlainText(children);
      const id = slugifyHeading(text.replace(/`/g, ''));
      return (
        <h2 id={id} className="group scroll-mt-24 text-3xl font-semibold mt-10 mb-5 text-foreground" {...props}>
          {children}
          <HeadingAnchor id={id} />
        </h2>
      );
    },
    h3: ({ children, ...props }: { children: React.ReactNode }) => {
      const text = toPlainText(children);
      const id = slugifyHeading(text.replace(/`/g, ''));
      return (
        <h3 id={id} className="group scroll-mt-24 text-2xl font-semibold mt-8 mb-4 text-foreground" {...props}>
          {children}
          <HeadingAnchor id={id} />
        </h3>
      );
    },
    h4: ({ children, ...props }: { children: React.ReactNode }) => {
      const text = toPlainText(children);
      const id = slugifyHeading(text.replace(/`/g, ''));
      return (
        <h4 id={id} className="group scroll-mt-24 text-xl font-semibold mt-6 mb-3 text-foreground" {...props}>
          {children}
          <HeadingAnchor id={id} />
        </h4>
      );
    },
    h5: ({ children, ...props }: { children: React.ReactNode }) => {
      const text = toPlainText(children);
      const id = slugifyHeading(text.replace(/`/g, ''));
      return (
        <h5 id={id} className="group scroll-mt-24 text-lg font-semibold mt-5 mb-2 text-foreground" {...props}>
          {children}
          <HeadingAnchor id={id} />
        </h5>
      );
    },
    h6: ({ children, ...props }: { children: React.ReactNode }) => {
      const text = toPlainText(children);
      const id = slugifyHeading(text.replace(/`/g, ''));
      return (
        <h6 id={id} className="group scroll-mt-24 text-base font-semibold mt-4 mb-2 text-foreground" {...props}>
          {children}
          <HeadingAnchor id={id} />
        </h6>
      );
    },
    p: ({ children, ...props }: { children: React.ReactNode }) => {
      const hasBlock = React.Children.toArray(children).some(child => {
        if (React.isValidElement(child)) {
          const t = child.type;
          return typeof t === 'string' && ['pre', 'div', 'blockquote', 'ul', 'ol', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table'].includes(t);
        }
        return false;
      });
      if (hasBlock) {
        return <div className="my-6" {...props}>{children}</div>;
      }
      return <p className="my-6 text-muted-foreground leading-relaxed" {...props}>{children}</p>;
    },
    ul: ({ children, ...props }: { children: React.ReactNode }) => (
      <ul className="my-6 ml-6 space-y-2 list-disc marker:text-blue-500" {...props}>{children}</ul>
    ),
    ol: ({ children, ...props }: { children: React.ReactNode }) => (
      <ol className="my-6 ml-6 space-y-2 list-decimal marker:text-blue-500" {...props}>{children}</ol>
    ),
    li: ({ children, ...props }: { children: React.ReactNode }) => (
      <li className="text-muted-foreground leading-relaxed pl-2" {...props}>{children}</li>
    ),
    blockquote: ({ children, ...props }: { children: React.ReactNode }) => (
      <blockquote className="my-6 pl-6 border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/20 py-4 pr-4 rounded-r-lg" {...props}>
        <div className="text-muted-foreground italic">{children}</div>
      </blockquote>
    ),
    hr: ({ ...props }: Record<string, unknown>) => (
      <hr className="my-12 border-t border-border" {...props} />
    ),
    a: ({ children, href, ...props }: { children: React.ReactNode; href?: string }) => (
      <a href={href} className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 underline underline-offset-4 transition-colors" {...props}>
        {children}
      </a>
    ),
    table: ({ children, ...props }: { children: React.ReactNode }) => (
      <div className="my-8 overflow-x-auto">
        <table className="w-full border-collapse border border-border rounded-lg" {...props}>{children}</table>
      </div>
    ),
    thead: ({ children, ...props }: { children: React.ReactNode }) => (
      <thead className="bg-muted" {...props}>{children}</thead>
    ),
    tbody: ({ children, ...props }: { children: React.ReactNode }) => (
      <tbody {...props}>{children}</tbody>
    ),
    tr: ({ children, ...props }: { children: React.ReactNode }) => (
      <tr className="border-b border-border last:border-b-0" {...props}>{children}</tr>
    ),
    th: ({ children, ...props }: { children: React.ReactNode }) => (
      <th className="border border-border px-4 py-3 text-left font-semibold text-foreground" {...props}>{children}</th>
    ),
    td: ({ children, ...props }: { children: React.ReactNode }) => (
      <td className="border border-border px-4 py-3 text-muted-foreground" {...props}>{children}</td>
    ),
    code: ({ children, className, ...props }: { children?: React.ReactNode; className?: string }) => {
      // Render Mermaid even if the renderer bypassed <pre> wrapper
      if (className && /language-mermaid/.test(className)) {
        return (
          <div className="my-8">
            <MermaidDiagram chart={String(children ?? '')} />
          </div>
        );
      }
      // Otherwise fall back to default rendering (allows rehype-highlight SSR output to style)
      return <code className={className} {...props}>{children}</code>;
    },
    MermaidDiagram: (props: { chart?: string } & Record<string, unknown>) => (
      <div className="my-12 flex justify-center">
        <div className="max-w-full overflow-x-auto">
          <MermaidDiagram chart={String(props.chart ?? '')} {...props} />
        </div>
      </div>
    ),
  } as const;

  const merged: MarkdownRendererComponents = { ...defaultComponents, ...components };
  // Sanitize MDX pitfalls before compiling:
  // - Angle-bracket generics like Vec<ArgSpec> get interpreted as JSX. Wrap them in backticks.
  // - Bare braces like { dims, numeric_count } in prose are treated as JS expressions. Wrap in backticks outside code fences.
  function sanitizeMarkdown(input: string): string {
    const lines = input.split(/\r?\n/);
    let inFence = false;
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (/^\s*```/.test(line)) {
        inFence = !inFence;
        continue;
      }
      if (inFence) continue;
      // Process only parts of the line that are not inside inline code (backticks)
      const segments = line.split(/(`[^`]*`)/g);
      for (let s = 0; s < segments.length; s++) {
        const seg = segments[s];
        if (seg.startsWith('`') && seg.endsWith('`')) continue; // keep inline code untouched
        let out = seg;
        // Wrap Rust-like generics in backticks
        out = out.replace(/([A-Za-z0-9_]+)<([A-Za-z0-9_]+)>/g, '`$1<$2>`');
        // Wrap brace groups in backticks to avoid MDX expression evaluation
        out = out.replace(/\{([^{}\n]+)\}/g, '`{$1}`');
        segments[s] = out;
      }
      lines[i] = segments.join('');
    }
    return lines.join('\n');
  }
  const sanitized = sanitizeMarkdown(source);
  return (
    <MDXRemote
      source={sanitized}
      components={merged}
      options={{
        mdxOptions: {
          remarkPlugins: [remarkGfm],
          rehypePlugins: [[rehypeHighlight, { ignoreMissing: true }]],
          development: process.env.NODE_ENV !== 'production',
        },
      }}
    />
  );
}


