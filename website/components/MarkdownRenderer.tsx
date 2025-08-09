import React from 'react';
import { MDXRemote } from 'next-mdx-remote/rsc';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { MermaidDiagram } from '@/components/MermaidDiagram';

type MDXComponent = (props: { children?: React.ReactNode } & Record<string, unknown>) => React.ReactElement | null;
type MarkdownRendererComponents = Record<string, MDXComponent | ((props: { children: React.ReactNode } & Record<string, unknown>) => React.ReactElement | null)>;

type MarkdownRendererProps = {
  source: string;
  components?: MarkdownRendererComponents;
};

// Server component to render Markdown/MDX content with shared components
export async function MarkdownRenderer({ source, components = {} }: MarkdownRendererProps) {
  const defaultComponents: MarkdownRendererComponents = {
    h1: ({ children, ...props }: { children: React.ReactNode }) => (
      <h1 className="text-4xl font-bold mt-12 mb-6 text-foreground first:mt-0" {...props}>{children}</h1>
    ),
    h2: ({ children, ...props }: { children: React.ReactNode }) => (
      <h2 className="text-3xl font-semibold mt-10 mb-5 text-foreground" {...props}>{children}</h2>
    ),
    h3: ({ children, ...props }: { children: React.ReactNode }) => (
      <h3 className="text-2xl font-semibold mt-8 mb-4 text-foreground" {...props}>{children}</h3>
    ),
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
  return (
    <MDXRemote
      source={source}
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


