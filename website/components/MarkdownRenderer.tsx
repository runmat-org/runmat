import React from 'react';
import { MDXRemote } from 'next-mdx-remote/rsc';
import remarkGfm from 'remark-gfm';
import rehypePrism from 'rehype-prism-plus';
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

  const mergeClassNames = (...classes: Array<string | null | undefined | false>) =>
    classes.filter(Boolean).join(' ');

  const defaultComponents: MarkdownRendererComponents = {
    h1: ({ children, ...props }: { children: React.ReactNode }) => (
      <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold mt-12 mb-6 text-foreground first:mt-0 break-words" {...props}>{children}</h1>
    ),
    h2: ({ children, ...props }: { children: React.ReactNode }) => {
      const text = toPlainText(children);
      const id = slugifyHeading(text.replace(/`/g, ''));
      return (
        <h2 id={id} className="group scroll-mt-24 text-3xl sm:text-4xl md:text-5xl font-semibold mt-10 mb-5 text-foreground break-words" {...props}>
          {children}
          <HeadingAnchor id={id} />
        </h2>
      );
    },
    h3: ({ children, ...props }: { children: React.ReactNode }) => {
      const text = toPlainText(children);
      const id = slugifyHeading(text.replace(/`/g, ''));
      return (
        <h3 id={id} className="group scroll-mt-24 text-2xl sm:text-3xl font-semibold mt-8 mb-4 text-foreground break-words" {...props}>
          {children}
          <HeadingAnchor id={id} />
        </h3>
      );
    },
    h4: ({ children, ...props }: { children: React.ReactNode }) => {
      const text = toPlainText(children);
      const id = slugifyHeading(text.replace(/`/g, ''));
      return (
        <h4 id={id} className="group scroll-mt-24 text-xl sm:text-2xl font-semibold mt-6 mb-3 text-foreground break-words" {...props}>
          {children}
          <HeadingAnchor id={id} />
        </h4>
      );
    },
    h5: ({ children, ...props }: { children: React.ReactNode }) => {
      const text = toPlainText(children);
      const id = slugifyHeading(text.replace(/`/g, ''));
      return (
        <h5 id={id} className="group scroll-mt-24 text-lg sm:text-xl font-semibold mt-5 mb-2 text-foreground break-words" {...props}>
          {children}
          <HeadingAnchor id={id} />
        </h5>
      );
    },
    h6: ({ children, ...props }: { children: React.ReactNode }) => {
      const text = toPlainText(children);
      const id = slugifyHeading(text.replace(/`/g, ''));
      return (
        <h6 id={id} className="group scroll-mt-24 text-base sm:text-lg font-semibold mt-4 mb-2 text-foreground break-words" {...props}>
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
        return <div className="my-6 break-words" {...props}>{children}</div>;
      }
      return <p className="my-6 text-muted-foreground leading-relaxed break-words" {...props}>{children}</p>;
    },
    ul: ({ children, ...props }: { children: React.ReactNode }) => (
      <ul className="my-6 ml-6 space-y-2 list-disc marker:text-blue-500 break-words" {...props}>{children}</ul>
    ),
    ol: ({ children, ...props }: { children: React.ReactNode }) => (
      <ol className="my-6 ml-6 space-y-2 list-decimal marker:text-blue-500 break-words" {...props}>{children}</ol>
    ),
    li: ({ children, ...props }: { children: React.ReactNode }) => (
      <li className="text-muted-foreground leading-relaxed pl-2 break-words" {...props}>{children}</li>
    ),
    blockquote: ({ children, ...props }: { children: React.ReactNode }) => (
      <blockquote className="my-6 pl-4 sm:pl-6 border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/20 py-4 pr-4 rounded-r-lg break-words overflow-x-hidden" {...props}>
        <div className="text-muted-foreground italic break-words">{children}</div>
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
      <div className="my-8 mx-0 overflow-x-auto max-w-full">
        <table className="w-full min-w-full sm:min-w-[500px] border-collapse border border-border rounded-lg table-mobile-wrap" {...props}>{children}</table>
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
      <th className="border border-border px-2 py-2 sm:px-4 sm:py-3 text-xs sm:text-sm text-left font-semibold text-foreground" {...props}>{children}</th>
    ),
    td: ({ children, ...props }: { children: React.ReactNode }) => (
      <td className="border border-border px-2 py-2 sm:px-4 sm:py-3 text-xs sm:text-sm text-muted-foreground" {...props}>{children}</td>
    ),
    pre: ({ children, ...props }: { children?: React.ReactNode }) => {
      // Check if this contains a mermaid code block
      const hasMermaid = React.Children.toArray(children).some(child => {
        if (React.isValidElement(child)) {
          const childType = typeof child.type === 'string' ? child.type : null;
          if (childType === 'code') {
            const childProps = child.props as { className?: string };
            return childProps?.className?.includes('language-mermaid') || 
                   childProps?.className?.includes('mermaid');
          }
        }
        return false;
      });

      // If it's mermaid, don't wrap in pre - let the code component handle it
      if (hasMermaid) {
        return <>{children}</>;
      }

      // Check if this is a code block (has code child with hljs or language- class, or any code child)
      const hasCodeBlock = React.Children.toArray(children).some(child => {
        if (React.isValidElement(child)) {
          const childType = typeof child.type === 'string' ? child.type : null;
          if (childType === 'code') {
            const childProps = child.props as { className?: string };
            return childProps?.className?.includes('hljs') || 
                   childProps?.className?.includes('language-') ||
                   true; // Any code element inside pre should be styled
          }
        }
        return false;
      });
      
      if (hasCodeBlock) {
        const { className, ...rest } = props as { className?: string };
        return (
          <pre className={mergeClassNames("markdown-pre my-6 mx-0 max-w-full overflow-x-auto", className)} {...rest}>
            {children}
          </pre>
        );
      }
      return <pre className="overflow-x-auto max-w-full break-words" {...props}>{children}</pre>;
    },
    code: ({ children, className, ...props }: { children?: React.ReactNode; className?: string }) => {
      // Render Mermaid - check for language-mermaid or mermaid class
      // Also handle legacy highlight classes (hljs) from old content
      const classList = className?.split(/\s+/) ?? [];
      const isMermaid = classList.some(cls => cls.includes('language-mermaid') || cls === 'mermaid');
      if (isMermaid) {
        // Extract the actual mermaid content - children might be wrapped in additional elements
        const toPlain = (node: React.ReactNode): string => {
          if (node == null) return '';
          if (typeof node === 'string' || typeof node === 'number') return String(node);
          if (Array.isArray(node)) return node.map(toPlain).join('');
          if (React.isValidElement(node)) {
            const props = node.props as { children?: React.ReactNode };
            return toPlain(props?.children);
          }
          return '';
        };
        const mermaidContent = toPlain(children);
        
        return (
          <div className="my-8 w-full">
            <MermaidDiagram chart={mermaidContent.trim()} />
          </div>
        );
      }

      const isCodeBlock = classList.some(cls => cls.startsWith('language-'));

      if (isCodeBlock) {
        const finalClassName = mergeClassNames("markdown-code", className);
        return <code className={finalClassName} {...props}>{children}</code>;
      }

      const inlineClassName = mergeClassNames("markdown-inline-code", className);
      return <code className={inlineClassName} {...props}>{children}</code>;
    },
    img: ({ src, alt, ...props }: { src?: string; alt?: string } & Record<string, unknown>) => (
      <img 
        src={src} 
        alt={alt} 
        className="my-6 max-w-full h-auto rounded-lg" 
        loading="lazy"
        {...props} 
      />
    ),
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
      // Skip processing inside code fences (including mermaid)
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
        // Protect literal HTML-like tags that appear in prose (e.g., <main>, <insert>) from being parsed as JSX/HTML by MDX.
        // We only target a small allowlist of known tokens used in docs to avoid interfering with intentional MDX components.
        out = out.replace(/<\/?(main|anonymous|insert)(\s[^>]*)?>/gi, (m) => '`' + m + '`');
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
          rehypePlugins: [[rehypePrism, { 
            ignoreMissing: true,
            defaultLanguage: 'plaintext',
          }]],
          development: process.env.NODE_ENV !== 'production',
        },
      }}
    />
  );
}


