"use client";
import Link from "next/link";
import { docsTree, DocsNode, flatten } from "@/content/docs";
import { Menu } from "lucide-react";
import { Suspense, useEffect, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

export default function DocsLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 md:px-6 py-8">
        <ConditionalLayout>{children}</ConditionalLayout>
      </div>
    </div>
  );
}

function ConditionalLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isElementsPage = pathname === '/docs/matlab-function-reference';
  
  if (isElementsPage) {
    // For matlab-function-reference page, render full-width without sidebar
    return <div>{children}</div>;
  }
  
  // For other pages, render with sidebar grid layout
  return (
    <div className="grid grid-cols-1 md:grid-cols-[260px_minmax(0,1fr)] gap-6">
      <Suspense fallback={<div className="hidden md:block" />}> 
        <Sidebar />
      </Suspense>
      <main className="min-w-0 md:pl-6 md:border-l md:border-border/60">{children}</main>
    </div>
  );
}

function Sidebar() {
  const [open, setOpen] = useState(false);
  const all = flatten();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [q, setQ] = useState("");
  // Keyboard shortcut: '/' focuses the search box
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const [inputRef, setInputRef] = useState<HTMLInputElement | null>(null);
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const isTypingField = target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || (target).isContentEditable);
      if (!isTypingField && (e.key === '/' || e.key.toLowerCase() === 's') && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault();
        inputRef?.focus();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [inputRef]);
  // keep input in sync with URL (?q=)
  useEffect(() => {
    const current = searchParams?.get("q") || "";
    if (current !== q) setQ(current);
  }, [searchParams]);

  const filtered = q
    ? all.filter((n) => n.title.toLowerCase().includes(q.toLowerCase()) && (n.slug || n.externalHref))
    : [];
  return (
    <aside>
      <button className="md:hidden flex items-center gap-2 text-sm mb-4" onClick={() => setOpen(!open)}>
        <Menu className="h-4 w-4" />
        Menu
      </button>
      <nav className={`md:block ${open ? "block" : "hidden"}`}>
        <div className="mb-5">
          <div className="relative">
            <input
              ref={setInputRef}
              value={q}
              onChange={(e) => {
                const v = e.target.value;
                setQ(v);
                const params = new URLSearchParams(searchParams?.toString());
                if (v) params.set('q', v); else params.delete('q');
                const nextPath = '/docs/search';
                router.push(`${nextPath}${params.toString() ? `?${params.toString()}` : ''}`);
              }}
              placeholder="Search docs…"
              className="w-full text-sm rounded-md border border-border bg-muted/40 pr-9 pl-3 py-2 h-10 focus:outline-none focus:ring-2 focus:ring-primary/60 focus:border-primary placeholder:text-muted-foreground/80 shadow-sm"
            />
            <div className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">
              <kbd className="px-1 py-0.5 rounded border">/</kbd>
            </div>
          </div>
        </div>
        <ul className="space-y-3 md:space-y-4">
          {docsTree.map((n, i) => (
            <li key={i}>
              <Section node={n} />
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  );
}

function Section({ node }: { node: DocsNode }) {
  const pathnameRaw = usePathname() || "/";
  const pathname = normalizePath(pathnameRaw);
  const title = <div className="text-sm font-semibold text-foreground/90 mb-2">{node.title}</div>;
  const showChildren =
    node.children &&
    (!node.hideChildrenInNav || (pathname ? nodeMatchesPath(node, pathname) : false));

  const renderLink = (href: string | undefined, label: string) =>
    href ? (
      <Link
        href={href}
        className={`text-sm hover:text-foreground ${pathname === normalizePath(href) ? "text-foreground font-medium" : "text-muted-foreground"}`}
      >
        {label}
      </Link>
    ) : (
      <span className="text-sm text-muted-foreground">{label}</span>
    );

  // Section header only if it has children or a direct link
  return (
    <div>
      {title}
      {node.externalHref && (
        <ul className="space-y-1 pl-3 border-l border-border/50">
          <li>
            <Link href={node.externalHref} className={`text-sm hover:text-foreground ${pathname === normalizePath(node.externalHref) ? 'text-foreground font-medium' : 'text-muted-foreground'}`}>
              {node.title}
            </Link>
          </li>
        </ul>
      )}
      {showChildren && (
        <ul className="space-y-1 pl-3 border-l border-border/50">
          {node.children.map((c, i) => (
            <li key={i}>
              {renderLink(c.slug ? ["/docs", ...c.slug].join("/") : c.externalHref, c.title)}
              {c.children && !c.hideChildrenInNav && (
                <ul className="pl-3 mt-1 space-y-1 border-l border-border/50">
                  {c.children.map((g, j) => (
                    <li key={j}>
                      {renderLink(g.slug ? ["/docs", ...g.slug].join("/") : g.externalHref, g.title)}
                    </li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function nodeMatchesPath(node: DocsNode, pathname: string): boolean {
  // Normalize pathname for consistent comparison
  const normalizedPathname = normalizePath(pathname);
  
  // Check if current path matches this node's slug or external href
  const slugPath = node.slug ? normalizePath(["/docs", ...node.slug].join("/")) : undefined;
  if (slugPath && (normalizedPathname === slugPath || normalizedPathname.startsWith(`${slugPath}/`))) return true;
  
  if (node.externalHref) {
    const external = normalizePath(node.externalHref);
    if (normalizedPathname === external || normalizedPathname.startsWith(`${external}/`)) return true;
    
    // Special case for fusion guide: also match child paths
    if (external === "/docs/fusion-guide") {
      if (normalizedPathname === "/docs/fusion-guide" || normalizedPathname.startsWith("/docs/fusion/")) {
        return true;
      }
    }
  }
  
  // Recursively check children
  return (node.children || []).some((child) => nodeMatchesPath(child, normalizedPathname));
}

function normalizePath(path: string): string {
  if (!path) return "/";
  const normalized = path.replace(/\/+$/, "") || "/";
  return normalized;
}


