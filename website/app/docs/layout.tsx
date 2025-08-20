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
        <div className="grid grid-cols-1 md:grid-cols-[260px_minmax(0,1fr)] gap-6">
          <Suspense fallback={<div className="hidden md:block" />}> 
            <Sidebar />
          </Suspense>
          <main className="min-w-0 md:pl-6 md:border-l md:border-border/60">{children}</main>
        </div>
      </div>
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
  const pathname = usePathname();
  const title = <div className="text-sm font-semibold text-foreground/90 mb-2">{node.title}</div>;
  // Section header only if it has children or a direct link
  return (
    <div>
      {title}
      {node.externalHref && (
        <ul className="space-y-1 pl-3 border-l border-border/50">
          <li>
            <Link href={node.externalHref} className={`text-sm hover:text-foreground ${pathname === node.externalHref ? 'text-foreground font-medium' : 'text-muted-foreground'}`}>
              {node.title}
            </Link>
          </li>
        </ul>
      )}
      {node.children && (
        <ul className="space-y-1 pl-3 border-l border-border/50">
          {node.children.map((c, i) => (
            <li key={i}>
              {c.externalHref ? (
                <Link href={c.externalHref} className={`text-sm hover:text-foreground ${pathname === c.externalHref ? 'text-foreground font-medium' : 'text-muted-foreground'}`}>{c.title}</Link>
              ) : c.slug ? (
                <Link href={["/docs", ...c.slug].join("/")} className={`text-sm hover:text-foreground ${pathname === ["/docs", ...c.slug].join("/") ? 'text-foreground font-medium' : 'text-muted-foreground'}`}>{c.title}</Link>
              ) : (
                <span className="text-sm text-muted-foreground">{c.title}</span>
              )}
              {c.children && (
                <ul className="pl-3 mt-1 space-y-1 border-l border-border/50">
                  {c.children.map((g, j) => (
                    <li key={j}>
                      {g.externalHref ? (
                        <Link href={g.externalHref} className={`text-sm hover:text-foreground ${pathname === g.externalHref ? 'text-foreground font-medium' : 'text-muted-foreground'}`}>
                          {g.title}
                        </Link>
                      ) : g.slug ? (
                        <Link href={["/docs", ...g.slug].join("/")} className={`text-sm hover:text-foreground ${pathname === ["/docs", ...g.slug].join("/") ? 'text-foreground font-medium' : 'text-muted-foreground'}`}>
                          {g.title}
                        </Link>
                      ) : (
                        <span className="text-sm text-muted-foreground">{g.title}</span>
                      )}
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


