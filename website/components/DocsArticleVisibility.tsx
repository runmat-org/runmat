"use client";
import { useSearchParams } from "next/navigation";

export function DocsArticleVisibility({ children }: { children: React.ReactNode }) {
  const params = useSearchParams();
  const q = (params.get('q') || '').trim();
  // Hide the article when searching; keep in DOM for SSG and SEO
  return <div className={q ? 'hidden' : undefined}>{children}</div>;
}


