"use client";

import { useMemo } from "react";
import { slugifyHeading } from "@/lib/utils";
import { OnThisPageNav, type TocHeading } from "@/components/OnThisPageNav";

function extractHeadings(md: string): TocHeading[] {
  const lines = md.split(/\r?\n/);
  const out: TocHeading[] = [];
  let inFence = false;
  for (const raw of lines) {
    const line = raw.trim();
    if (/^```/.test(line)) { inFence = !inFence; continue; }
    if (inFence) continue;
    const m = /^(#{2,6})\s+(.+)$/.exec(line);
    if (!m) continue;
    const depth = m[1].length;
    const text = m[2]
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .replace(/`/g, "")
      .replace(/\*\*/g, "");
    const id = slugifyHeading(text);
    out.push({ depth, text, id });
  }
  return out;
}

export function HeadingsNav({ source, maxDepth }: { source: string; maxDepth?: number }) {
  const headings = useMemo(() => {
    const all = extractHeadings(source);
    return maxDepth ? all.filter((h) => h.depth <= maxDepth) : all;
  }, [source, maxDepth]);
  return <OnThisPageNav headings={headings} />;
}

