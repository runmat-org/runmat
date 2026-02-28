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
    const text = m[2].replace(/`/g, "").replace(/\*\*/g, "");
    const id = slugifyHeading(text);
    out.push({ depth, text, id });
  }
  return out;
}

export function HeadingsNav({ source }: { source: string }) {
  const headings = useMemo(() => extractHeadings(source), [source]);
  return <OnThisPageNav headings={headings} />;
}

