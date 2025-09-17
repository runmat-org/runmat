import React from "react";
import { slugifyHeading } from "@/lib/utils";

type Heading = { depth: number; text: string; id: string };

function extractHeadings(md: string): Heading[] {
  const lines = md.split(/\r?\n/);
  const out: Heading[] = [];
  let inFence = false;
  for (const raw of lines) {
    const line = raw.trim();
    if (/^```/.test(line)) { inFence = !inFence; continue; }
    if (inFence) continue;
    const m = /^(#{2,6})\s+(.+)$/.exec(line);
    if (!m) continue;
    const depth = m[1].length;
    const text = m[2].replace(/`/g, "");
    const id = slugifyHeading(text);
    out.push({ depth, text, id });
  }
  return out;
}

export function TableOfContents({ source, title = "On this page" }: { source: string; title?: string }) {
  const headings = React.useMemo(() => extractHeadings(source), [source]);
  if (!headings.length) return null;
  return (
    <div className="sticky top-24">
      <div className="text-sm font-semibold text-foreground/90 mb-2">{title}</div>
      <ul className="text-sm space-y-2">
        {headings.map((h, i) => (
          <li key={i} className={h.depth > 2 ? "pl-4" : undefined}>
            <a href={`#${h.id}`} className="text-muted-foreground hover:text-foreground">
              {h.text}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

export { extractHeadings };

