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

// Server component: extract headings from the raw markdown source on the server,
// then pass the (small) headings array down to the client-side scroll-spy nav.
// This avoids shipping the full markdown text in the RSC payload and avoids
// re-running the regex extraction in the browser.
export function HeadingsNav({ source, maxDepth }: { source: string; maxDepth?: number }) {
  const all = extractHeadings(source);
  const headings = maxDepth ? all.filter((h) => h.depth <= maxDepth) : all;
  return <OnThisPageNav headings={headings} />;
}
