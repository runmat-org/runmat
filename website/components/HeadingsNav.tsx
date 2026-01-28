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
    const text = m[2].replace(/`/g, "").replace(/\*\*/g, "");
    const id = slugifyHeading(text);
    out.push({ depth, text, id });
  }
  return out;
}

export function HeadingsNav({ source }: { source: string }) {
  const headings = extractHeadings(source);
  if (!headings.length) return null;
  return (
    <aside className="hidden lg:block">
      <div className="sticky top-24">
        <div className="text-sm font-semibold text-foreground/90 mb-2">On this page</div>
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
    </aside>
  );
}

