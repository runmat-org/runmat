"use client";

import { useEffect, useMemo, useState } from "react";
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
  const headings = useMemo(() => extractHeadings(source), [source]);
  const [activeId, setActiveId] = useState<string>(headings[0]?.id ?? "");

  useEffect(() => {
    if (!headings.length) return;
    const headingEls = headings
      .map((h) => document.getElementById(h.id))
      .filter((el): el is HTMLElement => Boolean(el));
    if (!headingEls.length) return;

    let raf = 0;
    const updateActive = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const offset = 120; // accounts for sticky nav + heading scroll margin
        let current = headingEls[0].id;
        for (const el of headingEls) {
          if (el.getBoundingClientRect().top - offset <= 0) current = el.id;
          else break;
        }
        setActiveId(current);
      });
    };

    updateActive();
    window.addEventListener("scroll", updateActive, { passive: true });
    window.addEventListener("hashchange", updateActive);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("scroll", updateActive);
      window.removeEventListener("hashchange", updateActive);
    };
  }, [headings]);

  if (!headings.length) return null;
  return (
    <aside className="hidden lg:block self-start sticky top-24 h-max w-[220px] shrink-0">
      <div className="max-h-[calc(100vh-7rem)] overflow-y-auto">
        <div className="text-sm font-semibold text-foreground/90 mb-2">On this page</div>
        <ul className="text-sm space-y-2">
          {headings.map((h, i) => (
            <li key={i} className={h.depth > 2 ? "pl-4" : undefined}>
              <a
                href={`#${h.id}`}
                aria-current={activeId === h.id ? "true" : undefined}
                className={`block border-l-2 pl-2 transition-colors ${
                  activeId === h.id
                    ? "border-[#B864F4] text-foreground font-medium"
                    : "border-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                {h.text}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}

