"use client";

import { useEffect, useState } from "react";

export type TocHeading = { id: string; text: string; depth: number };

type OnThisPageNavProps = {
  headings: TocHeading[];
  asideClassName?: string;
  titleClassName?: string;
  listClassName?: string;
  getItemClassName?: (depth: number) => string | undefined;
};

export function OnThisPageNav({
  headings,
  asideClassName = "hidden lg:block self-start sticky top-24 h-max w-[220px] shrink-0",
  titleClassName = "text-sm font-semibold text-foreground/90 mb-2",
  listClassName = "text-sm space-y-2",
  getItemClassName = (depth) => (depth > 2 ? "pl-4" : undefined),
}: OnThisPageNavProps) {
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
    <aside className={asideClassName}>
      <div className="max-h-[calc(100vh-7rem)] overflow-y-auto">
        <div className={titleClassName}>On this page</div>
        <ul className={listClassName}>
          {headings.map((h, i) => (
            <li key={`${h.id}-${i}`} className={getItemClassName(h.depth)}>
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
