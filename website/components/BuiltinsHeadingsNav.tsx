"use client";

import { useEffect, useState } from "react";

type TocEntry = { id: string; text: string; depth: number };

export function BuiltinsHeadingsNav({ toc }: { toc: TocEntry[] }) {
  const [activeId, setActiveId] = useState<string>(toc[0]?.id ?? "");

  useEffect(() => {
    if (!toc.length) return;
    const headingEls = toc
      .map((h) => document.getElementById(h.id))
      .filter((el): el is HTMLElement => Boolean(el));
    if (!headingEls.length) return;

    let raf = 0;
    const updateActive = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const offset = 120;
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
  }, [toc]);

  if (!toc.length) return null;
  return (
    <aside className="hidden lg:block self-start sticky top-24 h-max text-sm w-[220px] shrink-0">
      <div className="max-h-[calc(100vh-7rem)] overflow-y-auto">
        <div className="mb-2 font-semibold text-foreground/80">On this page</div>
        <ul className="space-y-1">
          {toc.map((t) => (
            <li key={t.id} className={t.depth === 3 ? "pl-3" : undefined}>
              <a
                href={`#${t.id}`}
                aria-current={activeId === t.id ? "true" : undefined}
                className={`block border-l-2 pl-2 transition-colors ${
                  activeId === t.id
                    ? "border-[#B864F4] text-foreground font-medium"
                    : "border-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                {t.text}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
