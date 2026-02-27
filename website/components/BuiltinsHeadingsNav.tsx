"use client";

import { OnThisPageNav, type TocHeading } from "@/components/OnThisPageNav";

export function BuiltinsHeadingsNav({ toc }: { toc: TocHeading[] }) {
  return (
    <OnThisPageNav
      headings={toc}
      asideClassName="hidden lg:block self-start sticky top-24 h-max text-sm w-[220px] shrink-0"
      titleClassName="mb-2 font-semibold text-foreground/80"
      listClassName="space-y-1"
      getItemClassName={(depth) => (depth === 3 ? "pl-3" : undefined)}
    />
  );
}
