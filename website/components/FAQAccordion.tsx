import React from "react";

export type FAQItem = {
  id: string;
  question: string;
  answer: string;
  answerContent?: React.ReactNode;
};

type FAQAccordionProps = {
  items: FAQItem[];
  columns?: 1 | 2;
  size?: "default" | "compact";
};

export function FAQAccordion({ items, columns = 2, size = "default" }: FAQAccordionProps) {
  const isCompact = size === "compact";
  const gridClass = columns === 2
    ? "mx-auto grid max-w-5xl gap-4 md:grid-cols-2"
    : "mx-auto grid max-w-5xl gap-4";

  return (
    <div className={gridClass}>
      {items.map(item => (
        <details
          key={item.id}
          className="group self-start rounded-xl border border-border/60 bg-card shadow-sm"
        >
          <summary className={`flex cursor-pointer list-none items-center justify-between text-foreground ${isCompact ? "px-4 py-2.5" : "px-6 py-4"}`}>
            <span className={`font-medium ${isCompact ? "text-xs" : "text-sm"}`}>{item.question}</span>
            <span className="text-muted-foreground transition-transform duration-200 group-open:rotate-180 ml-2 shrink-0">
              ⌄
            </span>
          </summary>
          <div className={`text-foreground leading-relaxed ${isCompact ? "px-4 pb-3 text-xs" : "px-6 pb-4 text-sm"}`}>
            {item.answerContent ?? item.answer}
          </div>
        </details>
      ))}
    </div>
  );
}
