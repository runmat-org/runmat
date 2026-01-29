"use client";

import { useMemo } from "react";
import Prism from "prismjs";
import "prismjs/components/prism-matlab";
import { TryInBrowserButton } from "@/components/TryInBrowserButton";

type MatlabInlineCodeBlockProps = {
  code: string;
  showRunButton?: boolean;
  preClassName?: string;
};

export default function MatlabInlineCodeBlock({
  code,
  showRunButton = false,
  preClassName = "",
}: MatlabInlineCodeBlockProps) {
  const highlighted = useMemo(
    () => Prism.highlight(code, Prism.languages.matlab, "matlab"),
    [code]
  );

  return (
    <div className="w-full flex flex-col gap-3">
      <pre
        className={`markdown-pre m-0 max-w-full overflow-x-auto w-full ${preClassName}`}
        suppressHydrationWarning
      >
        <code
          className="markdown-code language-matlab"
          dangerouslySetInnerHTML={{ __html: highlighted }}
          suppressHydrationWarning
        />
      </pre>
      {showRunButton && (
        <div className="flex justify-end w-full">
          <TryInBrowserButton code={code} size="sm" source="matlab-inline-code-block" />
        </div>
      )}
    </div>
  );
}
