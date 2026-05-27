"use client";

import { Button } from "@/components/ui/button";
import { openWorkspace } from "@/lib/desktop";
import { cn } from "@/lib/utils";
import Link from "next/link";
import type { ReactNode } from "react";

interface TryInBrowserButtonProps extends React.ComponentProps<typeof Button> {
  code?: string;
  agentPrompt?: string;
  source?: string;
  exampleId?: string;
}

interface TryInBrowserLinkProps {
  href?: string;
  ariaLabel: string;
  code?: string;
  agentPrompt?: string;
  source?: string;
  exampleId?: string;
  className?: string;
  children: ReactNode;
}

const normalizeExampleId = (value?: string) => value?.trim() || undefined;

const hashCode = (value: string) => {
  let hash = 5381;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 33) ^ value.charCodeAt(i);
  }
  return (hash >>> 0).toString(36);
};

export function TryInBrowserButton({
  variant = "outline",
  size = "lg",
  code,
  agentPrompt,
  source = "try-in-browser",
  exampleId,
  className,
  children,
  ...props
}: TryInBrowserButtonProps) {
  const resolvedExampleId = normalizeExampleId(exampleId) ?? (code ? `code-${hashCode(code)}` : undefined);

  return (
      <Button
          variant={variant}
          size={size}
          className={cn("inline-flex items-center gap-2 font-medium transition-all duration-200 text-sm px-3.5 py-2 rounded-md border border-[#8b5cf6]/50 text-[#a78bfa] hover:bg-[#8b5cf6]/10 hover:border-[#8b5cf6]/80 hover:text-[#c4b5fd]", className)}
          data-ph-capture-attribute-destination="sandbox"
          data-ph-capture-attribute-source={source}
          data-ph-capture-attribute-cta="run-in-browser"
          data-ph-capture-attribute-example-id={resolvedExampleId}
          onClick={() =>
              openWorkspace(
                  [{path: "/example.m", content: code || "disp('hello from docs');"}],
                  {
                      targetPath: "/sandbox",
                      agentPrompt,
                      metadata: {
                          source,
                          ...(resolvedExampleId ? {exampleId: resolvedExampleId} : {}),
                      },
                  }
              )
          }
          {...props}
      >
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
               className="lucide lucide-play w-4 h-4" aria-hidden="true">
              <path d="M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z"></path>
          </svg>
          {children ?? "Run in browser"}
      </Button>
  );
}

export function TryInBrowserLink({
  href = "/sandbox",
  ariaLabel,
  code,
  agentPrompt,
  source = "try-in-browser-link",
  exampleId,
  className,
  children,
}: TryInBrowserLinkProps) {
  const linkClassName = cn(
    "block focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
    className,
  );

  if (!code && !agentPrompt) {
    return (
      <Link
        href={href}
        aria-label={ariaLabel}
        className={linkClassName}
        data-ph-capture-attribute-destination="sandbox"
        data-ph-capture-attribute-source={source}
        data-ph-capture-attribute-cta="open-product-media"
        data-ph-capture-attribute-example-id={exampleId}
      >
        {children}
      </Link>
    );
  }

  return (
    <a
      href={href}
      aria-label={ariaLabel}
      className={linkClassName}
      data-ph-capture-attribute-destination="sandbox"
      data-ph-capture-attribute-source={source}
      data-ph-capture-attribute-cta="open-product-media"
      data-ph-capture-attribute-example-id={exampleId}
      onClick={(event) => {
        const isNormalLeftClick =
          event.button === 0 &&
          !event.metaKey &&
          !event.ctrlKey &&
          !event.shiftKey &&
          !event.altKey;

        if (!isNormalLeftClick) {
          return;
        }

        event.preventDefault();

        try {
          const url = openWorkspace(
            [
              {
                path: "/example.m",
                content: code ?? `% ${agentPrompt}`,
              },
            ],
            {
              targetPath: href,
              agentPrompt,
              newTab: false,
              metadata: {
                source,
                ...(exampleId ? { exampleId } : {}),
              },
            },
          );

          if (!url) {
            window.location.assign(href);
          }
        } catch {
          window.location.assign(href);
        }
      }}
    >
      {children}
    </a>
  );
}
