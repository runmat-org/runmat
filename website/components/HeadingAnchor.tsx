"use client";
import { useState } from "react";
import { Link as LinkIcon, Check } from "lucide-react";

export function HeadingAnchor({ id }: { id: string }) {
  const [copied, setCopied] = useState(false);

  async function copyLink(e: React.MouseEvent<HTMLButtonElement>) {
    e.preventDefault();
    try {
      const { origin, pathname } = window.location;
      const url = `${origin}${pathname}#${id}`;
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      // ignore
    }
  }

  return (
    <button
      aria-label={copied ? "Copied" : "Copy link"}
      title={copied ? "Copied" : "Copy link"}
      onClick={copyLink}
      className="ml-2 inline-flex items-center rounded p-1 text-muted-foreground hover:text-foreground opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
    >
      {copied ? <Check className="h-4 w-4" /> : <LinkIcon className="h-4 w-4" />}
    </button>
  );
}


