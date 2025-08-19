"use client";
import { useSearchParams } from "next/navigation";
import { DocsSearchResults } from "./DocsSearchResults";
import dynamic from "next/dynamic";

// Load MarkdownRenderer only on the server; on client we never render it here.
const Noop: React.FC<{ source: string }> = () => null;
const ClientMarkdown = dynamic(async () => Noop, { ssr: false });

export function DocsContentSwitch({ source }: { source: string }) {
  const params = useSearchParams();
  const q = (params.get("q") || "").trim();
  if (q) {
    return <DocsSearchResults source={source} />;
  }
  // On client path, when no query present, render nothing here (server-rendered copy is visible)
  return <ClientMarkdown source={source} />;
}


