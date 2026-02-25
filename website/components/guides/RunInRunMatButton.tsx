"use client";

import { useCallback } from "react";
import { openWorkspace } from "@/lib/desktop";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface RunInRunMatButtonProps {
  className?: string;
}

export function RunInRunMatButton({ className }: RunInRunMatButtonProps) {
  const handleClick = useCallback(() => {
    openWorkspace(
      [{ path: "/untitled.m", content: "" }],
      { targetPath: "/sandbox", metadata: { source: "eigenvalue-explorer" }, newTab: true }
    );
  }, []);

  return (
    <Button
      variant="outline"
      size="lg"
      className={cn(
        "inline-flex items-center gap-2 font-medium border-[#8b5cf6]/50 text-[#a78bfa] hover:bg-[#8b5cf6]/10 hover:border-[#8b5cf6]/80 hover:text-[#c4b5fd]",
        className
      )}
      data-ph-capture-attribute-destination="sandbox"
      data-ph-capture-attribute-source="eigenvalue-explorer"
      data-ph-capture-attribute-cta="run-in-browser"
      onClick={handleClick}
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="size-4"
        aria-hidden
      >
        <path d="M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z" />
      </svg>
      Run in RunMat
    </Button>
  );
}
