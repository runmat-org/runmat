"use client";

import { Button } from "@/components/ui/button";
import { openWorkspace } from "@/lib/desktop";

interface TryInBrowserButtonProps extends Omit<React.ComponentProps<typeof Button>, "className"> {
  code?: string;
}

export function TryInBrowserButton({ 
  variant = "outline", 
  size = "lg", 
  code,
  ...props 
}: TryInBrowserButtonProps) {
  return (
      <Button
          variant={variant}
          size={size}
          className="inline-flex items-center gap-2 font-medium transition-all duration-200 text-sm px-3.5 py-2 rounded-md border border-[#8b5cf6]/50 text-[#a78bfa] hover:bg-[#8b5cf6]/10 hover:border-[#8b5cf6]/80 hover:text-[#c4b5fd]"
          onClick={() =>
              openWorkspace(
                  [{path: "/test.m", content: code || "disp('hello from docs');"}],
                  {targetPath: "/sandbox", metadata: {source: "example-page"}}
              )
          }
          {...props}
      >
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
               className="lucide lucide-play w-4 h-4" aria-hidden="true">
              <path d="M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z"></path>
          </svg>
          Run in browser
      </Button>
  );
}

