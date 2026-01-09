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
      className="h-12 px-8 text-lg bg-[#0E1B1E] dark:bg-[#0E1B1E] border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100"
      onClick={() =>
        openWorkspace(
          [{ path: "/test.m", content: code || "disp('hello from docs');" }],
          { targetPath: "/sandbox", metadata: { source: "example-page" } }
        )
      }
      {...props}
    >
      Try in Browser
    </Button>
  );
}

