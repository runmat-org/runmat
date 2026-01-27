"use client";

import Link from "next/link";

export function DesktopTab() {
  return (
    <div className="flex items-center justify-start gap-2 pt-4 text-sm">
      <span className="text-muted-foreground">Coming soon</span>
      <span className="text-muted-foreground">·</span>
      <Link
        href="#newsletter"
        className="text-primary underline underline-offset-4 transition-colors"
        onClick={(e) => {
          e.preventDefault();
          const element = document.getElementById("newsletter");
          if (element) {
            element.scrollIntoView({ behavior: "smooth", block: "start" });
          }
        }}
      >
        Get notified
      </Link>
    </div>
  );
}
