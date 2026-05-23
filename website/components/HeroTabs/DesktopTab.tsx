"use client";

import Link from "next/link";

export function DesktopTab() {
  return (
    <div className="flex items-center justify-start gap-2 pt-4 text-sm">
      <Link
        href="/download"
        className="text-primary underline underline-offset-4 transition-colors"
      >
        Download desktop app
      </Link>
    </div>
  );
}
