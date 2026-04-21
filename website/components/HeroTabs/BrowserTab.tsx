"use client";

import { ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { trackWebsiteEvent } from "@/components/GoogleAnalytics";

export function BrowserTab() {
  const handleClick = () => {
    trackWebsiteEvent("website.hero.cta_clicked", {
      category: "hero_tabs",
      label: "browser",
    });
    window.open("https://runmat.com/sandbox", "_blank", "noopener,noreferrer");
  };

  return (
    <div className="flex items-center justify-start pt-4">
      <Button
        size="lg"
        onClick={handleClick}
        className="h-12 px-8 text-base font-semibold bg-[hsl(var(--brand))] text-white border-0 shadow-none hover:bg-[hsl(var(--brand))]/90"
        data-ph-capture-attribute-destination="sandbox"
        data-ph-capture-attribute-source="hero-tabs-browser"
        data-ph-capture-attribute-cta="launch-browser-app"
      >
        Launch Browser App
        <ExternalLink className="ml-2 h-4 w-4" aria-hidden="true" />
      </Button>
    </div>
  );
}
