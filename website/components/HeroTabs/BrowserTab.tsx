"use client";

import { ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { trackEvent } from "@/components/GoogleAnalytics";

export function BrowserTab() {
  const handleClick = () => {
    trackEvent("hero_cta_click", "hero_tabs", "browser");
    window.open("https://runmat.org/sandbox", "_blank", "noopener,noreferrer");
  };

  return (
    <div className="flex items-center justify-start pt-4">
      <Button
        size="lg"
        onClick={handleClick}
        className="h-12 px-8 text-base font-semibold bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-xl border-0 transition-all duration-200 hover:from-blue-600 hover:to-purple-700"
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
