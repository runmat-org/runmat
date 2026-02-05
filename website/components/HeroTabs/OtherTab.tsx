"use client";

import { ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { trackEvent } from "@/components/GoogleAnalytics";

export function OtherTab() {
  const handleClick = () => {
    trackEvent("hero_cta_click", "hero_tabs", "other");
    window.open("https://runmat.com/download", "_blank", "noopener,noreferrer");
  };

  return (
    <div className="flex items-center justify-start pt-4">
      <Button
        variant="outline"
        size="lg"
        onClick={handleClick}
        className="h-12 px-8 text-base font-semibold"
      >
        View All Download Options
        <ExternalLink className="ml-2 h-4 w-4" aria-hidden="true" />
      </Button>
    </div>
  );
}
