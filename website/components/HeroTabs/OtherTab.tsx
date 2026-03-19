"use client";

import { ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { trackWebsiteEvent } from "@/components/GoogleAnalytics";

export function OtherTab() {
  const handleClick = () => {
    trackWebsiteEvent("website.hero.cta_clicked", {
      category: "hero_tabs",
      label: "other",
    });
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
