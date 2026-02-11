"use client";

import Link from "next/link";
import { useState } from "react";
import { Check } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type CloudTier = "free" | "pro" | "team";

const cloudTierConfig: Record<
  CloudTier,
  {
    price: string;
    description: string;
    features: string[];
    ctaLabel: string;
    ctaHref: string;
  }
> = {
  free: {
    price: "$0/mo",
    description: "Sign up free to get cloud storage and collaboration.",
    features: [
      "5 projects",
      "1GB cloud storage",
      "Community support",
      "Version history (7 days)",
    ],
    ctaLabel: "Start Free",
    ctaHref: "/sandbox",
  },
  pro: {
    price: "$30/mo",
    description: "For individuals and small teams shipping real work.",
    features: [
      "Unlimited projects",
      "50GB cloud storage",
      "LLM zero data retention",
      "Collaboration (5 seats)",
      "Version history (30 days)",
      "Priority support",
    ],
    ctaLabel: "Get Started",
    ctaHref: "/sandbox",
  },
  team: {
    price: "$99/mo",
    description: "For organizations that need collaboration and compliance.",
    features: [
      "Team seats + shared workspaces",
      "250GB cloud storage",
      "LLM zero data retention",
      "Versioning and audit logs",
      "Centralized billing",
      "Priority support",
    ],
    ctaLabel: "Contact Sales",
    ctaHref: "mailto:team@runmat.com?subject=RunMat%20Cloud%20Team",
  },
};

const cloudTierTabs: { id: CloudTier; label: string }[] = [
  { id: "free", label: "Free" },
  { id: "pro", label: "Pro" },
  { id: "team", label: "Team" },
];

export default function CloudPricingCard() {
  const [activeTier, setActiveTier] = useState<CloudTier>("pro");
  const currentTier = cloudTierConfig[activeTier];

  return (
    <Card className="relative h-full border border-blue-500/50 bg-muted/40 shadow-lg shadow-blue-500/10">
      <CardHeader className="space-y-4 pb-4">
        <div className="flex items-center justify-between">
          <Badge className="bg-violet-500/20 text-violet-800 border-violet-600/50 dark:text-violet-200 dark:border-violet-400/40 hover:bg-violet-500/20">
            Cloud
          </Badge>
          <Badge className="bg-gradient-to-r from-blue-500 to-purple-600 text-white border-0 hover:from-blue-500 hover:to-purple-600">
            Most Popular
          </Badge>
        </div>
        <CardTitle className="text-xl text-foreground">RunMat Cloud</CardTitle>
        <div
          role="tablist"
          aria-label="RunMat Cloud tiers"
          className="grid min-w-[7rem] grid-cols-3 gap-1 rounded-lg border border-border/60 bg-background/50 p-1"
        >
          {cloudTierTabs.map(({ id, label }) => {
            const selected = activeTier === id;
            return (
              <button
                key={id}
                id={`pricing-cloud-tab-${id}`}
                type="button"
                role="tab"
                aria-selected={selected}
                aria-controls={`pricing-cloud-panel-${id}`}
                onClick={() => setActiveTier(id)}
                className={cn(
                  "whitespace-nowrap rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                  selected
                    ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white"
                    : "text-gray-300 hover:bg-white/10 hover:text-white"
                )}
              >
                {label}
              </button>
            );
          })}
        </div>
      </CardHeader>
      <CardContent
        role="tabpanel"
        id={`pricing-cloud-panel-${activeTier}`}
        aria-labelledby={`pricing-cloud-tab-${activeTier}`}
        className="flex flex-1 min-h-0 flex-col space-y-5"
      >
        <div className="flex min-h-0 flex-1 flex-col space-y-5">
          <div>
            <p className="text-3xl font-bold text-foreground">{currentTier.price}</p>
            <p className="mt-2 text-sm text-muted-foreground">{currentTier.description}</p>
          </div>
          <ul className="space-y-2">
            {currentTier.features.map(feature => (
              <li key={feature} className="flex items-start gap-2 text-sm text-muted-foreground">
                <Check className="mt-0.5 h-4 w-4 flex-shrink-0 text-violet-300" />
                <span>{feature}</span>
              </li>
            ))}
          </ul>
        </div>
        <Button
          asChild
          className="w-full rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow border-0 hover:from-blue-600 hover:to-purple-700 transition-colors"
        >
          <Link href={currentTier.ctaHref}>{currentTier.ctaLabel}</Link>
        </Button>
      </CardContent>
    </Card>
  );
}
