"use client";

import Link from "next/link";
import { useState } from "react";
import { Check } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type CloudTier = "free" | "pro" | "team";

type Feature = string | { label: string; href: string };

interface CloudTierDef {
  price: string;
  description: string;
  inheritsFrom?: string;
  features: Feature[];
  ctaLabel: string;
  ctaHref: string;
}

const cloudTierConfig: Record<CloudTier, CloudTierDef> = {
  free: {
    price: "Free",
    description: "Sign up free to get cloud storage and collaboration.",
    features: [
      "Unlimited projects",
      { label: "200 MB storage", href: "#compare-storage" },
      "Version history (uses storage)",
      "Community support",
    ],
    ctaLabel: "Start free",
    ctaHref: "/sandbox",
  },
  pro: {
    price: "$30/mo per user",
    description: "For individuals and small teams shipping real work.",
    inheritsFrom: "Everything in Free, plus:",
    features: [
      { label: "10 GB storage", href: "#compare-storage" },
    ],
    ctaLabel: "Get started",
    ctaHref: "/sandbox",
  },
  team: {
    price: "$100/mo per user",
    description: "For organizations that need SSO and centralized identity management.",
    inheritsFrom: "Everything in Pro, plus:",
    features: [
      { label: "100 GB storage", href: "#compare-storage" },
      "SSO / SAML (and SCIM)",
      "Priority support",
    ],
    ctaLabel: "Get started",
    ctaHref: "/sandbox",
  },
};

const cloudTierTabs: { id: CloudTier; label: string }[] = [
  { id: "free", label: "Free" },
  { id: "pro", label: "Pro" },
  { id: "team", label: "Team" },
];

export default function CloudPricingCard() {
  const [activeTier, setActiveTier] = useState<CloudTier>("free");
  const currentTier = cloudTierConfig[activeTier];

  return (
    <Card className="relative flex h-full flex-col border border-blue-500/50 bg-muted/40 shadow-lg shadow-blue-500/10">
      <CardHeader className="space-y-4 pb-4">
        <Badge className="w-fit bg-violet-500/20 text-violet-800 border-violet-600/50 dark:text-violet-200 dark:border-violet-400/40 hover:bg-violet-500/20">
          Cloud
        </Badge>
        <CardTitle className="text-xl text-foreground">RunMat Cloud</CardTitle>
        <p className="text-xs text-muted-foreground">Everything in RunMat, plus cloud storage and collaboration.</p>
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
                    : "text-muted-foreground hover:bg-accent hover:text-foreground"
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

          <div className="space-y-2">
            {currentTier.inheritsFrom && (
              <p className="text-xs font-medium text-muted-foreground/70">
                {currentTier.inheritsFrom}
              </p>
            )}
            <ul className="space-y-2">
              {currentTier.features.map(feature => {
                const label = typeof feature === "string" ? feature : feature.label;
                const href = typeof feature === "string" ? undefined : feature.href;
                return (
                  <li key={label} className="flex items-start gap-2 text-sm text-muted-foreground">
                    <Check className="mt-0.5 h-4 w-4 flex-shrink-0 text-violet-300" />
                    {href ? (
                      <a href={href} className="no-underline hover:text-foreground transition-colors">
                        {label}
                      </a>
                    ) : (
                      <span>{label}</span>
                    )}
                  </li>
                );
              })}
            </ul>
          </div>
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
