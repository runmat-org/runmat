"use client";

import Link from "next/link";
import { useState } from "react";
import { Check } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type CloudTier = "pro" | "team";

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
  pro: {
    price: "$30/month per user",
    description: "For individuals and small teams using cloud projects.",
    features: [
      "Code versioning and execution history",
      "Team collaboration and sync",
      "10 GB cloud storage included",
      "$10/mo LLM credits",
    ],
    ctaLabel: "Sign up",
    ctaHref: "/login",
  },
  team: {
    price: "$100/mo per user",
    description: "For organizations that need centralized identity and support.",
    inheritsFrom: "Everything in Pro, plus:",
    features: [
      "SSO / SAML / SCIM",
      "$50/mo LLM credits",
      "100 GB storage included",
      "Priority support",
    ],
    ctaLabel: "Get started",
    ctaHref: "/login",
  },
};

const cloudTierTabs: { id: CloudTier; label: string }[] = [
  { id: "pro", label: "Pro" },
  { id: "team", label: "Team" },
];

export default function CloudPricingCard() {
  const [activeTier, setActiveTier] = useState<CloudTier>("pro");
  const currentTier = cloudTierConfig[activeTier];

  return (
    <Card className="relative flex h-full flex-col border border-[hsl(var(--brand))]/50 shadow-sm">
      <CardHeader className="space-y-3 pb-4">
        <CardTitle className="text-3xl font-semibold text-foreground">{currentTier.price}</CardTitle>
        <p className="text-[0.938rem] text-foreground">Cloud workspace for storage, collaboration, and AI credits.</p>

        <div
          role="tablist"
          aria-label="RunMat tiers"
          className="grid min-w-[7rem] grid-cols-2 gap-1 rounded-lg border border-border/60 bg-background/50 p-1 pb-0"
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
                      ? "bg-[hsl(var(--brand))] text-white"
                      : "text-muted-foreground hover:bg-accent hover:text-foreground"
                  )}
                >
                  {label}
                </button>
              );
            })}
          </div>
      </CardHeader>
      <CardContent className="flex flex-1 min-h-0 flex-col space-y-5">
        <div className="flex min-h-0 flex-1 flex-col space-y-5">


          <div
            role="tabpanel"
            id={`pricing-cloud-panel-${activeTier}`}
            aria-labelledby={`pricing-cloud-tab-${activeTier}`}
            className="space-y-2"
          >
            {currentTier.inheritsFrom && (
              <p className="text-[0.938rem] font-medium text-foreground">
                {currentTier.inheritsFrom}
              </p>
            )}
            <ul className="space-y-2">
              {currentTier.features.map(feature => {
                const label = typeof feature === "string" ? feature : feature.label;
                const href = typeof feature === "string" ? undefined : feature.href;
                return (
                  <li key={label} className="flex items-start gap-2 text-[0.938rem] text-foreground">
                    <Check className="mt-0.5 h-4 w-4 flex-shrink-0 text-foreground" />
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
          className="w-full rounded-none bg-[hsl(var(--brand))] text-white border-0 transition-colors shadow-none hover:bg-[hsl(var(--brand))]/90"
        >
          <Link href={currentTier.ctaHref}>{currentTier.ctaLabel}</Link>
        </Button>
      </CardContent>
    </Card>
  );
}
