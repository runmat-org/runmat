import type { Metadata } from "next";
import React from "react";
import Link from "next/link";
import { Check } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import CloudPricingCard from "@/components/pricing/CloudPricingCard";
import { CompareProductsTable } from "@/components/pricing/ComparisonTables";

export const metadata: Metadata = {
  title: "RunMat Pricing | Hobby, Pro, Team, and Enterprise",
  description:
    "Simple RunMat pricing: free runtime, $0 Hobby cloud tier. Pro from $30/mo per user, Team from $100/mo per user. Enterprise for on-prem and air-gapped deployment.",
  alternates: { canonical: "https://runmat.com/pricing" },
  openGraph: {
    title: "RunMat Pricing | Hobby, Pro, Team, and Enterprise",
    description:
      "Free runtime, $0 Hobby cloud tier. Pro from $30/mo per user, Team from $100/mo per user. RunMat Enterprise for on-prem and air-gapped deployment.",
    url: "https://runmat.com/pricing",
  },
  twitter: {
    card: "summary_large_image",
    title: "RunMat Pricing | Hobby, Pro, Team, and Enterprise",
    description:
      "Free runtime, $0 Hobby cloud tier. Pro from $30/mo per user, Team from $100/mo per user. RunMat Enterprise for on-prem and air-gapped deployment.",
  },
};

const runmatFreeFeatures = [
  "GPU acceleration (Metal/Vulkan/DX12)",
  "Interactive, high performance 2D/3D plotting",
  "CLI for scripts and CI/CD",
  "Zero-install (in browser) and desktop app (coming soon)",
];

const serverFeatures = [
  "Self-hosted air-gapped deployment",
  "Data residency and ITAR compliance",
  "SSO, SCIM, and audit logs",
  "Offline licensing",
  "Dedicated support",
];

const pricingFaqItems: { question: string; answer: string; answerContent?: React.ReactNode }[] = [
  {
    question: "What's the difference between RunMat, Cloud, and Enterprise?",
    answer:
      "RunMat is the free, open-source runtime — GPU acceleration, plotting, CLI, and browser sandbox. The desktop app is coming soon. RunMat Cloud adds cloud storage, project sharing, and version history on top, with Hobby, Pro, and Team tiers. RunMat Enterprise is everything in Cloud, deployed on your own infrastructure for air-gapped, compliance-ready environments.",
  },
  {
    question: "Is RunMat really free?",
    answer:
      "Yes. The RunMat runtime is open source and the browser sandbox is free. The desktop app is coming soon and will also be free. RunMat Cloud has a $0 Hobby tier with unlimited projects and 100 MB storage. Pro ($30/mo per user) and Team ($100/mo per user) are paid; RunMat Enterprise is custom pricing.",
  },
  {
    question: "Do I need an account to use RunMat?",
    answer:
      "No. The browser sandbox works without an account. The desktop app (coming soon) will also work without one. An account is only required for cloud storage and team features.",
  },
  {
    question: "What's included in Cloud Hobby vs Pro vs Team?",
    answer:
      "Hobby: unlimited projects, 100 MB storage, version history (counts toward storage). Pro: unlimited projects, 10GB storage, version history (counts toward storage) ($30/mo per user). Team: unlimited projects, SSO / SAML / SCIM, 100GB storage, version history (counts toward storage), priority support ($100/mo per user).",
  },
  {
    question: "How does RunMat Cloud billing work?",
    answer:
      "Pro and Team are monthly subscriptions billed per user (per seat). You can upgrade or change plan from your account. Need more storage? Add space from your account settings.",
  },
  {
    question: "When do I need RunMat Enterprise instead of Cloud?",
    answer:
      "Choose Enterprise when you need on-prem or air-gapped deployment, strict data residency, or SSO and audit compliance that must stay in your environment.",
  },
  {
    question: "Is there a free trial for Pro or Team?",
    answer: "Start on the Hobby tier and upgrade to Pro or Team from your account when you're ready. No sales call needed.",
  },
  {
    question: "Who do I contact for Enterprise pricing?",
    answer: "You can sign up for Pro and Team directly from your account. For Enterprise, reach out via our contact page or email team@runmat.com.",
    answerContent: <>You can sign up for Pro and Team directly from your account. For Enterprise, reach out via our <Link href="/contact" className="underline hover:text-foreground">contact page</Link> or email <a href="mailto:team@runmat.com" className="underline hover:text-foreground">team@runmat.com</a>.</>,
  },
];

const pricingFaqJsonLd = {
  "@context": "https://schema.org",
  "@type": "FAQPage",
  mainEntity: pricingFaqItems.map(item => ({
    "@type": "Question",
    name: item.question,
    acceptedAnswer: {
      "@type": "Answer",
      text: item.answer,
    },
  })),
};

export default function PricingPage() {
  return (
    <div className="min-h-screen bg-background">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(pricingFaqJsonLd).replace(/<\//g, "<\\/"),
        }}
      />
      <div className="container mx-auto px-4 md:px-6">
        <section className="w-full pt-16 md:pt-24 lg:pt-32 pb-10 md:pb-12">
          <div className="mx-auto max-w-3xl space-y-4 text-center">
            <h1 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
              Pricing
            </h1>
            <h2>
              Start for free. Upgrade to get the capacity that exactly matches your team&apos;s needs.
            </h2>
          </div>
        </section>

        <section className="pb-16 md:pb-24 lg:pb-32">
          <div className="grid gap-4 lg:grid-cols-3">
            <Card className="flex h-full flex-col border border-border/60">
              <CardHeader className="space-y-3 pb-4">
                <Badge className="w-fit bg-green-500/20 text-green-800 border-green-600/50 dark:text-green-200 dark:border-green-400/40 hover:bg-green-500/20">
                  OSS
                </Badge>
                <CardTitle className="text-lg font-semibold text-foreground">RunMat Runtime</CardTitle>
                <p className="text-[0.938rem] text-foreground">Open-source, GPU-accelerated math runtime. MATLAB syntax, no account required.</p>
                <p className="text-3xl font-bold text-foreground">Free, forever</p>
              </CardHeader>
              <CardContent className="flex flex-1 min-h-0 flex-col space-y-6">
                <div className="flex min-h-0 flex-1 flex-col">
                  <ul className="space-y-2">
                    {runmatFreeFeatures.map(feature => (
                      <li key={feature} className="flex items-start gap-2 text-sm text-muted-foreground">
                        <Check className="mt-0.5 h-4 w-4 flex-shrink-0 text-green-300" />
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="flex flex-col gap-2">
                  <Button asChild variant="outline" className="w-full">
                    <Link href="https://github.com/runmat-org/runmat" target="_blank" rel="noopener noreferrer">
                      View on GitHub
                    </Link>
                  </Button>
                  <Button asChild variant="outline" className="w-full">
                    <Link href="/download">Download</Link>
                  </Button>
                  <Button
                    asChild
                    className="w-full rounded-none border-0 bg-[hsl(var(--brand))] text-white transition-opacity shadow-none hover:bg-[hsl(var(--brand))]/90"
                  >
                    <Link href="/sandbox">Open Browser Sandbox</Link>
                  </Button>
                </div>
              </CardContent>
            </Card>

            <CloudPricingCard />

            <Card className="flex h-full flex-col border border-border/60">
              <CardHeader className="space-y-3 pb-4">
                <Badge className="w-fit bg-amber-500/20 text-amber-800 border-amber-600/50 dark:text-amber-200 dark:border-amber-400/40 hover:bg-amber-500/20">
                  Enterprise
                </Badge>
                <CardTitle className="text-lg font-semibold text-foreground">RunMat Enterprise</CardTitle>
                <p className="text-[0.938rem] text-foreground">On-prem deployment and compliance.</p>
                <p className="text-3xl font-bold text-foreground">Custom</p>
                <p className="text-sm text-muted-foreground">Self-hosted deployment for secure, air-gapped environments.</p>
              </CardHeader>
              <CardContent className="flex flex-1 min-h-0 flex-col space-y-6">
                <div className="flex min-h-0 flex-1 flex-col">
                  <ul className="space-y-2">
                    {serverFeatures.map(feature => (
                      <li key={feature} className="flex items-start gap-2 text-sm text-muted-foreground">
                        <Check className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-300" />
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                <Button
                  asChild
                  className="w-full rounded-none border-0 bg-[hsl(var(--brand))] text-white transition-opacity shadow-none hover:bg-[hsl(var(--brand))]/90"
                >
                  <Link
                    href="/contact?type=enterprise"
                    data-ph-capture-attribute-destination="contact"
                    data-ph-capture-attribute-source="pricing-enterprise"
                    data-ph-capture-attribute-cta="contact-sales"
                  >
                    Contact Sales
                  </Link>
                </Button>
              </CardContent>
            </Card>
          </div>
        </section>

        <CompareProductsTable />

        {/* FAQ */}
        <section className="pt-8 md:pt-12 pb-16 md:pb-24 lg:pb-32">
          <div className="mx-auto flex max-w-[58rem] flex-col items-center space-y-4 text-center mb-12">
            <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl text-foreground">
              Frequently asked questions
            </h2>
          </div>
          <div className="mx-auto grid max-w-5xl gap-3 md:grid-cols-2">
            {pricingFaqItems.map(item => (
              <details
                key={item.question}
                className="group self-start rounded-lg border border-border/60 bg-card shadow-sm"
              >
                <summary className="flex cursor-pointer list-none items-center justify-between px-4 py-2.5 text-foreground">
                  <span className="text-xs font-medium">{item.question}</span>
                  <span className="text-muted-foreground transition-transform duration-200 group-open:rotate-180 ml-2 shrink-0">
                    ⌄
                  </span>
                </summary>
                <div className="px-4 pb-3 text-xs text-foreground leading-relaxed">
                  {item.answerContent ?? item.answer}
                </div>
              </details>
            ))}
          </div>
        </section>

        <div className="text-center pb-16">
          <p className="text-[0.938rem] text-foreground">
            Still have questions?{" "}
            <Link href="/contact" className="underline hover:text-foreground">
              Get in touch
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
