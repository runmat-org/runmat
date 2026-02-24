import type { Metadata } from "next";
import Link from "next/link";
import { Check } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import CloudPricingCard from "@/components/pricing/CloudPricingCard";
import { CompareProductsTable } from "@/components/pricing/ComparisonTables";

export const metadata: Metadata = {
  title: "RunMat Pricing | Free, Cloud, and Enterprise",
  description:
    "Simple RunMat pricing: free runtime and app (open source CLI, free browser & desktop), RunMat Cloud, and RunMat Enterprise for on-prem.",
  alternates: { canonical: "https://runmat.com/pricing" },
};

const runmatFreeFeatures = [
  "GPU acceleration (Metal/Vulkan/DX12)",
  "Interactive, high performance 2D/3D plotting",
  "CLI for scripts and CI/CD",
  "Zero-install (in browser) and IDE desktop app",
];

const serverFeatures = [
  "Self-hosted air-gapped deployment",
  "SSO and audit logs",
  "Dedicated support",
];

const pricingFaqItems: { question: string; answer: string }[] = [
  {
    question: "What's the difference between RunMat (free), Cloud, and Enterprise?",
    answer:
      "RunMat is free: the runtime is open source (CLI), and the browser sandbox and desktop app are free to use. RunMat Cloud adds cloud storage, sharing, and team workspaces with optional Pro and Team paid plans. RunMat Enterprise is self-hosted and air-gapped with SSO and audit logs for organizations.",
  },
  {
    question: "Is RunMat really free?",
    answer:
      "Yes. The RunMat runtime is open source and the browser and desktop app are free. RunMat Cloud has a free tier with unlimited projects and 200MB storage. Pro ($30/mo per user) and Team ($100/mo per user) are paid; RunMat Enterprise is custom pricing.",
  },
  {
    question: "Do I need an account to use RunMat?",
    answer:
      "No. The browser sandbox and desktop app work without an account. An account is only required for cloud storage and team features.",
  },
  {
    question: "What's included in Cloud Free vs Pro vs Team?",
    answer:
      "Free: unlimited projects, 200MB storage, limited LLM (small models only); no built-in version history. Pro: unlimited projects, 10GB storage, version history (counts toward storage), $10/mo LLM credits included ($30/mo per user). Team: SSO / SAML (and SCIM), audit logs, 100GB storage, version history, $25/mo LLM credits included, unlimited seats, priority support ($100/mo per user).",
  },
  {
    question: "How does RunMat Cloud billing work?",
    answer:
      "Pro and Team are monthly subscriptions billed per user (per seat). You can upgrade or change plan from your account. Beyond included storage and LLM credits, usage is pay-as-you-go; you can set a usage cap so you never spend more than you're comfortable with.",
  },
  {
    question: "When do I need RunMat Enterprise instead of Cloud?",
    answer:
      "Choose Enterprise when you need on-prem or air-gapped deployment, strict data residency, or SSO and audit compliance that must stay in your environment.",
  },
  {
    question: "Is there a free trial for Pro or Team?",
    answer:
      "You can start on the free tier and upgrade when you need more. Contact us for trial or pilot options for Team.",
  },
  {
    question: "Who do I contact for Team or Enterprise pricing?",
    answer:
      "For Team plans and RunMat Enterprise (custom deployment), use the Contact Sales option on this page or visit our contact page.",
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
        <section className="w-full py-16 md:py-24 lg:py-32">
          <div className="mx-auto max-w-3xl space-y-5 text-center">
            <h1 className="font-heading text-3xl leading-[1.1] sm:text-4xl md:text-5xl">
              Simple, transparent pricing
            </h1>
            <p className="mx-auto max-w-[42rem] leading-relaxed text-base text-muted-foreground sm:text-lg">
              From open source CLI to enterprise on-prem. Pick what fits your workflow.
            </p>
          </div>
        </section>

        <section className="pb-20 md:pb-28">
          <div className="grid gap-4 lg:grid-cols-3">
            <Card className="flex h-full flex-col border border-border/60 bg-muted/40">
              <CardHeader className="space-y-3 pb-4">
                <Badge className="w-fit bg-green-500/20 text-green-800 border-green-600/50 dark:text-green-200 dark:border-green-400/40 hover:bg-green-500/20">
                  Open Source Runtime
                </Badge>
                <CardTitle className="text-xl text-foreground">RunMat</CardTitle>
                <p className="text-3xl font-bold text-foreground">Free, forever</p>
                <p className="text-sm text-muted-foreground">
                  Blazing fast, MIT licensed math runtime
                </p>
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
                    className="w-full rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow border-0 hover:from-blue-600 hover:to-purple-700 transition-colors"
                  >
                    <Link href="/sandbox">Try in Browser</Link>
                  </Button>
                </div>
              </CardContent>
            </Card>

            <CloudPricingCard />

            <Card className="flex h-full flex-col border border-border/60 bg-muted/40">
              <CardHeader className="space-y-3 pb-4">
                <Badge className="w-fit bg-amber-500/20 text-amber-800 border-amber-600/50 dark:text-amber-200 dark:border-amber-400/40 hover:bg-amber-500/20">
                  Enterprise
                </Badge>
                <CardTitle className="text-xl text-foreground">RunMat Enterprise</CardTitle>
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
                <Button asChild variant="outline" className="w-full">
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
        <section className="py-16 md:py-24 lg:py-32">
          <div className="mx-auto max-w-4xl space-y-3 text-center mb-8">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">
              Frequently asked questions
            </h2>
            <p className="text-lg text-muted-foreground">
              Common questions about plans, billing, and RunMat Cloud.
            </p>
          </div>
          <div className="mx-auto grid max-w-5xl gap-4 md:grid-cols-2">
            {pricingFaqItems.map(item => (
              <details
                key={item.question}
                className="group self-start rounded-xl border border-border/60 bg-card shadow-lg"
              >
                <summary className="flex cursor-pointer list-none items-center justify-between px-6 py-4 text-foreground">
                  <span className="text-lg font-medium">{item.question}</span>
                  <span className="text-muted-foreground transition-transform duration-200 group-open:rotate-180">
                    ⌄
                  </span>
                </summary>
                <div className="px-6 pb-4 text-lg text-muted-foreground">
                  {item.answer}
                </div>
              </details>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
