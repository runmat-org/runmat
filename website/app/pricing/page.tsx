import type { Metadata } from "next";
import Link from "next/link";
import { Check, X } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import CloudPricingCard from "@/components/pricing/CloudPricingCard";

export const metadata: Metadata = {
  title: "RunMat Pricing | Runtime, Cloud, and Server Plans",
  description:
    "Simple RunMat pricing from open source runtime and free app to RunMat Cloud and enterprise RunMat Server.",
  alternates: { canonical: "https://runmat.com/pricing" },
};

const runtimeFeatures = [
  "MIT licensed",
  "CLI for scripts and CI/CD",
  "Cross-platform (macOS/Linux/Windows)",
  "GPU acceleration (Metal/Vulkan/DX12)",
  "JIT compiler and fusion engine",
];

const appFeatures = [
  "Full IDE with code editor and file explorer",
  "Browser sandbox (zero install)",
  "Desktop app",
  "Interactive 2D/3D plotting",
  "Real-time type and shape diagnostics",
];

const serverFeatures = [
  "Self-hosted air-gapped deployment",
  "Seat/storage/token caps in license",
  "Isolated network environments",
  "SSO and audit logs",
  "Dedicated support",
];

export default function PricingPage() {
  return (
    <div className="min-h-screen bg-background">
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
          <div className="grid gap-4 lg:grid-cols-4">
            <Card className="border border-border/60 bg-muted/40">
              <CardHeader className="space-y-3 pb-4">
                <Badge className="w-fit bg-green-500/20 text-green-200 border-green-400/40 hover:bg-green-500/20">
                  Open Source
                </Badge>
                <CardTitle className="text-xl text-foreground">RunMat Runtime</CardTitle>
                <p className="text-3xl font-bold text-foreground">Free forever</p>
                <p className="text-sm text-muted-foreground">The open source engine behind all RunMat products. Use it standalone via the CLI.</p>
              </CardHeader>
              <CardContent className="space-y-6">
                <ul className="space-y-2">
                  {runtimeFeatures.map(feature => (
                    <li key={feature} className="flex items-start gap-2 text-sm text-muted-foreground">
                      <Check className="mt-0.5 h-4 w-4 flex-shrink-0 text-green-300" />
                      <span>{feature}</span>
                    </li>
                  ))}
                </ul>
                <Button asChild variant="outline" className="w-full">
                  <Link href="https://github.com/runmat-org/runmat" target="_blank" rel="noopener noreferrer">
                    View on GitHub
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card className="border border-border/60 bg-muted/40">
              <CardHeader className="space-y-3 pb-4">
                <Badge className="w-fit bg-blue-500/20 text-blue-200 border-blue-400/40 hover:bg-blue-500/20">
                  No Account Required
                </Badge>
                <CardTitle className="text-xl text-foreground">RunMat App</CardTitle>
                <p className="text-3xl font-bold text-foreground">Included</p>
                <p className="text-sm text-muted-foreground">Full IDE, browser sandbox, and desktop app. Start instantly.</p>
              </CardHeader>
              <CardContent className="space-y-6">
                <ul className="space-y-2">
                  {appFeatures.map(feature => (
                    <li key={feature} className="flex items-start gap-2 text-sm text-muted-foreground">
                      <Check className="mt-0.5 h-4 w-4 flex-shrink-0 text-blue-300" />
                      <span>{feature}</span>
                    </li>
                  ))}
                </ul>
                <Button
                  asChild
                  className="w-full rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow border-0 hover:from-blue-600 hover:to-purple-700 transition-colors"
                >
                  <Link href="/sandbox">Try in Browser</Link>
                </Button>
              </CardContent>
            </Card>

            <CloudPricingCard />

            <Card className="border border-border/60 bg-muted/40">
              <CardHeader className="space-y-3 pb-4">
                <Badge className="w-fit bg-amber-500/20 text-amber-200 border-amber-400/40 hover:bg-amber-500/20">
                  Enterprise
                </Badge>
                <CardTitle className="text-xl text-foreground">RunMat Server</CardTitle>
                <p className="text-3xl font-bold text-foreground">Custom</p>
                <p className="text-sm text-muted-foreground">Self-hosted deployment for secure, air-gapped environments.</p>
              </CardHeader>
              <CardContent className="space-y-6">
                <ul className="space-y-2">
                  {serverFeatures.map(feature => (
                    <li key={feature} className="flex items-start gap-2 text-sm text-muted-foreground">
                      <Check className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-300" />
                      <span>{feature}</span>
                    </li>
                  ))}
                </ul>
                <Button asChild variant="outline" className="w-full">
                  <Link href="mailto:team@runmat.com?subject=RunMat%20Server%20Inquiry">Contact Sales</Link>
                </Button>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Compare products: Runtime & App / Cloud / Server (3 columns) */}
        <section className="pb-16 md:pb-24">
          <h2 className="font-heading text-2xl font-semibold text-foreground mb-6 text-center sm:text-left">
            Compare products
          </h2>
          <div className="overflow-x-auto -mx-4 px-4 md:mx-0 md:px-0">
            <table className="w-full min-w-[480px] border-collapse text-sm">
              <thead>
                <tr className="border-b border-border/60 sticky top-0 z-10 bg-background">
                  <th className="text-left py-3 pr-4 font-medium text-muted-foreground sticky left-0 z-20 bg-background min-w-[180px]">
                    Feature
                  </th>
                  <th className="text-center py-3 px-4 font-medium text-foreground min-w-[110px]">Runtime & App</th>
                  <th className="text-center py-3 px-4 font-medium text-foreground min-w-[100px]">Cloud</th>
                  <th className="text-center py-3 px-4 font-medium text-foreground min-w-[100px]">Server</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground">
                <tr className="border-b border-border/40">
                  <td colSpan={4} className="py-2 pt-6 font-semibold text-foreground sticky left-0 bg-background">
                    Core Runtime
                  </td>
                </tr>
                {[
                  ["MATLAB syntax execution", "check", "check", "check"],
                  ["GPU acceleration (Metal/Vulkan/DX12)", "check", "check", "check"],
                  ["JIT compilation and fusion engine", "check", "check", "check"],
                  ["Cross-platform (macOS/Linux/Windows)", "check", "check", "check"],
                ].map(([label, ...cells]) => (
                  <ProductRow key={label} label={label} cells={cells as [string, string, string]} />
                ))}
                <tr className="border-b border-border/40">
                  <td colSpan={4} className="py-2 pt-6 font-semibold text-foreground sticky left-0 bg-background">
                    Development Environment
                  </td>
                </tr>
                {[
                  ["Browser sandbox", "check", "check", "check"],
                  ["Desktop app", "check", "check", "check"],
                  ["Code editor and file explorer", "check", "check", "check"],
                  ["Interactive 2D/3D plotting", "check", "check", "check"],
                  ["Type and shape diagnostics", "check", "check", "check"],
                ].map(([label, ...cells]) => (
                  <ProductRow key={label} label={label} cells={cells as [string, string, string]} />
                ))}
                <tr className="border-b border-border/40">
                  <td colSpan={4} className="py-2 pt-6 font-semibold text-foreground sticky left-0 bg-background">
                    Collaboration and Storage
                  </td>
                </tr>
                {[
                  ["Cloud file storage", "x", "check", "check"],
                  ["Project sharing", "x", "check", "check"],
                  ["Team workspaces", "x", "check", "check"],
                  ["File versioning", "x", "check", "check"],
                ].map(([label, ...cells]) => (
                  <ProductRow key={label} label={label} cells={cells as [string, string, string]} />
                ))}
                <tr className="border-b border-border/40">
                  <td colSpan={4} className="py-2 pt-6 font-semibold text-foreground sticky left-0 bg-background">
                    Security and Compliance
                  </td>
                </tr>
                {[
                  ["SOC 2 compliance", "x", "check", "check"],
                  ["LLM zero data retention", "x", "Paid plans", "check"],
                  ["SSO / SAML", "x", "Team plan", "check"],
                  ["Audit logs", "x", "Team plan", "check"],
                  ["Air-gapped deployment", "x", "x", "check"],
                ].map(([label, ...cells]) => (
                  <ProductRow key={label} label={label} cells={cells as [string, string, string]} />
                ))}
                <tr className="border-b border-border/40">
                  <td colSpan={4} className="py-2 pt-6 font-semibold text-foreground sticky left-0 bg-background">
                    Support
                  </td>
                </tr>
                {[
                  ["Community (GitHub)", "check", "check", "check"],
                  ["Email support", "x", "check", "check"],
                  ["Priority support", "x", "Paid plans", "check"],
                  ["Dedicated support and SLA", "x", "x", "check"],
                ].map(([label, ...cells]) => (
                  <ProductRow key={label} label={label} cells={cells as [string, string, string]} />
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Compare Cloud plans: Free / Pro / Team */}
        <section className="pb-20 md:pb-28">
          <h2 className="font-heading text-2xl font-semibold text-foreground mb-6 text-center sm:text-left">
            Compare Cloud plans
          </h2>
          <div className="overflow-x-auto -mx-4 px-4 md:mx-0 md:px-0">
            <table className="w-full min-w-[520px] border-collapse text-sm">
              <thead>
                <tr className="border-b border-border/60 sticky top-0 z-10 bg-background">
                  <th className="text-left py-3 pr-4 font-medium text-muted-foreground sticky left-0 z-20 bg-background min-w-[160px]">
                    Feature
                  </th>
                  <th className="text-center py-3 px-4 font-medium text-foreground min-w-[100px]">Free</th>
                  <th className="text-center py-3 px-4 font-medium text-foreground min-w-[120px]">Pro ($30/mo)</th>
                  <th className="text-center py-3 px-4 font-medium text-foreground min-w-[120px]">Team ($99/mo)</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground">
                <tr className="border-b border-border/40">
                  <td colSpan={4} className="py-2 pt-6 font-semibold text-foreground sticky left-0 bg-background">
                    Usage
                  </td>
                </tr>
                {[
                  ["Projects", "5", "Unlimited", "Unlimited"],
                  ["Cloud storage", "1 GB", "50 GB", "250 GB"],
                  ["Version history", "7 days", "30 days", "90 days"],
                ].map(([label, ...cells]) => (
                  <CloudRow key={label} label={label} cells={cells as [string, string, string]} />
                ))}
                <tr className="border-b border-border/40">
                  <td colSpan={4} className="py-2 pt-6 font-semibold text-foreground sticky left-0 bg-background">
                    Collaboration
                  </td>
                </tr>
                {[
                  ["Seats", "1", "5", "Unlimited"],
                  ["Shared workspaces", "x", "check", "check"],
                  ["Real-time collaboration", "x", "x", "Coming soon"],
                ].map(([label, ...cells]) => (
                  <CloudRow key={label} label={label} cells={cells as [string, string, string]} />
                ))}
                <tr className="border-b border-border/40">
                  <td colSpan={4} className="py-2 pt-6 font-semibold text-foreground sticky left-0 bg-background">
                    AI and Compute
                  </td>
                </tr>
                {[
                  ["LLM-assisted coding", "check", "check", "check"],
                  ["LLM zero data retention", "x", "check", "check"],
                ].map(([label, ...cells]) => (
                  <CloudRow key={label} label={label} cells={cells as [string, string, string]} />
                ))}
                <tr className="border-b border-border/40">
                  <td colSpan={4} className="py-2 pt-6 font-semibold text-foreground sticky left-0 bg-background">
                    Security
                  </td>
                </tr>
                {[
                  ["SOC 2 compliance", "check", "check", "check"],
                  ["SSO / SAML", "x", "x", "check"],
                  ["Audit logs", "x", "x", "check"],
                ].map(([label, ...cells]) => (
                  <CloudRow key={label} label={label} cells={cells as [string, string, string]} />
                ))}
                <tr className="border-b border-border/40">
                  <td colSpan={4} className="py-2 pt-6 font-semibold text-foreground sticky left-0 bg-background">
                    Billing and Support
                  </td>
                </tr>
                {[
                  ["Community support", "check", "check", "check"],
                  ["Priority support", "x", "check", "check"],
                  ["Centralized billing", "x", "x", "check"],
                  ["Invoice / PO billing", "x", "x", "check"],
                ].map(([label, ...cells]) => (
                  <CloudRow key={label} label={label} cells={cells as [string, string, string]} />
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}

function ProductRow({
  label,
  cells,
}: {
  label: string;
  cells: [string, string, string];
}) {
  return (
    <tr className="border-b border-border/40">
      <td className="py-3 pr-4 sticky left-0 bg-background min-w-[180px]">{label}</td>
      {cells.map((cell, i) => (
        <td key={`${label}-${i}`} className="py-3 px-4 text-center min-w-[100px]">
          {cell === "check" ? (
            <Check className="inline-block h-4 w-4 text-green-500" aria-hidden />
          ) : cell === "x" ? (
            <X className="inline-block h-4 w-4 text-muted-foreground" aria-hidden />
          ) : (
            <span className="text-muted-foreground">{cell}</span>
          )}
        </td>
      ))}
    </tr>
  );
}

function CloudRow({
  label,
  cells,
}: {
  label: string;
  cells: [string, string, string];
}) {
  return (
    <tr className="border-b border-border/40">
      <td className="py-3 pr-4 sticky left-0 bg-background min-w-[160px]">{label}</td>
      {cells.map((cell, i) => (
        <td key={`${label}-${i}`} className="py-3 px-4 text-center min-w-[100px]">
          {cell === "check" ? (
            <Check className="inline-block h-4 w-4 text-green-500" aria-hidden />
          ) : cell === "x" ? (
            <X className="inline-block h-4 w-4 text-muted-foreground" aria-hidden />
          ) : (
            <span className="text-muted-foreground">{cell}</span>
          )}
        </td>
      ))}
    </tr>
  );
}
