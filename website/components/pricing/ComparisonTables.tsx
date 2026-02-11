"use client";

import { Fragment, useState } from "react";
import { Check, X, ChevronRight } from "lucide-react";

const columnTints = [
  "rgba(59, 130, 246, 0.04)",
  "rgba(139, 92, 246, 0.04)",
  "rgba(245, 158, 11, 0.04)",
] as const;

type ProductRow = [string, string, string, string, string];
type CloudRow = [string, string, string, string, string];

const productRows: ProductRow[] = [
  ["MATLAB syntax execution", "check", "check", "check", "Execute standard MATLAB-syntax code across all RunMat products."],
  ["GPU acceleration (Metal/Vulkan/DX12)", "check", "check", "check", "Automatic GPU offload with Metal, Vulkan, and DirectX 12 support."],
  ["JIT compilation and fusion engine", "check", "check", "check", "Just-in-time compilation and operation fusion for faster execution."],
  ["Cross-platform (macOS/Linux/Windows)", "check", "check", "check", "Run the same code on macOS, Linux, and Windows."],
  ["Browser sandbox", "check", "check", "check", "Run code in the browser with zero install via the sandbox."],
  ["Desktop app", "check", "check", "check", "Full desktop IDE with local file access and offline use."],
  ["Code editor and file explorer", "check", "check", "check", "Integrated editor and project file browser."],
  ["Interactive 2D/3D plotting", "check", "check", "check", "Interactive plots you can rotate, zoom, and inspect."],
  ["Type and shape diagnostics", "check", "check", "check", "Real-time type and matrix shape tracking with dimension errors."],
  ["Cloud file storage", "x", "check", "check", "Persist and sync projects in the cloud."],
  ["Project sharing", "x", "check", "check", "Share projects and collaborate with your team."],
  ["Team workspaces", "x", "check", "check", "Organize work in shared team workspaces."],
  ["File versioning", "x", "check", "check", "Version history and restore for project files."],
  ["SOC 2 compliance", "x", "check", "check", "Audited security and compliance controls."],
  ["SSO / SAML", "x", "Team plan", "check", "Single sign-on and SAML for enterprise."],
  ["Audit logs", "x", "Team plan", "check", "Audit logging for team and compliance."],
  ["Air-gapped deployment", "x", "x", "check", "Deploy RunMat Enterprise in isolated networks."],
  ["Community (GitHub)", "check", "check", "check", "Community support and open source on GitHub."],
  ["Priority support", "x", "Team plan", "check", "Priority support on Team and Enterprise."],
];

const productSectionLabels = [
  "Core Runtime",
  "Development Environment",
  "Collaboration and Storage",
  "Security and Compliance",
  "Support",
];
const productSectionSizes = [4, 5, 4, 4, 2];

const cloudRows: CloudRow[] = [
  ["Projects", "Unlimited", "Unlimited", "Unlimited", "Number of projects you can create and store."],
  ["Cloud storage", "200 MB", "10 GB", "100 GB", "Total cloud storage for project files and assets."],
  ["Version history", "check", "check", "check", "File version history included for all plans; counts toward storage. Configure how many versions to keep per project."],
  ["Shared workspaces", "x", "check", "check", "Create and use shared team workspaces."],
  ["Real-time collaboration", "x", "x", "Coming soon", "See edits and presence in real time."],
  ["LLM-assisted coding", "check", "check", "check", "AI-assisted coding features in the editor."],
  ["Included LLM credits", "Limited", "$10/mo", "$25/mo", "Free: limited token budgets, small-model-only. Pro: $10/month included credits; Team: $25/month included; pay-as-you-go overage on paid tiers."],
  ["SOC 2 compliance", "check", "check", "check", "RunMat Cloud is SOC 2 compliant."],
  ["SSO / SAML", "x", "x", "check", "Single sign-on and SAML on Team plan."],
  ["Audit logs", "x", "x", "check", "Audit logs for team activity on Team plan."],
  ["Community support", "check", "check", "check", "Community and documentation support."],
  ["Priority support", "x", "x", "check", "Priority email support on Team plan."],
];

const cloudSectionLabels = ["Usage", "Collaboration", "AI and Compute", "Security", "Support"];
const cloudSectionSizes = [3, 2, 2, 3, 2];

function SectionHeaderRow({ label }: { label: string }) {
  return (
    <tr className="border-t border-border" style={{ backgroundColor: "hsl(var(--muted) / 0.25)" }}>
      <td colSpan={4} className="px-5 py-3 text-xs font-semibold text-foreground tracking-wider uppercase align-middle">
        {label}
      </td>
    </tr>
  );
}

function ExpandableRow({
  label,
  cells,
  description,
}: {
  label: string;
  cells: [string, string, string];
  description: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <tr
        className="border-b border-border/30 cursor-pointer hover:bg-muted/5 transition-colors"
        onClick={() => setOpen((o) => !o)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setOpen((o) => !o);
          }
        }}
        aria-expanded={open}
      >
        <td className="py-3.5 px-5 text-muted-foreground">
          <span className="inline-flex items-center gap-2">
            <ChevronRight
              className={`h-4 w-4 shrink-0 transition-transform duration-200 ${open ? "rotate-90" : ""}`}
              aria-hidden
            />
            {label}
          </span>
        </td>
        {cells.map((cell, i) => (
          <td
            key={`${label}-${i}`}
            className="py-3.5 px-3 text-center"
            style={{ backgroundColor: columnTints[i] }}
          >
            {cell === "check" ? (
              <Check className="inline-block h-4 w-4 text-green-500" aria-hidden />
            ) : cell === "x" ? (
              <X className="inline-block h-4 w-4 text-muted-foreground/40" aria-hidden />
            ) : (
              <span className="text-xs text-muted-foreground">{cell}</span>
            )}
          </td>
        ))}
      </tr>
      {open && (
        <tr className="border-b border-border/30 bg-muted/5">
          <td colSpan={4} className="px-5 py-3 pl-11 text-sm text-muted-foreground">
            {description}
          </td>
        </tr>
      )}
    </>
  );
}

export function CompareProductsTable() {
  return (
    <section className="pb-16 md:pb-24">
      <h2 className="font-heading text-2xl font-semibold text-foreground mb-8 text-center sm:text-left">
        Compare products
      </h2>
      <div className="rounded-xl border border-border/60 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[480px] text-sm">
            <thead>
              <tr className="border-b border-border/40">
                <th className="text-left py-4 px-5 text-sm font-medium text-muted-foreground w-[40%]">
                  Feature
                </th>
                <th className="text-center py-4 px-3 w-[20%]">
                  <div className="text-sm font-semibold text-foreground">RunMat</div>
                  <div className="text-xs font-normal text-muted-foreground mt-0.5">Essential runtime features</div>
                </th>
                <th className="text-center py-4 px-3 w-[20%]">
                  <div className="text-sm font-semibold text-foreground">Cloud</div>
                  <div className="text-xs font-normal text-muted-foreground mt-0.5">Cloud-based solution</div>
                </th>
                <th className="text-center py-4 px-3 w-[20%]">
                  <div className="text-sm font-semibold text-foreground">Enterprise</div>
                  <div className="text-xs font-normal text-muted-foreground mt-0.5">Enterprise deployment</div>
                </th>
              </tr>
            </thead>
            <tbody>
              {productSectionLabels.map((sectionLabel, sectionIdx) => (
                <Fragment key={sectionLabel}>
                  <SectionHeaderRow label={sectionLabel} />
                  {productRows
                    .slice(
                      productSectionSizes.slice(0, sectionIdx).reduce((a, b) => a + b, 0),
                      productSectionSizes.slice(0, sectionIdx + 1).reduce((a, b) => a + b, 0)
                    )
                    .map(([label, ...rest]) => {
                      const cells = rest.slice(0, 3) as [string, string, string];
                      const description = rest[3] as string;
                      return (
                        <ExpandableRow
                          key={label}
                          label={label}
                          cells={cells}
                          description={description}
                        />
                      );
                    })}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

export function CompareCloudTable() {
  return (
    <section className="pb-20 md:pb-28">
      <h2 className="font-heading text-2xl font-semibold text-foreground mb-8 text-center sm:text-left">
        Compare Cloud plans
      </h2>
      <div className="rounded-xl border border-border/60 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[460px] text-sm">
            <thead>
              <tr className="border-b border-border/40">
                <th className="text-left py-4 px-5 text-sm font-medium text-muted-foreground w-[34%]">
                  Feature
                </th>
                <th className="text-center py-4 px-3 w-[22%]">
                  <div className="text-sm font-semibold text-foreground">Free</div>
                  <div className="text-xs font-normal text-muted-foreground mt-0.5">Get started</div>
                </th>
                <th className="text-center py-4 px-3 w-[22%]">
                  <div className="text-sm font-semibold text-foreground">Pro ($30/mo per user)</div>
                  <div className="text-xs font-normal text-muted-foreground mt-0.5">For individuals and teams</div>
                </th>
                <th className="text-center py-4 px-3 w-[22%]">
                  <div className="text-sm font-semibold text-foreground">Team ($100/mo per user)</div>
                  <div className="text-xs font-normal text-muted-foreground mt-0.5">Organization-wide</div>
                </th>
              </tr>
            </thead>
            <tbody>
              {cloudSectionLabels.map((sectionLabel, sectionIdx) => (
                <Fragment key={sectionLabel}>
                  <SectionHeaderRow label={sectionLabel} />
                  {cloudRows
                    .slice(
                      cloudSectionSizes.slice(0, sectionIdx).reduce((a, b) => a + b, 0),
                      cloudSectionSizes.slice(0, sectionIdx + 1).reduce((a, b) => a + b, 0)
                    )
                    .map(([label, ...rest]) => {
                      const cells = rest.slice(0, 3) as [string, string, string];
                      const description = rest[3] as string;
                      return (
                        <ExpandableRow
                          key={label}
                          label={label}
                          cells={cells}
                          description={description}
                        />
                      );
                    })}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
