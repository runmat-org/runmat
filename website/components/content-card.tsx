"use client";

import Link from "next/link";
import { ArrowUpRight, type LucideIcon, BarChart3, Cloud, Bug, Zap, Scale, Shield, GitBranch, Database, RotateCcw, Cpu, Rocket, Lightbulb, BookOpen, Braces, Gauge, PenTool, FileText, FlaskConical, Settings, Globe, Layers, Terminal } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { CARD_PATTERNS } from "@/components/card-patterns";

const ICON_KEYWORDS: [string[], LucideIcon][] = [
  [["plot", "chart", "figure", "graph", "visualization"], BarChart3],
  [["cloud", "durable", "state"], Cloud],
  [["debug", "fprintf", "worst"], Bug],
  [["loop", "slow", "performance", "fast"], Zap],
  [["alternative", "compare", "benchmark", "vs"], Scale],
  [["airgap", "mission", "critical", "security"], Shield],
  [["version", "git", "control"], GitBranch],
  [["checkpoint", "persist", "data", "storage"], Database],
  [["restor", "historical", "run state"], RotateCcw],
  [["gpu", "nvidia", "cuda", "accelerat"], Cpu],
  [["introduc", "launch", "announc"], Rocket],
  [["why", "built", "story"], Lightbulb],
  [["what is", "learn", "guide", "tutorial"], BookOpen],
  [["rust", "code", "language", "llm"], Braces],
  [["accel", "runtime", "fastest"], Gauge],
  [["defense", "whiteboard", "style"], PenTool],
  [["doc", "reference", "api"], FileText],
  [["test", "experiment", "lab"], FlaskConical],
  [["config", "setup", "install"], Settings],
  [["web", "browser", "online"], Globe],
  [["layer", "stack", "architect"], Layers],
  [["terminal", "cli", "command"], Terminal],
];

function getIconForTitle(title: string): LucideIcon {
  const lower = title.toLowerCase();
  for (const [keywords, icon] of ICON_KEYWORDS) {
    if (keywords.some((kw) => lower.includes(kw))) return icon;
  }
  return FileText;
}

export interface ContentCardProps {
  href: string;
  title: string;
  image?: string;
  imageAlt?: string;
  typeBadge?: { label: string; color?: string };
  excerpt?: string;
  date?: string;
  ctaLabel?: string;
  index?: number;
}

export function ContentCard({
  href,
  title,
  image,
  imageAlt,
  typeBadge,
  excerpt,
  date,
  ctaLabel = "Read",
  index = 0,
}: ContentCardProps) {
  const PatternSvg = CARD_PATTERNS[index % CARD_PATTERNS.length];
  const Icon = getIconForTitle(title);

  return (
    <Link href={href} className="block h-full group">
      <Card className="group overflow-hidden transition-all cursor-pointer h-full flex flex-col bg-muted/50 hover:bg-card py-0 gap-0">
        <div className="relative w-full h-56 overflow-hidden flex-shrink-0 flex items-center justify-center">
          {image ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={image}
              alt={imageAlt || title}
              className="absolute inset-0 w-full h-full object-cover object-top"
            />
          ) : (
            <>
              <PatternSvg />
              <Icon className="absolute z-10 size-[70px] text-gray-800/50 dark:text-gray-200/50" strokeWidth={1.2} />
            </>
          )}
        </div>

        <CardContent className="p-4 pt-3 flex flex-col flex-1 gap-0">
          {typeBadge && (
            <div
              className={`text-xs uppercase tracking-wider mb-2 ${typeBadge.color ? "" : "text-muted-foreground"}`}
              style={typeBadge.color ? { color: typeBadge.color } : undefined}
            >
              {typeBadge.label}
            </div>
          )}

          <h3 className="text-lg font-semibold leading-snug line-clamp-2 min-h-[2.75rem] mb-1.5">
            {title}
          </h3>

          {excerpt && (
            <p className="text-sm text-foreground line-clamp-2 leading-snug mb-2">
              {excerpt}
            </p>
          )}

          {date && (
            <p className="text-sm text-muted-foreground mb-3">
              {new Date(date).toLocaleDateString("en-US", {
                year: "numeric",
                month: "short",
                day: "numeric",
                timeZone: "UTC",
              })}
            </p>
          )}

          <div className="mt-auto flex items-center text-sm text-muted-foreground group-hover:text-primary transition-colors">
            {ctaLabel}
            <ArrowUpRight className="ml-1 h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
