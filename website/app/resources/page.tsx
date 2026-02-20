import { Metadata } from "next";
import Link from "next/link";
import { ArrowRight, ArrowUpRight, BookOpen, Newspaper, Sparkles } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  getDisplayResourceTypes,
  getFeaturedResources,
  getLatestResources,
  resourceTypeLabel,
  ResourceItem,
  getResourceTypeLink,
} from "@/lib/resources";

export const metadata: Metadata = {
  title: "Resources | RunMat",
  description:
    "Guides, Q&A, case studies, news, and curated docs/benchmarks to learn RunMat. Hand-picked, never auto-indexed.",
  alternates: { canonical: "https://runmat.org/resources" },
  openGraph: {
    title: "RunMat Resources",
    description: "Guides, Q&A, case studies, news, and curated docs/benchmarks.",
    url: "https://runmat.org/resources",
    type: "website",
  },
};

const TYPE_ICONS: Record<string, React.ReactNode> = {
  docs: <BookOpen className="h-4 w-4" />,
  guides: <Sparkles className="h-4 w-4" />,
  blogs: <Newspaper className="h-4 w-4" />,
  "case-studies": <Sparkles className="h-4 w-4" />,
  webinars: <Sparkles className="h-4 w-4" />,
  benchmarks: <Sparkles className="h-4 w-4" />,
};

export default function ResourcesPage() {
  const featured = getFeaturedResources().slice(0, 3);
  const latest = getLatestResources(9);

  const typesWithItems = getDisplayResourceTypes().map((type) => ({
    type,
    label: resourceTypeLabel(type),
    href: getResourceTypeLink(type),
  }));

  return (
    <div className="min-h-screen bg-[#0a0a0f]">
      <div className="mx-auto max-w-[1400px] px-6 lg:px-12 py-20 lg:py-24 space-y-12">
        {/* Hero */}
        <section className="text-center space-y-3">
          <div className="flex items-center justify-center gap-2 text-gray-400 text-sm">
            <Badge
              variant="secondary"
              className="px-2 py-0.5 text-[11px] uppercase tracking-wider bg-white/5 text-white border-white/10"
            >
              Resources
            </Badge>
            <span>Guides, Q&A, case studies, news</span>
          </div>
          <h1 className="text-5xl lg:text-6xl font-semibold tracking-tight text-white">
            Resource Hub
          </h1>
        </section>

        {/* Browse by type */}
        <section className="space-y-6">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <h2 className="text-2xl font-semibold text-white">Browse by type</h2>
            <span className="text-sm text-gray-400">Jump to a category</span>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
            {typesWithItems.map((type) => (
              <TypeCard
                key={type.type}
                href={type.href}
                icon={TYPE_ICONS[type.type] ?? <Sparkles className="h-5 w-5" />}
                label={type.label}
                type={type.type}
              />
            ))}
          </div>
        </section>

        {/* Featured */}
        <section id="featured" className="space-y-6">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <h2 className="text-2xl font-semibold text-white">Featured</h2>
            <span className="text-sm text-gray-400">Hand-picked highlights</span>
          </div>
          {featured.length ? (
            <div className="grid gap-6 lg:grid-cols-3">
              {featured.map((item) => (
                <FeaturedResourceCard key={item.id} item={item} />
              ))}
            </div>
          ) : (
            <EmptyState />
          )}
        </section>

        {/* Latest */}
        <section id="latest" className="space-y-6">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <h2 className="text-2xl font-semibold text-white">Latest</h2>
            <span className="text-sm text-gray-400">Most recent across all types</span>
          </div>
          {latest.length ? (
            <div className="grid gap-4 lg:grid-cols-3 sm:grid-cols-2">
              {latest.map((item) => (
                <ResourceListCard key={item.id} item={item} />
              ))}
            </div>
          ) : (
            <EmptyState />
          )}
        </section>
      </div>
    </div>
  );
}

function TypeCard({
  href,
  icon,
  label,
  type,
}: {
  href: string;
  icon: React.ReactNode;
  label: string;
  type: string;
}) {
  const color = typeColor(type);
  return (
    <Link href={href} className="block h-full group">
      <Card className="h-full border border-white/10 bg-white/5 hover:bg-white/10 transition-all rounded-lg p-5 text-left">
        <CardHeader className="space-y-3 p-0">
          <div
            className="w-10 h-10 rounded-md flex items-center justify-center bg-white/5 transition-colors"
            style={{ color: color.text, backgroundColor: color.bg }}
          >
            {icon}
          </div>
          <div className="flex items-center justify-between w-full">
            <span className="text-sm font-medium text-white">{label}</span>
            <ArrowRight className="w-4 h-4 text-gray-500 group-hover:text-white group-hover:translate-x-1 transition-all" />
          </div>
        </CardHeader>
      </Card>
    </Link>
  );
}

function FeaturedResourceCard({ item }: { item: ResourceItem }) {
  const color = typeColor(item.type);
  return (
    <Link href={item.href || "#"} className="block h-full group">
      <Card className="h-full border border-white/10 bg-white/[0.03] hover:bg-white/[0.06] transition-all rounded-xl p-5 cursor-pointer flex flex-col gap-4">
        <CardHeader className="space-y-3 p-0">
          <div className="text-xs uppercase tracking-wider mb-1" style={{ color: color.text }}>
            {resourceTypeLabel(item.type)}
          </div>
          <CardTitle className="text-xl font-semibold leading-tight text-white mb-1 transition-colors">
            {item.title}
          </CardTitle>
          <CardDescription className="line-clamp-3 text-sm text-gray-400">{item.description}</CardDescription>
        </CardHeader>
        <CardContent className="flex items-center justify-end text-sm pt-4 pb-0 px-0">
          <span className="inline-flex items-center gap-2 text-sm transition-all" style={{ color: color.text }}>
            Read <ArrowUpRight className="h-4 w-4" />
          </span>
        </CardContent>
      </Card>
    </Link>
  );
}

function ResourceListCard({ item }: { item: ResourceItem }) {
  const color = typeColor(item.type);
  return (
    <Link href={item.href || "#"} className="block h-full group">
      <Card className="h-full border border-white/[0.06] bg-white/[0.02] hover:bg-white/[0.04] transition-all rounded-lg p-5 cursor-pointer">
        <CardHeader className="space-y-2 p-0">
          <div className="text-xs uppercase tracking-wider mb-1" style={{ color: color.text }}>
            {resourceTypeLabel(item.type)}
          </div>
          <CardTitle className="text-base font-semibold leading-snug line-clamp-2 text-white">
            {item.title}
          </CardTitle>
          <CardDescription className="line-clamp-2 text-sm text-gray-400 mb-1">
            {item.description}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex items-center justify-end text-xs pt-0 pb-0 px-0">
          <span className="inline-flex items-center gap-1 text-xs transition-all" style={{ color: color.text }}>
            View <ArrowUpRight className="h-3 w-3" />
          </span>
        </CardContent>
      </Card>
    </Link>
  );
}

function EmptyState() {
  return (
    <Card>
      <CardContent className="p-6 text-muted-foreground text-sm">No resources available yet.</CardContent>
    </Card>
  );
}

function typeColor(type: string) {
  switch (type) {
    case "docs":
      return { text: "#a78bfa", bg: "rgba(167,139,250,0.1)" };
    case "guides":
      return { text: "#60a5fa", bg: "rgba(96,165,250,0.1)" };
    case "blogs":
      return { text: "#fb923c", bg: "rgba(251,146,60,0.1)" };
    case "case-studies":
      return { text: "#4ade80", bg: "rgba(74,222,128,0.1)" };
    case "webinars":
      return { text: "#f472b6", bg: "rgba(244,114,182,0.1)" };
    case "benchmarks":
      return { text: "#22d3ee", bg: "rgba(34,211,238,0.1)" };
    default:
      return { text: "#9ca3af", bg: "rgba(255,255,255,0.05)" };
  }
}

