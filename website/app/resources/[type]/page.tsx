import { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft, ArrowUpRight } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  getResourcesByType,
  resourceTypeLabel,
  ResourceItem,
  ResourceType,
  getRoutableResourceTypes,
  getGuidesCollection,
} from "@/lib/resources";

type Params = { type: string };

export function generateStaticParams() {
  return getRoutableResourceTypes().map((type) => ({ type }));
}

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const { type: rawType } = await params;
  const available = getRoutableResourceTypes();
  const type = normalizeType(rawType, available);
  if (!type) return {};
  const label = resourceTypeLabel(type);
  const path = `https://runmat.org/resources/${type}`;
  return {
    title: `${label} | RunMat Resources`,
    description: `${label} curated resources for RunMat.`,
    alternates: { canonical: path },
    openGraph: {
      title: `${label} | RunMat Resources`,
      description: `${label} curated resources for RunMat.`,
      url: path,
      type: "website",
    },
  };
}

export default async function ResourcesByTypePage({ params }: { params: Promise<Params> }) {
  const { type: rawType } = await params;
  const available = getRoutableResourceTypes();
  const type = normalizeType(rawType, available);
  if (!type) notFound();

  const items =
    type === "guides" ? getGuidesCollection() : getResourcesByType(type);
  const label = resourceTypeLabel(type);

  return (
    <div className="min-h-screen bg-[#0a0a0f]">
      <div className="mx-auto max-w-[1400px] px-6 lg:px-12 py-20 lg:py-24 space-y-10">
        <div className="flex items-center gap-3">
          <Link href="/resources" className="inline-flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors">
            <ArrowLeft className="h-4 w-4" />
            Back to Resources
          </Link>
          <Badge variant="secondary" className="capitalize bg-white/5 text-white border-white/10">
            {label}
          </Badge>
        </div>
        <div className="space-y-2">
          <h1 className="text-4xl lg:text-5xl font-semibold tracking-tight text-white">{label}</h1>
          <p className="text-sm text-gray-400">
            Guides and walkthroughs curated for this category.
          </p>
        </div>

        {items.length === 0 ? (
          <Card>
            <CardContent className="p-6 text-muted-foreground">
              No resources found for this type yet.
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {items.map((item) => (
              <ResourceCard key={item.id} item={item} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ResourceCard({ item }: { item: ResourceItem }) {
  const color = typeColor(item.type);
  return (
    <Link href={item.href || "#"} className="block h-full group">
      <Card className="h-full border border-white/[0.06] bg-white/[0.02] hover:bg-white/[0.04] transition-all rounded-lg p-5 cursor-pointer">
        <CardHeader className="space-y-2 p-0">
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

function normalizeType(input: string, allowed: ResourceType[]) {
  if (!Array.isArray(allowed) || allowed.length === 0) return null;
  const t = input.toLowerCase();
  return allowed.includes(t as ResourceType) ? (t as ResourceType) : null;
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

