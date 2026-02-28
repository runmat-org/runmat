import { Metadata } from "next";
import Link from "next/link";
import { ArrowRight, BookOpen, Newspaper, Sparkles } from "lucide-react";

import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { ContentCard } from "@/components/content-card";
import {
  getDisplayResourceTypes,
  getFeaturedResources,
  getLatestResources,
  resourceTypeLabel,
  getResourceTypeLink,
} from "@/lib/resources";
import { typeColor } from "@/lib/type-colors";

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
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-16 md:px-6 md:py-24 space-y-12">
        {/* Hero */}
        <div className="mx-auto max-w-[58rem] text-center">
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl">
            Resource Hub
          </h1>
          <p className="mt-6 text-base text-muted-foreground sm:text-lg">
            Hand-picked guides, docs, and benchmarks to learn RunMat.
          </p>
        </div>

        {/* Browse by type */}
        <section className="space-y-6">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <h2 className="text-2xl font-semibold text-foreground">Browse by type</h2>
            <span className="text-sm text-muted-foreground">Jump to a category</span>
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
            <h2 className="text-2xl font-semibold text-foreground">Featured</h2>
            <span className="text-sm text-muted-foreground">Hand-picked highlights</span>
          </div>
          {featured.length ? (
            <div className="grid gap-6 lg:grid-cols-3">
              {featured.map((item) => (
                <ContentCard
                  key={item.id}
                  href={item.href ?? "#"}
                  title={item.title}
                  image={item.image}
                  imageAlt={item.imageAlt}
                  typeBadge={{ label: resourceTypeLabel(item.type) }}
                  excerpt={item.description}
                  ctaLabel="Read"
                />
              ))}
            </div>
          ) : (
            <EmptyState />
          )}
        </section>

        {/* Latest */}
        <section id="latest" className="space-y-6">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <h2 className="text-2xl font-semibold text-foreground">Latest</h2>
            <span className="text-sm text-muted-foreground">Most recent across all types</span>
          </div>
          {latest.length ? (
            <div className="grid gap-4 lg:grid-cols-3 sm:grid-cols-2">
              {latest.map((item) => (
                <ContentCard
                  key={item.id}
                  href={item.href ?? "#"}
                  title={item.title}
                  image={item.image}
                  imageAlt={item.imageAlt}
                  typeBadge={{ label: resourceTypeLabel(item.type) }}
                  excerpt={item.description}
                  ctaLabel="View"
                />
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
      <Card className="h-full bg-muted/5 hover:bg-muted/50 transition-all rounded-lg p-5 text-left">
        <CardHeader className="space-y-3 p-0">
          <div
            className="w-10 h-10 rounded-md flex items-center justify-center transition-colors"
            style={{ color: color.text, backgroundColor: color.bg }}
          >
            {icon}
          </div>
          <div className="flex items-center justify-between w-full">
            <span className="text-sm font-medium text-foreground">{label}</span>
            <ArrowRight className="w-4 h-4 text-muted-foreground group-hover:text-foreground group-hover:translate-x-1 transition-all" />
          </div>
        </CardHeader>
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

