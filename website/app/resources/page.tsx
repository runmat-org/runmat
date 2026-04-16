import { Metadata } from "next";
import Link from "next/link";

import { Card, CardContent } from "@/components/ui/card";
import { ContentCard } from "@/components/content-card";
import {
  getDisplayResourceTypes,
  getFeaturedResources,
  getLatestResources,
  resourceTypeLabel,
  getResourceTypeLink,
} from "@/lib/resources";

export const metadata: Metadata = {
  title: "Resources | RunMat",
  description:
    "Guides, Q&A, case studies, news, and curated docs/benchmarks to learn RunMat. Hand-picked, never auto-indexed.",
  alternates: { canonical: "https://runmat.com/resources" },
  openGraph: {
    title: "RunMat Resources",
    description: "Guides, Q&A, case studies, news, and curated docs/benchmarks.",
    url: "https://runmat.com/resources",
    type: "website",
  },
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
          <h1 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
            Resource Hub
          </h1>
        </div>

        {/* Browse by type */}
        <section className="space-y-6">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <h2 className="text-2xl font-semibold text-foreground">Browse by type</h2>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
            {typesWithItems.map((type) => (
              <TypeCard
                key={type.type}
                href={type.href}
                label={type.label}
              />
            ))}
          </div>
        </section>

        {/* Featured */}
        <section id="featured" className="space-y-6">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <h2 className="text-2xl font-semibold text-foreground">Featured</h2>
          </div>
          {featured.length ? (
            <div className="grid gap-6 lg:grid-cols-3">
              {featured.map((item, i) => (
                <ContentCard
                  key={item.id}
                  href={item.href ?? "#"}
                  title={item.title}
                  typeBadge={{ label: resourceTypeLabel(item.type) }}
                  excerpt={item.description}
                  ctaLabel="Read"
                  index={i}
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
          </div>
          {latest.length ? (
            <div className="grid gap-4 lg:grid-cols-3 sm:grid-cols-2">
              {latest.map((item, i) => (
                <ContentCard
                  key={item.id}
                  href={item.href ?? "#"}
                  title={item.title}
                  typeBadge={{ label: resourceTypeLabel(item.type) }}
                  excerpt={item.description}
                  ctaLabel="View"
                  index={i + featured.length}
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
  label,
}: {
  href: string;
  label: string;
}) {
  return (
    <Link href={href} className="block border border-border rounded-lg px-4 py-3 text-sm font-medium text-foreground hover:bg-accent transition-colors">
      {label}
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

