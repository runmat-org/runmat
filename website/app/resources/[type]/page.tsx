import { Metadata } from "next";
import { notFound } from "next/navigation";

import { Card, CardContent } from "@/components/ui/card";
import { ContentCard } from "@/components/content-card";
import { shouldShowImage } from "@/lib/thumbnail-config";
import {
  getResourcesByType,
  resourceTypeLabel,
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
  const path = `https://runmat.com/resources/${type}`;
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
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-16 md:px-6 md:py-24 space-y-10">
        <div className="mx-auto max-w-[58rem] text-center">
          <h1 className="font-heading text-3xl leading-[1.1] sm:text-3xl md:text-5xl">
            {label}
          </h1>
        </div>

        {items.length === 0 ? (
          <Card>
            <CardContent className="p-6 text-muted-foreground">
              No resources found for this type yet.
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {items.map((item, i) => {
              const useImg = shouldShowImage(item.slug);
              return (
                <ContentCard
                  key={item.id}
                  href={item.href ?? "#"}
                  title={item.title}
                  image={useImg ? item.image : undefined}
                  imageAlt={useImg ? item.imageAlt : undefined}
                  excerpt={item.description}
                  ctaLabel="View"
                  index={i}
                />
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

function normalizeType(input: string, allowed: ResourceType[]) {
  if (!Array.isArray(allowed) || allowed.length === 0) return null;
  const t = input.toLowerCase();
  return allowed.includes(t as ResourceType) ? (t as ResourceType) : null;
}

