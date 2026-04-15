"use client";

import Link from "next/link";
import Image from "next/image";
import { ArrowUpRight } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

const GRID_PATTERNS = [
  "/patterns/grid-blue.svg",
  "/patterns/grid-green.svg",
  "/patterns/grid-violet.svg",
  "/patterns/grid-scatter.svg",
  "/patterns/grid-heatmap.svg",
  "/patterns/grid-vector.svg",
  "/patterns/grid-histogram.svg",
  "/patterns/grid-bar.svg",
  "/patterns/grid-stem.svg",
  "/patterns/grid-surf.svg",
  "/patterns/grid-teal.svg",
  "/patterns/grid-indigo.svg",
  "/patterns/grid-coral.svg",
  "/patterns/grid-navy.svg",
  "/patterns/grid-sage.svg",
  "/patterns/grid-plum.svg",
];

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
  const pattern = GRID_PATTERNS[index % GRID_PATTERNS.length];

  return (
    <Link href={href} className="block h-full group">
      <Card className="group overflow-hidden transition-all cursor-pointer h-full flex flex-col bg-muted/50 hover:bg-card py-0 gap-0">
        <div className="relative w-full h-44 overflow-hidden flex-shrink-0">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={pattern}
            alt=""
            className="absolute inset-0 w-full h-full object-cover"
          />
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
