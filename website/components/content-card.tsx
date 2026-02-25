"use client";

import Link from "next/link";
import Image from "next/image";
import { ArrowUpRight } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

export interface ContentCardProps {
  href: string;
  title: string;
  image?: string;
  imageAlt?: string;
  typeBadge?: { label: string; color?: string };
  excerpt?: string;
  date?: string;
  ctaLabel?: string;
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
}: ContentCardProps) {
  return (
    <Link href={href} className="block h-full group">
      <Card className="group overflow-hidden transition-all hover:shadow-lg cursor-pointer h-full flex flex-col">
        <CardContent className="p-5 flex flex-col h-full gap-0">
          {image ? (
            <div className="relative w-full h-40 rounded-lg mb-3 overflow-hidden flex-shrink-0 bg-muted/10 border border-border/50">
              <Image
                src={image}
                alt={imageAlt ?? title}
                fill
                className="object-contain transition-transform duration-300 group-hover:scale-105"
                sizes="(max-width: 768px) 100vw, (max-width: 1024px) 50vw, 33vw"
              />
            </div>
          ) : (
            <div className="w-full h-40 rounded-lg mb-3 flex-shrink-0 bg-gradient-to-br from-purple-500 via-purple-600 to-blue-600 group-hover:from-purple-600 group-hover:via-purple-700 group-hover:to-blue-700 transition-all border border-border/50" />
          )}

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
            <p className="text-sm text-muted-foreground line-clamp-2 leading-snug mb-2">
              {excerpt}
            </p>
          )}

          {date && (
            <p className="text-sm text-muted-foreground mb-3">
              {new Date(date).toLocaleDateString("en-US", {
                year: "numeric",
                month: "short",
                day: "numeric",
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
