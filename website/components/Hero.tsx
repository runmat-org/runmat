import Link from "next/link";
import type { ReactNode } from "react";
import LazyVideo from "@/components/LazyVideo";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface HeroLink {
  label: ReactNode;
  href: string;
}

interface HeroProps {
  title: string;
  description: string;
  primaryCta: HeroLink;
  secondaryCta: HeroLink;
  supportingLinks: {
    prefix: string;
    links: HeroLink[];
    suffix: string;
  };
  media: {
    label: string;
    note?: string;
    tone?: "muted" | "brand" | "surface";
    image?: string;
    poster?: string;
    video?: string;
  };
}

const mediaToneClasses = {
  muted: "border-border bg-muted text-foreground",
  brand: "border-brand/30 bg-brand/15 text-foreground",
  surface: "border-border bg-card text-card-foreground",
} as const;

export default function Hero({
  title,
  description,
  primaryCta,
  secondaryCta,
  supportingLinks,
  media,
}: HeroProps) {
  return (
    <section
      className="relative isolate -mt-px overflow-hidden bg-background pt-24 text-foreground md:pt-32"
      id="hero"
    >
      <div className="pointer-events-none" />
      <div className="container mx-auto px-4 md:px-6">
        <div className="mx-auto flex max-w-3xl flex-col items-center text-center">
          <h1 className="text-5xl font-semibold tracking-tight sm:text-6xl md:text-7xl">
            {title}
          </h1>
          <p className="mt-5 max-w-2xl text-sm leading-6 text-muted-foreground sm:text-base">
            {description}
          </p>
          <div className="mt-7 flex flex-col items-center justify-center gap-3 sm:flex-row">
            <Button
              size="lg"
              asChild
              className="h-11 rounded-none border-0 bg-[hsl(var(--brand))] px-7 text-sm font-semibold text-white shadow-none hover:bg-[hsl(var(--brand))]/90"
            >
              <Link href={primaryCta.href}>{primaryCta.label}</Link>
            </Button>
            <Button
              size="lg"
              variant="outline"
              asChild
              className="h-11 rounded-none border-border bg-card px-7 text-sm font-medium text-foreground shadow-none hover:bg-accent hover:text-accent-foreground"
            >
              <Link href={secondaryCta.href}>{secondaryCta.label}</Link>
            </Button>
          </div>
          <p className="mt-5 text-xs text-muted-foreground">
            {supportingLinks.prefix}{" "}
            {supportingLinks.links.map((link, index) => (
              <span key={link.href}>
                <Link href={link.href} className="underline underline-offset-2">
                  {link.label}
                </Link>
                {index < supportingLinks.links.length - 1 ? " and " : ""}
              </span>
            ))}
            {supportingLinks.suffix}
          </p>
        </div>

        <div className="mx-auto mt-24 max-w-5xl pb-20 md:mt-32">
          <div
            role="img"
            aria-label={media.label}
            className={cn(
              "relative aspect-video w-full overflow-hidden rounded-3xl border",
              mediaToneClasses[media.tone ?? "muted"],
            )}
          >
            {media.video ? (
              <LazyVideo
                className="absolute inset-0 h-full w-full object-cover"
                muted
                loop
                playsInline
                initialPosterVariant="poster"
                poster={media.poster}
                aria-label={media.label}
              >
                <source src={media.video} type="video/mp4" />
              </LazyVideo>
            ) : media.image ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={media.image}
                alt={media.label}
                className="absolute inset-0 h-full w-full object-cover"
              />
            ) : null}
          </div>
        </div>
      </div>
    </section>
  );
}
