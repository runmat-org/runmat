import Link from "next/link";
import type { ReactNode } from "react";
import LazyVideo from "@/components/LazyVideo";
import { TryInBrowserLink } from "@/components/TryInBrowserButton";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export type MediaTone = "muted" | "brand" | "surface";

export const mediaToneClasses: Record<MediaTone, string> = {
  muted: "bg-muted text-foreground",
  brand: "bg-brand/15 text-foreground",
  surface: "bg-card text-card-foreground",
};

const mediaToneBorderClasses: Record<MediaTone, string> = {
  muted: "border-border",
  brand: "border-brand/30",
  surface: "border-border",
};

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
    tone?: MediaTone;
    image?: string;
    poster?: string;
    video?: string;
    link?: {
      href?: string;
      ariaLabel: string;
      code?: string;
      agentPrompt?: string;
      source?: string;
      exampleId?: string;
    };
  };
}

export default function Hero({
  title,
  description,
  primaryCta,
  secondaryCta,
  supportingLinks,
  media,
}: HeroProps) {
  const mediaBlock = (
    <div
      role="img"
      aria-label={media.label}
      className={cn(
        "relative aspect-video w-full overflow-hidden rounded-3xl border",
        mediaToneClasses[media.tone ?? "muted"],
        mediaToneBorderClasses[media.tone ?? "muted"],
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
  );

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
          {media.link ? (
            <TryInBrowserLink {...media.link} className="rounded-3xl">
              {mediaBlock}
            </TryInBrowserLink>
          ) : (
            mediaBlock
          )}
        </div>
      </div>
    </section>
  );
}
