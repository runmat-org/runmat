import Link from "next/link";
import { Button } from "@/components/ui/button";
import Media, { type MediaTone } from "@/components/Media";

interface HeroLink {
  label: string;
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
              className="h-10 rounded-full bg-primary px-6 text-sm font-medium text-primary-foreground shadow-none hover:bg-primary/90"
            >
              <Link href={primaryCta.href}>{primaryCta.label}</Link>
            </Button>
            <Button
              size="lg"
              variant="secondary"
              asChild
              className="h-10 rounded-full bg-secondary px-6 text-sm font-medium text-secondary-foreground shadow-none hover:bg-secondary/80"
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
          <Media
            label={media.label}
            note={media.note}
            tone={media.tone}
            className="min-h-[360px] rounded-3xl shadow-2xl shadow-foreground/10"
          />
        </div>
      </div>
    </section>
  );
}
