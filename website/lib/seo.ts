import type { Metadata } from "next";

const SITE_ORIGIN = "https://runmat.com";

function toAbsoluteUrl(pathOrUrl: string): string {
  if (!pathOrUrl) return SITE_ORIGIN;
  if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) {
    return pathOrUrl;
  }
  return new URL(pathOrUrl, SITE_ORIGIN).toString();
}

export type PageSeoOptions = {
  title: string;
  description: string;
  canonicalPath: string; // e.g. "/docs/cli"
  ogType?: "website" | "article";
  ogImagePath?: string; // e.g. "/docs/opengraph-image"
};

export function buildPageMetadata(opts: PageSeoOptions): Metadata {
  const canonical = toAbsoluteUrl(opts.canonicalPath);
  const ogImage = opts.ogImagePath ? toAbsoluteUrl(opts.ogImagePath) : undefined;
  const ogType = opts.ogType ?? "website";

  return {
    title: opts.title,
    description: opts.description,
    alternates: { canonical },
    openGraph: {
      title: opts.title,
      description: opts.description,
      type: ogType,
      url: canonical,
      ...(ogImage ? { images: [ogImage] } : {}),
    },
    twitter: {
      card: "summary_large_image",
      title: opts.title,
      description: opts.description,
      ...(ogImage ? { images: [ogImage] } : {}),
    },
  };
}
