import type { Metadata } from "next";

const SITE_ORIGIN = "https://runmat.com";
const SITE_NAME = "RunMat";
// Default OG image served from app/opengraph-image.tsx at the site root.
const DEFAULT_OG_IMAGE_PATH = "/opengraph-image";
const DEFAULT_OG_IMAGE_ALT = "RunMat — open-source MATLAB-compatible runtime";

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
  ogImageAlt?: string;
};

export function buildPageMetadata(opts: PageSeoOptions): Metadata {
  const canonical = toAbsoluteUrl(opts.canonicalPath);
  const ogImagePath = opts.ogImagePath ?? DEFAULT_OG_IMAGE_PATH;
  const ogImageAlt = opts.ogImageAlt ?? DEFAULT_OG_IMAGE_ALT;
  const ogImage = toAbsoluteUrl(ogImagePath);
  const ogType = opts.ogType ?? "website";
  const ogImageEntry = [{ url: ogImage, ...(ogImageAlt ? { alt: ogImageAlt } : {}) }];

  return {
    title: opts.title,
    description: opts.description,
    alternates: { canonical },
    openGraph: {
      title: opts.title,
      description: opts.description,
      type: ogType,
      url: canonical,
      siteName: SITE_NAME,
      images: ogImageEntry,
    },
    twitter: {
      card: "summary_large_image",
      title: opts.title,
      description: opts.description,
      images: ogImageEntry,
    },
  };
}
