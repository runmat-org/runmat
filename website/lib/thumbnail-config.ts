export const SLUGS_WITH_THUMBNAIL = new Set([
  "introducing-runmat-cloud",
  "matlab-plotting-guide",
  "matlab-alternatives",
  "why-we-built-runmat",
  "in-defense-of-matlab-whiteboard-style-code",
  "introducing-runmat",
  "why-rust",
  "runmat-accel-intro-blog",
]);

export function shouldShowImage(slug: string): boolean {
  return SLUGS_WITH_THUMBNAIL.has(slug);
}
