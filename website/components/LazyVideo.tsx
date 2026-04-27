"use client";

import { useEffect, useRef, useState, type ComponentProps } from "react";

type VideoProps = Omit<ComponentProps<"video">, "autoPlay" | "preload"> & {
  mobilePoster?: string;
  mobileMediaQuery?: string;
  initialPosterVariant?: "none" | "poster" | "mobilePoster";
};

/**
 * A video element that defers loading until it scrolls into view.
 *
 * Unlike a plain `<video autoPlay preload="none">`, where `autoplay`
 * overrides `preload` and forces an immediate download, this component
 * keeps `preload="none"` effective by only calling `.play()` once the
 * element enters the viewport via Intersection Observer.
 *
 * When `mobilePoster` is provided, it is used for viewports matching
 * `mobileMediaQuery` (default: "(max-width: 768px)") so mobile clients
 * pull a smaller still image as their LCP candidate.
 */
export default function LazyVideo({
  mobilePoster,
  mobileMediaQuery = "(max-width: 768px)",
  initialPosterVariant = "none",
  poster,
  ...props
}: VideoProps) {
  const ref = useRef<HTMLVideoElement>(null);
  const [activePoster, setActivePoster] = useState<string | undefined>(() => {
    if (!mobilePoster) return poster;
    if (initialPosterVariant === "poster") return poster;
    if (initialPosterVariant === "mobilePoster") return mobilePoster;

    // Default: no initial poster to avoid SSR baking the desktop URL into HTML,
    // which can cause mobile browsers to fetch both desktop and mobile posters.
    return undefined;
  });

  useEffect(() => {
    if (!mobilePoster) return;
    const media = window.matchMedia(mobileMediaQuery);
    const update = () => setActivePoster(media.matches ? mobilePoster : poster);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, [mobilePoster, mobileMediaQuery, poster]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.play().catch(() => {});
          observer.disconnect();
        }
      },
      { rootMargin: "200px" },
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return <video ref={ref} preload="none" poster={activePoster} {...props} />;
}
