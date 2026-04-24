"use client";

import { useEffect, useRef, useState, type ComponentProps } from "react";

type VideoProps = Omit<ComponentProps<"video">, "autoPlay" | "preload"> & {
  mobilePoster?: string;
  mobileMediaQuery?: string;
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
  poster,
  ...props
}: VideoProps) {
  const ref = useRef<HTMLVideoElement>(null);
  const [activePoster, setActivePoster] = useState<string | undefined>(poster);

  useEffect(() => {
    if (!mobilePoster || typeof window === "undefined") return;
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
