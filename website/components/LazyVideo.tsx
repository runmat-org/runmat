"use client";

import { useEffect, useRef, type ComponentProps } from "react";

type VideoProps = Omit<ComponentProps<"video">, "autoPlay" | "preload">;

/**
 * A video element that defers loading until it scrolls into view.
 *
 * Unlike a plain `<video autoPlay preload="none">`, where `autoplay`
 * overrides `preload` and forces an immediate download, this component
 * keeps `preload="none"` effective by only calling `.play()` once the
 * element enters the viewport via Intersection Observer.
 */
export default function LazyVideo(props: VideoProps) {
  const ref = useRef<HTMLVideoElement>(null);

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

  return <video ref={ref} preload="none" {...props} />;
}
