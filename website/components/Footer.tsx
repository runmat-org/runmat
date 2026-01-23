"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Heart } from "lucide-react";
import Image from "next/image";
import NewsletterCta from "@/components/NewsletterCta";

export default function Footer() {
  // Default to plain Dystr URL during SSR; hydrate with UTM params on client
  const [dystrHref, setDystrHref] = useState("https://dystr.com");
  useEffect(() => {
    try {
      const baseUrl = "https://dystr.com";
      const url = new URL(baseUrl);
      const params = new URLSearchParams(window.location.search);
      params.forEach((value, key) => {
        if (key.toLowerCase().startsWith("utm_") && value) {
          url.searchParams.append(key, value);
        }
      });
      setDystrHref(url.toString());
    } catch {
      // ignore; keep base URL
    }
  }, []);

  return (
    <footer className="bg-background">
      <div className="container mx-auto flex flex-col md:flex-row items-start justify-between gap-8 py-8 px-4 md:px-6 border-t">
        <div className="hidden md:flex flex-col gap-3">
          <div className="flex items-center gap-3">
            <Image
              src="/runmat-logo.svg"
              alt="RunMat"
              width={136}
              height={24}
              className="h-6 w-auto"
              priority
            />
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            The Fastest Runtime for Math
          </p>
        </div>
        <div className="grid w-full md:flex-1 gap-6 md:grid-cols-2 text-sm text-muted-foreground">
          <div>
            <h3 className="text-foreground font-semibold mb-2">Resources</h3>
            <ul className="space-y-1">
              <li><Link href="/license" className="hover:underline">License</Link></li>
              <li><Link href="/docs/telemetry" className="hover:underline">Telemetry</Link></li>
              <li><Link href="/matlab-online" className="hover:underline">MATLAB Online</Link></li>
            </ul>
          </div>
          <div>
            <h3 className="text-foreground font-semibold mb-2">Learn</h3>
            <ul className="space-y-1">
              <li><Link href="/blog" className="hover:underline">Blog</Link></li>
              <li><Link href="/benchmarks" className="hover:underline">Benchmarks</Link></li>
              <li><Link href="/docs/fusion-guide" className="hover:underline">Fusion Guide</Link></li>
            </ul>
          </div>
        </div>
        <div className="w-full md:w-auto md:flex-1 md:max-w-md">
          <NewsletterCta
            description="Get updates on releases, benchmarks, and deep dives."
            align="left"
            className="w-full"
          />
        </div>
      </div>
      <div className="border-t">
        <div className="container flex flex-col items-center justify-between gap-4 py-6 md:flex-row">
          <div className="text-sm text-center md:text-left" style={{ color: 'hsl(var(--muted-foreground))' }}>
            <p>
            © 2025 Dystr Inc. All rights reserved. MIT+ Licensed.
            </p>
            <p>
              MATLAB is a registered trademark of The MathWorks, Inc.
            </p>
            <p>
              RunMat is not affiliated with, endorsed by, or sponsored by The MathWorks, Inc. or the Free Software Foundation.
            </p>
          </div>
          <p className="flex items-center text-center text-sm md:text-left" style={{ color: 'hsl(var(--muted-foreground))' }}>
            Made with
            <Heart className="mx-1 h-4 w-4 fill-red-500 text-red-500" />
            for the scientific community by{" "}
            <Link
              href={dystrHref}
              target="_blank"
              rel="noopener noreferrer"
              className="ml-1 font-medium underline underline-offset-4"
            >
              Dystr
            </Link>
          </p>
        </div>
      </div>
    </footer>
  );
}