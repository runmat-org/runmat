"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Heart } from "lucide-react";
import { SiGithub } from "react-icons/si";
import Logo from "@/components/Logo";
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
    <footer className="border-t bg-background">
      <div className="container relative mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-8 py-8 px-4 md:px-6">
        <div className="flex w-full flex-col items-start gap-3 md:flex-[2]">
          <div className="flex items-center gap-3">
            <Logo height={24} className="shrink-0" />
            <p className="text-center text-base leading-none md:text-left md:text-lg">
              <span className="font-semibold brand-text-sheen">RunMat</span>
            </p>
          </div>
          <p className="text-left text-sm leading-loose text-muted-foreground">
            A modern, high-performance runtime for MATLAB and GNU Octave code.
          </p>
          <NewsletterCta
            title="Subscribe to our newsletter"
            description="Get updates on releases, benchmarks, and deep dives."
            align="left"
            className="w-full"
          />
        </div>
        <div className="flex items-center gap-4 md:flex-[0_0_auto] absolute right-4 top-4 md:static">
          <Link
            href="https://github.com/runmat-org/runmat"
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground hover:text-foreground flex items-center"
          >
            <SiGithub className="h-5 w-5" aria-label="GitHub" />
            <span className="sr-only">GitHub</span>
          </Link>
        </div>
      </div>
      <div className="border-t">
        <div className="container flex flex-col items-center justify-between gap-4 py-6 md:flex-row">
          <div className="text-sm text-muted-foreground md:text-left">
            <p>
            © 2025 Dystr Inc. All rights reserved. MIT+ Licensed.
            </p>
            <p>
              MATLAB is a registered trademark of The MathWorks, Inc.
            </p>
            <p>
              GNU Octave is a registered trademark of the Free Software Foundation.
            </p>
            <p>
              RunMat is not affiliated with, endorsed by, or sponsored by The MathWorks, Inc. or the Free Software Foundation.
            </p>
          </div>
          <p className="flex items-center text-center text-sm text-muted-foreground md:text-left">
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