"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Heart, MapPin } from "lucide-react";
import { SiGithub, SiLinkedin, SiX } from "react-icons/si";
import Image from "next/image";
import NewsletterCta from "@/components/NewsletterCta";

export default function Footer() {
  // Default to plain Dystr URL during SSR; hydrate with UTM params on client
  const [dystrHref, setDystrHref] = useState("https://dystr.com");
  const currentYear = new Date().getFullYear();
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
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-3">
            <Link href="/" className="focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded">
              <Image
                src="/runmat-logo.svg"
                alt="RunMat"
                width={136}
                height={24}
                className="h-6 w-auto"
                priority
              />
            </Link>
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            The Fastest Runtime for Math
          </p>
          <div className="flex items-center gap-3 mt-1">
            <Link href="https://github.com/runmat-org/runmat" target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-foreground transition-colors">
              <SiGithub className="h-5 w-5" />
              <span className="sr-only">GitHub</span>
            </Link>
            <Link href="https://x.com/runmat_com" target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-foreground transition-colors">
              <SiX className="h-5 w-5" />
              <span className="sr-only">X (Twitter)</span>
            </Link>
            <Link href="https://www.linkedin.com/company/runmat" target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-foreground transition-colors">
              <SiLinkedin className="h-5 w-5" />
              <span className="sr-only">LinkedIn</span>
            </Link>
          </div>
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <MapPin className="h-3.5 w-3.5" />
            <span>San Francisco, CA · Seattle, WA · New York, NY</span>
          </div>
        </div>
        <div className="grid w-full md:flex-1 gap-y-6 md:gap-x-8 md:grid-cols-[minmax(6rem,1fr)_minmax(6rem,1fr)] text-sm text-muted-foreground">
          <div>
            <h3 className="text-foreground font-semibold mb-2">Resources</h3>
            <ul className="space-y-1">
              <li><Link href="/about" className="hover:underline">About</Link></li>
              <li><Link href="/pricing" className="hover:underline">Pricing</Link></li>
              <li><Link href="/license" className="hover:underline">License</Link></li>
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
        <div className="w-full md:w-auto md:flex-1 md:max-w-md" id="newsletter">
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
            © {currentYear} Dystr Inc. All rights reserved. MIT+ Licensed.
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