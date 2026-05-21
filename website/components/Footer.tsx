"use client";

import Link from "next/link";
import { useMemo } from "react";
import { Heart } from "lucide-react";
import { SiGithub, SiLinkedin, SiX } from "react-icons/si";
import Image from "next/image";
import NewsletterCta from "@/components/NewsletterCta";

export default function Footer() {
  const currentYear = useMemo(() => new Date().getFullYear(), []);

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
              />
            </Link>
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Run math blazing fast
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
        </div>
        <div className="grid w-full md:flex-1 gap-y-6 md:gap-x-8 md:grid-cols-[minmax(6rem,1fr)_minmax(6rem,1fr)] text-sm text-muted-foreground">
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-foreground mb-3">Company</h3>
            <ul className="space-y-1.5">
              <li><Link href="/about" className="hover:text-foreground transition-colors">About</Link></li>
              <li><Link href="/pricing" className="hover:text-foreground transition-colors">Pricing</Link></li>
              <li><Link href="/contact" className="hover:text-foreground transition-colors">Contact</Link></li>
              <li><Link href="/license" className="hover:text-foreground transition-colors">License</Link></li>
            </ul>
          </div>
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-foreground mb-3">Learn</h3>
            <ul className="space-y-1.5">
              <li><Link href="/docs" className="hover:text-foreground transition-colors">Docs</Link></li>
              <li><Link href="/blog" className="hover:text-foreground transition-colors">Blog</Link></li>
              <li><Link href="/benchmarks" className="hover:text-foreground transition-colors">Benchmarks</Link></li>
              <li><Link href="/matlab-online" className="hover:text-foreground transition-colors">RunMat vs MATLAB Online</Link></li>
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
          <div className="text-sm text-center md:text-left text-muted-foreground">
            <p className="flex items-center">
              © {currentYear} Dystr
            {" · "}
              Made with
              <Heart className="mx-1 h-4 w-4 fill-destructive text-destructive" />
              for the scientific community.
            </p>
            <p>
              MATLAB is a registered trademark of The MathWorks, Inc. RunMat is not affiliated with The MathWorks, Inc.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}