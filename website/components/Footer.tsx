import Link from "next/link";
import { Heart } from "lucide-react";
import { SiGithub } from "react-icons/si";
import Logo from "@/components/Logo";

export default function Footer() {
  return (
    <footer className="border-t bg-background">
      <div className="container mx-auto flex flex-col items-center justify-between gap-4 py-10 md:h-24 md:flex-row md:py-0 px-4 md:px-6">
        <div className="flex flex-col items-center gap-4 px-8 md:flex-row md:gap-2 md:px-0">
          <div className="flex items-center gap-3">
            <Logo height={24} className="shrink-0" />
            <p className="text-center text-base leading-none md:text-left md:text-lg">
              <span className="font-semibold brand-text-sheen">RunMat</span>
            </p>
          </div>
          <p className="text-center text-sm leading-loose text-muted-foreground md:text-left">
            A modern, high-performance runtime for MATLAB and GNU Octave code.
          </p>
        </div>
        <div className="flex items-center gap-4">
          <Link
            href="https://github.com/runmat-org/runmat"
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground hover:text-foreground"
          >
            <SiGithub className="h-5 w-5" />
            <span className="sr-only">GitHub</span>
          </Link>
        </div>
      </div>
      <div className="border-t">
        <div className="container flex flex-col items-center justify-between gap-4 py-6 md:flex-row">
          <p className="text-center text-sm text-muted-foreground md:text-left">
            © 2025 Dystr Inc. All rights reserved. MIT+ Licensed.
          </p>
          <p className="flex items-center text-center text-sm text-muted-foreground md:text-left">
            Made with
            <Heart className="mx-1 h-4 w-4 fill-red-500 text-red-500" />
            for the scientific community by{" "}
            <Link
              href="https://dystr.com"
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