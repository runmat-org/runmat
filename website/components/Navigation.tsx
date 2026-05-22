"use client";

import * as React from "react";
import Link from "next/link";
import Image from "next/image";
import { cn } from "@/lib/utils";
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
  NavigationMenuContent,
  navigationMenuTriggerStyle,
} from "@/components/ui/navigation-menu";
import { Button } from "@/components/ui/button";
import { Download, Menu, BookOpen, FileText, Minus } from "lucide-react";
import { SiGithub } from "react-icons/si";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ThemeToggle } from "./ThemeToggle";
import { trackWebsiteEvent } from "@/components/GoogleAnalytics";

export default function Navigation() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [authenticated, setAuthenticated] = useState(false);
  const router = useRouter();
  const projectHref = "/p";
  const primaryCtaHref = authenticated ? "/p" : "/sandbox";
  const primaryCtaLabel = authenticated ? "Open RunMat" : "Try in Browser";
  const secondaryCtaLabel = authenticated ? "My Projects" : "Sign In";

  useEffect(() => {
    let cancelled = false;
    const loadAuthStatus = async () => {
      try {
        const response = await fetch("/api/auth-status", {
          cache: "no-store",
          credentials: "same-origin",
        });
        if (!response.ok) {
          return;
        }
        const result = (await response.json()) as { authenticated?: boolean };
        if (!cancelled) {
          setAuthenticated(result.authenticated === true);
        }
      } catch {
        // Keep logged-out copy when the lightweight status probe is unavailable.
      }
    };
    void loadAuthStatus();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleTryInBrowserClick = () => {
    trackWebsiteEvent("website.nav.cta_clicked", {
      category: "navigation",
      label: authenticated ? "open_runmat" : "try_in_browser",
    });
  };
  const handleDocsClick = () => {
    router.push("/docs");
  };
  const handleResourcesClick = () => {
    router.push("/resources");
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex items-center justify-between px-4 md:px-6">
        <div className="mr-4 hidden md:flex">
          <Link href="/" className="mr-6 flex items-center">
            <Image
              src="/runmat-logo.svg"
              alt="RunMat"
              width={136}
              height={24}
              className="h-5 w-auto"
            />
          </Link>
          <NavigationMenu viewport={false}>
            <NavigationMenuList>
              <NavigationMenuItem>
                <NavigationMenuTrigger
                  className={navigationMenuTriggerStyle()}
                  onClick={handleDocsClick}
                >
                  Docs
                </NavigationMenuTrigger>
                <NavigationMenuContent className="w-56 border-border bg-popover backdrop-blur-xl shadow-sm">
                  <ul className="grid gap-1 p-1">
                    <li>
                      <NavigationMenuLink asChild>
                        <Link
                          href="/docs/getting-started"
                          className="whitespace-nowrap px-3 py-2 text-sm text-popover-foreground/90 transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground"
                        >
                          Getting Started
                        </Link>
                      </NavigationMenuLink>
                    </li>
                    <li>
                      <NavigationMenuLink asChild>
                        <Link
                          href="/docs/matlab-function-reference"
                          className="whitespace-nowrap px-3 py-2 text-sm text-popover-foreground/90 transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground"
                        >
                          Function Reference
                        </Link>
                      </NavigationMenuLink>
                    </li>
                    <li>
                      <NavigationMenuLink asChild>
                        <Link
                          href="/docs/changelog"
                          className="whitespace-nowrap px-3 py-2 text-sm text-popover-foreground/90 transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground"
                        >
                          Changelog
                        </Link>
                      </NavigationMenuLink>
                    </li>
                    <li>
                      <NavigationMenuLink asChild>
                        <Link
                          href="/docs/cli"
                          className="whitespace-nowrap px-3 py-2 text-sm text-popover-foreground/90 transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground"
                        >
                          CLI Reference
                        </Link>
                      </NavigationMenuLink>
                    </li>
                    <li className="my-1 h-px bg-border" />
                    <li>
                      <NavigationMenuLink asChild>
                        <Link
                          href="/docs"
                          className="whitespace-nowrap px-3 py-2 text-sm text-popover-foreground/90 transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground"
                        >
                          View all docs
                        </Link>
                      </NavigationMenuLink>
                    </li>
                  </ul>
                </NavigationMenuContent>
              </NavigationMenuItem>
              <NavigationMenuItem>
                <NavigationMenuTrigger
                  className={navigationMenuTriggerStyle()}
                  onClick={handleResourcesClick}
                >
                  Resources
                </NavigationMenuTrigger>
                <NavigationMenuContent className="w-56 border-border bg-popover backdrop-blur-xl shadow-sm">
                  <ul className="grid gap-1 p-1">
                    <li>
                      <NavigationMenuLink asChild>
                        <Link
                          href="/blog"
                          className="whitespace-nowrap px-3 py-2 text-sm text-popover-foreground/90 transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground"
                        >
                          Blog
                        </Link>
                      </NavigationMenuLink>
                    </li>
                    <li>
                      <NavigationMenuLink asChild>
                        <Link
                          href="/resources/guides"
                          className="whitespace-nowrap px-3 py-2 text-sm text-popover-foreground/90 transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground"
                        >
                          Guides
                        </Link>
                      </NavigationMenuLink>
                    </li>
                    <li>
                      <NavigationMenuLink asChild>
                        <Link
                          href="/benchmarks"
                          className="whitespace-nowrap px-3 py-2 text-sm text-popover-foreground/90 transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground"
                        >
                          Benchmarks
                        </Link>
                      </NavigationMenuLink>
                    </li>
                    <li>
                      <NavigationMenuLink asChild>
                        <Link
                          href="/about"
                          className="whitespace-nowrap px-3 py-2 text-sm text-popover-foreground/90 transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground"
                        >
                          About RunMat
                        </Link>
                      </NavigationMenuLink>
                    </li>
                    <li className="my-1 h-px bg-border" />
                    <li>
                      <NavigationMenuLink asChild>
                        <Link
                          href="/resources"
                          className="whitespace-nowrap px-3 py-2 text-sm text-popover-foreground/90 transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground"
                        >
                          Resource Hub
                        </Link>
                      </NavigationMenuLink>
                    </li>
                  </ul>
                </NavigationMenuContent>
              </NavigationMenuItem>
              <NavigationMenuItem>
                <NavigationMenuLink asChild>
                  <Link href="/pricing" className={navigationMenuTriggerStyle()}>
                    Pricing
                  </Link>
                </NavigationMenuLink>
              </NavigationMenuItem>
            </NavigationMenuList>
          </NavigationMenu>
        </div>
        <div className="flex flex-1 items-center justify-between space-x-2 md:justify-end">
          <div className="w-full flex-1 md:w-auto md:flex-none">
            <Link href="/" className="flex items-center md:hidden">
              <Image
                src="/runmat-logo.svg"
                alt="RunMat"
                width={120}
                height={20}
                className="h-4 w-auto"
              />
            </Link>
          </div>

          {/* Mobile quick actions + burger menu */}
          <div className="flex items-center space-x-1 md:hidden">
            <Button variant="ghost" size="lg" className="h-10 w-10 p-0" asChild>
              <Link
                href="https://github.com/runmat-org/runmat"
                target="_blank"
                rel="noopener noreferrer"
              >
                <SiGithub className="h-5 w-5" />
                <span className="sr-only">GitHub</span>
              </Link>
            </Button>
            <div className="[&>button]:h-10 [&>button]:w-10 [&>button]:p-0 [&_svg]:h-5 [&_svg]:w-5">
              <ThemeToggle />
            </div>
            <Button
              variant="ghost"
              size="lg"
              className="h-10 w-10 p-0 hover:bg-transparent focus-visible:bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              <Menu className="h-5 w-5" />
              <span className="sr-only">Toggle Menu</span>
            </Button>
          </div>

          {/* Desktop navigation */}
          <nav className="hidden md:flex items-center">
            <Link
              href="https://github.com/runmat-org/runmat"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center h-11 px-3 text-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
            >
              <SiGithub className="h-4 w-4" />
              <span className="sr-only">GitHub</span>
            </Link>
            <ThemeToggle />
            <Link
              href="/download"
              className="inline-flex items-center gap-1.5 h-11 px-4 text-sm text-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
            >
              <Download className="h-4 w-4" />
              Download
            </Link>
            <Link
              href={projectHref}
              className="inline-flex items-center justify-center h-11 px-5 text-sm font-medium whitespace-nowrap border-x border-border text-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
            >
              {secondaryCtaLabel}
            </Link>
            <Link
              href={primaryCtaHref}
              className="inline-flex items-center justify-center h-11 px-6 text-sm font-semibold whitespace-nowrap bg-[hsl(var(--brand))] hover:bg-[hsl(var(--brand))]/90 text-white transition-colors"
              onClick={handleTryInBrowserClick}
              target="_blank"
              rel="noopener noreferrer"
              data-ph-capture-attribute-destination={authenticated ? "projects" : "sandbox"}
              data-ph-capture-attribute-source="nav-desktop"
              data-ph-capture-attribute-cta={authenticated ? "open-runmat" : "try-in-browser"}
            >
              {primaryCtaLabel}
            </Link>
          </nav>
        </div>
      </div>
      {mobileMenuOpen && (
        <div className="container md:hidden">
          <div className="border-b border-border py-4">
            <div className="grid gap-2">
              <Link
                href="/docs"
                className="flex w-full items-center p-2 text-sm font-medium hover:bg-accent"
                onClick={() => setMobileMenuOpen(false)}
              >
                <BookOpen className="mr-2 h-4 w-4" />
                Docs
              </Link>
              <Link
                href="/docs/getting-started"
                className="flex w-full items-center p-2 text-sm font-medium hover:bg-accent pl-6"
                onClick={() => setMobileMenuOpen(false)}
              >
                <Minus className="mr-2 h-3 w-3 text-muted-foreground" />
                Getting Started
              </Link>
              <Link
                href="/docs/matlab-function-reference"
                className="flex w-full items-center p-2 text-sm font-medium hover:bg-accent pl-6"
                onClick={() => setMobileMenuOpen(false)}
              >
                <Minus className="mr-2 h-3 w-3 text-muted-foreground" />
                Function Reference
              </Link>
              <Link
                href="/docs/changelog"
                className="flex w-full items-center p-2 text-sm font-medium hover:bg-accent pl-6"
                onClick={() => setMobileMenuOpen(false)}
              >
                <Minus className="mr-2 h-3 w-3 text-muted-foreground" />
                Changelog
              </Link>
              <Link
                href="/docs/cli"
                className="flex w-full items-center p-2 text-sm font-medium hover:bg-accent pl-6"
                onClick={() => setMobileMenuOpen(false)}
              >
                <Minus className="mr-2 h-3 w-3 text-muted-foreground" />
                CLI Reference
              </Link>
              <Link
                href="/resources"
                className="flex w-full items-center p-2 text-sm font-medium hover:bg-accent mt-2"
                onClick={() => setMobileMenuOpen(false)}
              >
                <FileText className="mr-2 h-4 w-4" />
                Resources
              </Link>
              <Link
                href="/blog"
                className="flex w-full items-center p-2 text-sm font-medium hover:bg-accent pl-6"
                onClick={() => setMobileMenuOpen(false)}
              >
                <Minus className="mr-2 h-3 w-3 text-muted-foreground" />
                Blog
              </Link>
              <Link
                href="/resources/guides"
                className="flex w-full items-center p-2 text-sm font-medium hover:bg-accent pl-6"
                onClick={() => setMobileMenuOpen(false)}
              >
                <Minus className="mr-2 h-3 w-3 text-muted-foreground" />
                Guides
              </Link>
              <Link
                href="/benchmarks"
                className="flex w-full items-center p-2 text-sm font-medium hover:bg-accent pl-6"
                onClick={() => setMobileMenuOpen(false)}
              >
                <Minus className="mr-2 h-3 w-3 text-muted-foreground" />
                Benchmarks
              </Link>
              <Link
                href="/about"
                className="flex w-full items-center p-2 text-sm font-medium hover:bg-accent pl-6"
                onClick={() => setMobileMenuOpen(false)}
              >
                <Minus className="mr-2 h-3 w-3 text-muted-foreground" />
                About RunMat
              </Link>
              <Link
                href="/pricing"
                className="flex w-full items-center p-2 text-sm font-medium hover:bg-accent mt-2"
                onClick={() => setMobileMenuOpen(false)}
              >
                Pricing
              </Link>
              <Link
                href="/download"
                className="flex w-full items-center p-2 text-sm font-medium hover:bg-accent"
                onClick={() => setMobileMenuOpen(false)}
              >
                <Download className="mr-2 h-4 w-4" />
                Download
              </Link>
              <Link
                href={primaryCtaHref}
                className="flex w-full items-center justify-center mt-2 p-3 text-sm font-semibold bg-[hsl(var(--brand))] text-white transition-colors shadow-none hover:bg-[hsl(var(--brand))]/90"
                onClick={() => {
                  setMobileMenuOpen(false);
                  handleTryInBrowserClick();
                }}
                target="_blank"
                rel="noopener noreferrer"
              >
                {primaryCtaLabel}
              </Link>
              <Link
                href={projectHref}
                className="flex w-full items-center justify-center p-2 text-sm font-medium hover:bg-accent"
                onClick={() => setMobileMenuOpen(false)}
              >
                {secondaryCtaLabel}
              </Link>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}

const ListItem = React.forwardRef<
  React.ElementRef<"a">,
  React.ComponentPropsWithoutRef<"a">
>(({ className, title, children, ...props }, ref) => {
  return (
    <li>
      <NavigationMenuLink asChild>
        <a
          ref={ref}
          className={cn(
            "block select-none space-y-1 p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground",
            className
          )}
          {...props}
        >
          <div className="text-sm font-medium leading-none">{title}</div>
          <p className="line-clamp-2 text-sm leading-snug text-muted-foreground">
            {children}
          </p>
        </a>
      </NavigationMenuLink>
    </li>
  );
});
ListItem.displayName = "ListItem";
