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
import { Download, Menu, BookOpen, FileText, Scale, Minus } from "lucide-react";
import { SiGithub } from "react-icons/si";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { ThemeToggle } from "./ThemeToggle";
import { trackEvent } from "@/components/GoogleAnalytics";

export default function Navigation() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const router = useRouter();
  const handleTryInBrowserClick = () => {
    trackEvent("nav_cta_click", "navigation", "try_in_browser");
  };
  const handleDocsClick = () => {
    router.push("/docs");
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-16 items-center justify-between px-4 md:px-6">
        <div className="mr-4 hidden md:flex">
          <Link href="/" className="mr-6 flex items-center">
            <Image
              src="/runmat-logo.svg"
              alt="RunMat"
              width={136}
              height={24}
              className="h-6 w-auto"
              priority
            />
          </Link>
          <NavigationMenu>
            <NavigationMenuList>
              <NavigationMenuItem>
                <NavigationMenuTrigger
                  className={navigationMenuTriggerStyle()}
                  onClick={handleDocsClick}
                >
                  Docs
                </NavigationMenuTrigger>
                <NavigationMenuContent className="w-56 rounded-lg border border-gray-700/60 bg-black/95 backdrop-blur-xl shadow-lg">
                  <ul className="grid gap-1 p-1">
                    <li>
                      <NavigationMenuLink asChild>
                        <Link
                          href="/docs/getting-started"
                          className="whitespace-nowrap rounded-md px-3 py-2 text-sm text-white/90 transition-colors hover:bg-white/5 hover:text-white focus:bg-white/5 focus:text-white"
                        >
                          Getting Started
                        </Link>
                      </NavigationMenuLink>
                    </li>
                    <li>
                      <NavigationMenuLink asChild>
                        <Link
                          href="/docs/matlab-function-reference"
                          className="whitespace-nowrap rounded-md px-3 py-2 text-sm text-white/90 transition-colors hover:bg-white/5 hover:text-white focus:bg-white/5 focus:text-white"
                        >
                          Function Reference
                        </Link>
                      </NavigationMenuLink>
                    </li>
                    <li>
                      <NavigationMenuLink asChild>
                        <Link
                          href="/docs/architecture"
                          className="whitespace-nowrap rounded-md px-3 py-2 text-sm text-white/90 transition-colors hover:bg-white/5 hover:text-white focus:bg-white/5 focus:text-white"
                        >
                          Architecture
                        </Link>
                      </NavigationMenuLink>
                    </li>
                    <li>
                      <NavigationMenuLink asChild>
                        <Link
                          href="/docs/cli"
                          className="whitespace-nowrap rounded-md px-3 py-2 text-sm text-white/90 transition-colors hover:bg-white/5 hover:text-white focus:bg-white/5 focus:text-white"
                        >
                          CLI Reference
                        </Link>
                      </NavigationMenuLink>
                    </li>
                    <li className="my-1 h-px bg-gray-700/50" />
                    <li>
                      <NavigationMenuLink asChild>
                        <Link
                          href="/docs"
                          className="whitespace-nowrap rounded-md px-3 py-2 text-sm text-white/90 transition-colors hover:bg-white/5 hover:text-white focus:bg-white/5 focus:text-white"
                        >
                          View all docs
                        </Link>
                      </NavigationMenuLink>
                    </li>
                  </ul>
                </NavigationMenuContent>
              </NavigationMenuItem>
              <NavigationMenuItem>
                <NavigationMenuLink className={navigationMenuTriggerStyle()} asChild>
                  <Link href="/blog">
                    Blog
                  </Link>
                </NavigationMenuLink>
              </NavigationMenuItem>
              <NavigationMenuItem>
                <NavigationMenuLink className={navigationMenuTriggerStyle()} asChild>
                  <Link href="/benchmarks">
                    Benchmarks
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
                className="h-5 w-auto"
                priority
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
          <nav className="hidden md:flex items-center space-x-2">
            <Button variant="ghost" size="sm" asChild>
              <Link
                href="https://github.com/runmat-org/runmat"
                target="_blank"
                rel="noopener noreferrer"
              >
                <SiGithub className="h-4 w-4" />
                <span className="sr-only">GitHub</span>
              </Link>
            </Button>
            <ThemeToggle />
            <Button size="sm" asChild>
              <Link href="/download">
                <Download className="mr-2 h-4 w-4" />
                Download
              </Link>
            </Button>
            <Link
              href="https://runmat.com/sandbox"
              className="inline-flex items-center justify-center h-10 px-6 text-sm font-semibold flex-shrink-0 whitespace-nowrap rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow hover:from-blue-600 hover:to-purple-700 transition-colors"
              onClick={handleTryInBrowserClick}
              target="_blank"
              rel="noopener noreferrer"
              data-ph-capture-attribute-destination="sandbox"
              data-ph-capture-attribute-source="nav-desktop"
              data-ph-capture-attribute-cta="try-in-browser"
            >
              Try in Browser
            </Link>
          </nav>
        </div>
      </div>
      {mobileMenuOpen && (
        <div className="container md:hidden">
          <div className="border-b border-border py-4">
            <div className="grid gap-2">
              <Link
                href="/blog"
                className="flex w-full items-center rounded-md p-2 text-sm font-medium hover:bg-accent"
                onClick={() => setMobileMenuOpen(false)}
              >
                <FileText className="mr-2 h-4 w-4" />
                Blog
              </Link>
              <Link
                href="/docs"
                className="flex w-full items-center rounded-md p-2 text-sm font-medium hover:bg-accent"
                onClick={() => setMobileMenuOpen(false)}
              >
                <BookOpen className="mr-2 h-4 w-4" />
                Documentation
              </Link>
              <Link
                href="/docs/getting-started"
                className="flex w-full items-center rounded-md p-2 text-sm font-medium hover:bg-accent pl-6"
                onClick={() => setMobileMenuOpen(false)}
              >
                <Minus className="mr-2 h-3 w-3 text-muted-foreground" />
                Getting Started
              </Link>
              <Link
                href="/docs/architecture"
                className="flex w-full items-center rounded-md p-2 text-sm font-medium hover:bg-accent pl-6"
                onClick={() => setMobileMenuOpen(false)}
              >
                <Minus className="mr-2 h-3 w-3 text-muted-foreground" />
                Architecture
              </Link>
              <Link
                href="/docs/cli"
                className="flex w-full items-center rounded-md p-2 text-sm font-medium hover:bg-accent pl-6"
                onClick={() => setMobileMenuOpen(false)}
              >
                <Minus className="mr-2 h-3 w-3 text-muted-foreground" />
                CLI Reference
              </Link>
              <Link
                href="/docs/configuration"
                className="flex w-full items-center rounded-md p-2 text-sm font-medium hover:bg-accent pl-6"
                onClick={() => setMobileMenuOpen(false)}
              >
                <Minus className="mr-2 h-3 w-3 text-muted-foreground" />
                Configuration
              </Link>
              <Link
                href="/benchmarks"
                className="flex w-full items-center rounded-md p-2 text-sm font-medium hover:bg-accent"
                onClick={() => setMobileMenuOpen(false)}
              >
                <Scale className="mr-2 h-4 w-4" />
                Benchmarks
              </Link>
              <Link
                href="/download"
                className="flex w-full items-center rounded-md p-2 text-sm font-medium hover:bg-accent mt-2"
                onClick={() => setMobileMenuOpen(false)}
              >
                <Download className="mr-2 h-4 w-4" />
                Download
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
            "block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground",
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
