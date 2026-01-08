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
  navigationMenuTriggerStyle,
} from "@/components/ui/navigation-menu";
import { Button } from "@/components/ui/button";
import { Download, Menu, BookOpen, FileText, Scale, Minus } from "lucide-react";
import { SiGithub } from "react-icons/si";
import { useState } from "react";
import { ThemeToggle } from "./ThemeToggle";

export default function Navigation() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-16 items-center justify-between px-4 md:px-6">
        <div className="mr-4 hidden md:flex">
          <Link href="/" className="mr-6 flex items-center">
            <Image
              src="/runmat-logo.svg"
              alt="RunMat"
              width={160}
              height={28}
              className="h-7 w-auto"
              priority
            />
          </Link>
          <NavigationMenu>
            <NavigationMenuList>
              <NavigationMenuItem>
                <NavigationMenuLink className={navigationMenuTriggerStyle()} asChild>
                  <Link href="/docs">
                    Documentation
                  </Link>
                </NavigationMenuLink>
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
                width={140}
                height={24}
                className="h-6 w-auto"
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