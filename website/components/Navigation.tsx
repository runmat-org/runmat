"use client";

import * as React from "react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  navigationMenuTriggerStyle,
} from "@/components/ui/navigation-menu";
import { Button } from "@/components/ui/button";
import { Download, Menu } from "lucide-react";
import { SiGithub } from "react-icons/si";
import { useState } from "react";
import { ThemeToggle } from "./ThemeToggle";

export default function Navigation() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-16 items-center justify-between px-4 md:px-6">
        <div className="mr-4 hidden md:flex">
          <Link href="/" className="mr-6 flex items-center space-x-2">
            <svg height="20" viewBox="0 0 264 134" fill="none" xmlns="http://www.w3.org/2000/svg" className="pr-2">
              <path d="M30 33H0V101H30V33Z" fill="url(#paint0_linear_0_1)" />
              <path d="M264 33H230V101H264V33Z" fill="url(#paint1_linear_0_1)" />
              <path d="M30 33V0H96V33H30Z" fill="url(#paint2_linear_0_1)" />
              <path d="M165 134V101H230V134H165Z" fill="url(#paint3_linear_0_1)" />
              <path d="M96 65V33H132V65H96Z" fill="url(#paint4_linear_0_1)" />
              <path d="M132 101V66H165V101H132Z" fill="url(#paint5_linear_0_1)" />
              <defs>
                <linearGradient id="paint0_linear_0_1" x1="15" y1="33" x2="15" y2="101" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#CA26D6" />
                  <stop offset="1" stopColor="#D500C6" />
                </linearGradient>
                <linearGradient id="paint1_linear_0_1" x1="247" y1="33" x2="247" y2="101" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#CA26D6" />
                  <stop offset="1" stopColor="#D500C6" />
                </linearGradient>
                <linearGradient id="paint2_linear_0_1" x1="63" y1="0" x2="63" y2="33" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#CA26D6" />
                  <stop offset="1" stopColor="#D500C6" />
                </linearGradient>
                <linearGradient id="paint3_linear_0_1" x1="197.5" y1="101" x2="197.5" y2="134" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#CA26D6" />
                  <stop offset="1" stopColor="#D500C6" />
                </linearGradient>
                <linearGradient id="paint4_linear_0_1" x1="114" y1="33" x2="114" y2="65" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#CA26D6" />
                  <stop offset="1" stopColor="#D500C6" />
                </linearGradient>
                <linearGradient id="paint5_linear_0_1" x1="148.5" y1="66" x2="148.5" y2="101" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#CA26D6" />
                  <stop offset="1" stopColor="#D500C6" />
                </linearGradient>
              </defs>
            </svg>
            <span className="hidden font-bold sm:inline-block">RunMat</span>
          </Link>
          <NavigationMenu>
            <NavigationMenuList>
              <NavigationMenuItem>
                <NavigationMenuLink className={navigationMenuTriggerStyle()} asChild>
                  <Link href="/">
                    Home
                  </Link>
                </NavigationMenuLink>
              </NavigationMenuItem>
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
                  <Link href="/license">
                    License
                  </Link>
                </NavigationMenuLink>
              </NavigationMenuItem>
            </NavigationMenuList>
          </NavigationMenu>
        </div>
        <Button
          variant="ghost"
          className="mr-2 px-0 text-base hover:bg-transparent focus-visible:bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 md:hidden"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        >
          <Menu className="h-6 w-6" />
          <span className="sr-only">Toggle Menu</span>
        </Button>
        <div className="flex flex-1 items-center justify-between space-x-2 md:justify-end">
          <div className="w-full flex-1 md:w-auto md:flex-none">
            <Link href="/" className="flex items-center space-x-2 md:hidden">
              <div className="h-6 w-6 rounded" style={{background: 'linear-gradient(135deg, #3ea7fd 0%, #bb51ff 100%)'}} />
              <span className="font-bold">RunMat</span>
            </Link>
          </div>
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
                href="/"
                className="flex w-full items-center rounded-md p-2 text-sm font-medium hover:bg-accent"
                onClick={() => setMobileMenuOpen(false)}
              >
                Home
              </Link>
              <Link
                href="/docs/getting-started"
                className="flex w-full items-center rounded-md p-2 text-sm font-medium hover:bg-accent"
                onClick={() => setMobileMenuOpen(false)}
              >
                Getting Started
              </Link>
              <Link
                href="/blog"
                className="flex w-full items-center rounded-md p-2 text-sm font-medium hover:bg-accent"
                onClick={() => setMobileMenuOpen(false)}
              >
                Blog
              </Link>
              <Link
                href="/license"
                className="flex w-full items-center rounded-md p-2 text-sm font-medium hover:bg-accent"
                onClick={() => setMobileMenuOpen(false)}
              >
                License
              </Link>
              <div className="flex items-center space-x-2 pt-2">
                <Button variant="outline" size="sm" asChild>
                  <Link
                    href="https://github.com/runmat-org/runmat"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                  <SiGithub className="mr-2 h-4 w-4" />
                    GitHub
                  </Link>
                </Button>
                <Button size="sm" asChild>
                  <Link href="/download">
                    <Download className="mr-2 h-4 w-4" />
                    Download
                  </Link>
                </Button>
              </div>
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