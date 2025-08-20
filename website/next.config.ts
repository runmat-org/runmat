import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow API routes to run on Vercel (no static export)
  images: { unoptimized: true },
  experimental: {
    optimizePackageImports: ["lucide-react"],
  },
  // Ensure Markdown files living outside the website/ dir are traced for SSG/SSR on Vercel
  outputFileTracingIncludes: {
    "/app/**": [
      "../docs/**/*.md",
      "../crates/**/*.md",
    ],
  },
};

export default nextConfig;