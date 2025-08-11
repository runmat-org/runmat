import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow API routes to run on Vercel (no static export)
  images: { unoptimized: true },
  experimental: {
    optimizePackageImports: ["lucide-react"],
  },
};

export default nextConfig;