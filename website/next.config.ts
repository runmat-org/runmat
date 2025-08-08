import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  // Enable static optimization
  experimental: {
    optimizePackageImports: ['lucide-react']
  }
};

export default nextConfig;