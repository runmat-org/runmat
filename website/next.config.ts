import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow API routes to run on Vercel (no static export)
  images: { 
    unoptimized: true,
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'web.runmatstatic.com',
      },
    ],
  },
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
  async redirects() {
    return [
      {
        source: '/docs/CLI.md',
        destination: '/docs/cli',
        permanent: true,
      },
      {
        source: '/docs/LANGUAGE_COVERAGE.md',
        destination: '/docs/language-coverage',
        permanent: true,
      },
      {
        source: '/docs/LIBRARY.md',
        destination: '/docs/library',
        permanent: true,
      },
      {
        source: '/docs/INTRODUCTION_TO_RUNMAT.PDF',
        destination: '/docs/accelerate/fusion-intro',
        permanent: true,
      },
      {
        source: '/docs/accelerate/gpu-residency',
        destination: '/docs/accelerate/fusion-intro',
        permanent: true,
      },
      {
        source: '/docs/reference/builtins/containers.Map.isKey',
        destination: '/docs/reference/builtins/containers.map',
        permanent: true,
      },
      {
        source: '/blog/matlab-alternatives-runmat-vs-octave-julia-python',
        destination: '/blog/free-matlab-alternatives',
        permanent: true,
      },
      {
        source: '/blog/matlab-alternatives',
        destination: '/blog/free-matlab-alternatives',
        permanent: true,
      },
      {
        source: '/blog/runmat-accel-intro-blog',
        destination: '/blog/runmat-accelerate-fastest-runtime-for-your-math',
        permanent: true,
      },
      {
        source: '/blog/why-rust',
        destination: '/blog/rust-llm-training-distribution',
        permanent: true,
      },
    ];
  },
};

export default nextConfig;