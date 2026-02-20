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
        source: '/run-matlab-online',
        destination: '/matlab-online',
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
      {
        source: '/blog/matlab-nvidia-gpu',
        destination: '/blog/how-to-use-gpu-in-matlab',
        permanent: true,
      },
    ];
  },
};

export default nextConfig;