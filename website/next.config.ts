import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow API routes to run on Vercel (no static export)
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'web.runmatstatic.com',
      },
    ],
  },
  experimental: {
    optimizePackageImports: ["lucide-react", "react-icons"],
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
      {
        source: '/terms',
        destination: '/docs/terms',
        permanent: true,
      },
      {
        source: '/docs/elements-of-matlab',
        destination: '/docs/matlab-function-reference',
        permanent: true,
      },

      // Privacy — no dedicated page yet; temporary redirect until one is created
      {
        source: '/privacy',
        destination: '/docs/terms',
        permanent: false,
      },

      // Raw .md file URLs -> correct doc routes
      {
        source: '/CLI.md',
        destination: '/docs/cli',
        permanent: true,
      },
      {
        source: '/LANGUAGE_COVERAGE.md',
        destination: '/docs/language-coverage',
        permanent: true,
      },
      {
        source: '/DESIGN_PHILOSOPHY.md',
        destination: '/docs/design-philosophy',
        permanent: true,
      },
      {
        source: '/INTRODUCTION_TO_RUNMAT_GPU.md',
        destination: '/docs/accelerate/fusion-intro',
        permanent: true,
      },
      {
        source: '/docs/CLI.md',
        destination: '/docs/cli',
        permanent: true,
      },
      {
        source: '/docs/LIBRARY.md',
        destination: '/docs/library',
        permanent: true,
      },
      {
        source: '/docs/LANGUAGE_COVERAGE.md',
        destination: '/docs/language-coverage',
        permanent: true,
      },
      {
        source: '/docs/INTRODUCTION_TO_RUNMAT_GPU.md',
        destination: '/docs/accelerate/fusion-intro',
        permanent: true,
      },
      {
        source: '/resources/DESIGN_PHILOSOPHY.md',
        destination: '/docs/design-philosophy',
        permanent: true,
      },

      // Old acceleration prefix -> correct routes
      {
        source: '/docs/acceleration/gpu/gpuArray',
        destination: '/docs/reference/builtins/gpuarray',
        permanent: true,
      },
      {
        source: '/docs/acceleration/gpu/gather',
        destination: '/docs/reference/builtins/gather',
        permanent: true,
      },

      // Wrong accelerate sub-paths
      {
        source: '/docs/accelerate/gpu-residency',
        destination: '/docs/accelerate/gpu-behavior',
        permanent: true,
      },
      {
        source: '/docs/accelerate/how-it-works',
        destination: '/docs/how-it-works',
        permanent: true,
      },

      // Old category-based reference paths -> flat builtins path
      {
        source: '/docs/reference/regex/:slug',
        destination: '/docs/reference/builtins/:slug',
        permanent: true,
      },
      {
        source: '/docs/reference/trigonometry/:slug',
        destination: '/docs/reference/builtins/:slug',
        permanent: true,
      },
      {
        source: '/docs/reference/introspection/:slug',
        destination: '/docs/reference/builtins/:slug',
        permanent: true,
      },
      {
        source: '/docs/reference/diagnostics/:slug',
        destination: '/docs/reference/builtins/:slug',
        permanent: true,
      },
      {
        source: '/docs/reference/core/:slug',
        destination: '/docs/reference/builtins/:slug',
        permanent: true,
      },
      {
        source: '/docs/search/:slug',
        destination: '/docs/reference/builtins/:slug',
        permanent: true,
      },

      // Exact-match redirects for multi-segment old paths
      {
        source: '/docs/io/filetext/fread',
        destination: '/docs/reference/builtins/fread',
        permanent: true,
      },
      {
        source: '/strings/core/strlength',
        destination: '/docs/reference/builtins/strlength',
        permanent: true,
      },
      {
        source: '/builtins/io/repl_fs/cd',
        destination: '/docs/reference/builtins/cd',
        permanent: true,
      },
    ];
  },
};

export default nextConfig;