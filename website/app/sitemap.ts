import { MetadataRoute } from 'next'
import { getPublicBlogPosts } from '@/lib/blog'
import { getAllBenchmarks } from '@/lib/benchmarks'
import { loadBuiltins } from '@/lib/builtins'
import { flatten } from '@/content/docs'

export const dynamic = 'force-static'

function toDate(value: string | undefined, fallback: Date): Date {
  if (!value) return fallback
  const d = new Date(value)
  return Number.isNaN(d.getTime()) ? fallback : d
}

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = 'https://runmat.com'
  const currentDate = new Date()

  const videoPosters = 'https://web.runmatstatic.com/video/posters'
  const videoBase = 'https://web.runmatstatic.com/video'

  const staticRoutes: MetadataRoute.Sitemap = [
    {
      url: baseUrl,
      lastModified: currentDate,
      changeFrequency: 'daily',
      priority: 1,
      videos: [
        {
          title: 'RunMat wave interference simulation',
          thumbnail_loc: `${videoPosters}/runmat-wave-simulation.png`,
          content_loc: `${videoBase}/runmat-wave-simulation.mp4`,
          description: 'GPU-accelerated wave interference simulation rendered in real time using RunMat surf().',
        },
        {
          title: 'RunMat 3D interactive plotting',
          thumbnail_loc: `${videoPosters}/3d-interactive-plotting-runmat.png`,
          content_loc: `${videoBase}/3d-interactive-plotting-runmat.mp4`,
          description: 'GPU-accelerated 3D surface plots running in the browser with no install or license required.',
        },
        {
          title: 'RunMat shape tracking and type system',
          thumbnail_loc: `${videoPosters}/runmat-shape-tracking.png`,
          content_loc: `${videoBase}/runmat-shape-tracking.mp4`,
          description: 'Variable explorer showing real-time shape, type, and value inspection while running MATLAB code.',
        },
        {
          title: 'RunMat automatic versioning',
          thumbnail_loc: `${videoPosters}/runmat-versioning.png`,
          content_loc: `${videoBase}/runmat-versioning.mp4`,
          description: 'Automatic file history and project snapshots with no git setup required.',
        },
      ],
    },
    {
      url: `${baseUrl}/download`,
      lastModified: currentDate,
      changeFrequency: 'daily',
      priority: 0.9,
    },
    {
      url: `${baseUrl}/matlab-online`,
      lastModified: currentDate,
      changeFrequency: 'weekly',
      priority: 0.9,
    },
    {
      url: `${baseUrl}/docs`,
      lastModified: currentDate,
      changeFrequency: 'daily',
      priority: 0.8,
    },
    {
      url: `${baseUrl}/blog`,
      lastModified: currentDate,
      changeFrequency: 'daily',
      priority: 0.7,
    },
    {
      url: `${baseUrl}/benchmarks`,
      lastModified: currentDate,
      changeFrequency: 'daily',
      priority: 0.5,
    },
    {
      url: `${baseUrl}/pricing`,
      lastModified: currentDate,
      changeFrequency: 'daily',
      priority: 0.8,
    },
    {
      url: `${baseUrl}/about`,
      lastModified: currentDate,
      changeFrequency: 'monthly',
      priority: 0.6,
    },
    {
      url: `${baseUrl}/contact`,
      lastModified: currentDate,
      changeFrequency: 'yearly',
      priority: 0.5,
    },
    {
      url: `${baseUrl}/license`,
      lastModified: currentDate,
      changeFrequency: 'yearly',
      priority: 0.3,
    },
    {
      url: `${baseUrl}/resources`,
      lastModified: currentDate,
      changeFrequency: 'weekly',
      priority: 0.6,
    },
    {
      url: `${baseUrl}/resources/guides`,
      lastModified: currentDate,
      changeFrequency: 'weekly',
      priority: 0.5,
    },
    {
      url: `${baseUrl}/resources/guides/what-is-matlab`,
      lastModified: currentDate,
      changeFrequency: 'monthly',
      priority: 0.6,
    },
    {
      url: `${baseUrl}/docs/matlab-function-reference`,
      lastModified: currentDate,
      changeFrequency: 'daily',
      priority: 0.8,
    },
    {
      url: `${baseUrl}/sandbox`,
      lastModified: currentDate,
      changeFrequency: 'daily',
      priority: 0.8,
    },
  ]

  const docRoutes: MetadataRoute.Sitemap = flatten()
    .map(node => {
      if (node.externalHref && node.externalHref.startsWith('/')) {
        return `${baseUrl}${node.externalHref}`
      }
      if (node.slug && node.slug.length > 0) {
        return `${baseUrl}/docs/${node.slug.join('/')}`
      }
      return null
    })
    .filter((url): url is string => Boolean(url))
    .map(url => ({
      url,
      lastModified: currentDate,
      changeFrequency: 'weekly',
      priority: 0.6,
    }))

  const blogVideoMap: Record<string, MetadataRoute.Sitemap[number]['videos']> = {
    'matlab-fprintf': [
      {
        title: 'RunMat variable explorer: debugging without fprintf',
        thumbnail_loc: `${videoPosters}/runmat-debugging.png`,
        content_loc: `${videoBase}/runmat-debugging.mp4`,
        description: 'Demo of RunMat variable explorer inspecting workspace state during a thermal simulation.',
      },
    ],
    'free-matlab-alternatives': [
      {
        title: 'RunMat 3D interactive plotting in the browser',
        thumbnail_loc: `${videoPosters}/3d-interactive-plotting-runmat.png`,
        content_loc: `${videoBase}/3d-interactive-plotting-runmat.mp4`,
        description: 'GPU-accelerated 3D surface plots running in the browser with no install or license required.',
      },
    ],
  }

  const blogPostRoutes: MetadataRoute.Sitemap = getPublicBlogPosts().map(post => ({
    url: `${baseUrl}/blog/${post.slug}`,
    lastModified: toDate(post.dateModified || post.date, currentDate),
    changeFrequency: 'monthly',
    priority: 0.6,
    ...(blogVideoMap[post.slug] ? { videos: blogVideoMap[post.slug] } : {}),
  }))

  const benchmarkRoutes: MetadataRoute.Sitemap = getAllBenchmarks().map(benchmark => {
    return {
      url: `${baseUrl}/benchmarks/${benchmark.slug}`,
      lastModified: toDate(benchmark.date, currentDate),
      changeFrequency: 'monthly',
      priority: 0.5,
    }
  })

  const builtinRoutes: MetadataRoute.Sitemap = loadBuiltins().map(builtin => ({
    url: `${baseUrl}/docs/reference/builtins/${builtin.slug}`,
    lastModified: currentDate,
    changeFrequency: 'weekly',
    priority: 0.5,
  }))

  const seen = new Set<string>()
  const merged: MetadataRoute.Sitemap = []
  for (const route of [...staticRoutes, ...blogPostRoutes, ...benchmarkRoutes, ...builtinRoutes, ...docRoutes]) {
    if (seen.has(route.url)) continue
    seen.add(route.url)
    merged.push(route)
  }

  return merged
}
