import { MetadataRoute } from 'next'
import { getAllBlogPosts } from '@/app/blog/page'
import { loadBuiltins } from '@/lib/builtins'
import { flatten } from '@/content/docs'

export const dynamic = 'force-static'

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = 'https://runmat.org'
  const currentDate = new Date()

  const staticRoutes: MetadataRoute.Sitemap = [
    {
      url: baseUrl,
      lastModified: currentDate,
      changeFrequency: 'daily',
      priority: 1,
    },
    {
      url: `${baseUrl}/download`,
      lastModified: currentDate,
      changeFrequency: 'daily',
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
      priority: 0.5,
    },
    {
      url: `${baseUrl}/license`,
      lastModified: currentDate,
      changeFrequency: 'daily',
      priority: 0.3,
    },
    {
      url: `${baseUrl}/docs/reference/builtins`,
      lastModified: currentDate,
      changeFrequency: 'daily',
      priority: 0.6,
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

  const blogPostRoutes: MetadataRoute.Sitemap = getAllBlogPosts().map(post => ({
    url: `${baseUrl}/blog/${post.slug}`,
    lastModified: currentDate,
    changeFrequency: 'weekly',
    priority: 0.4,
  }))

  const builtinRoutes: MetadataRoute.Sitemap = loadBuiltins().map(builtin => ({
    url: `${baseUrl}/docs/reference/builtins/${builtin.slug}`,
    lastModified: currentDate,
    changeFrequency: 'weekly',
    priority: 0.5,
  }))

  const seen = new Set<string>()
  const merged: MetadataRoute.Sitemap = []
  for (const route of [...staticRoutes, ...docRoutes, ...blogPostRoutes, ...builtinRoutes]) {
    if (seen.has(route.url)) continue
    seen.add(route.url)
    merged.push(route)
  }

  return merged
}
