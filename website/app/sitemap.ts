import { MetadataRoute } from 'next'

export const dynamic = 'force-static'

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = 'https://runmat.org'
  const currentDate = new Date()
  
  return [
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
      url: `${baseUrl}/docs/getting-started`,
      lastModified: currentDate,
      changeFrequency: 'daily',
      priority: 0.8,
    },
    {
      url: `${baseUrl}/docs/architecture`,
      lastModified: currentDate,
      changeFrequency: 'daily',
      priority: 0.7,
    },
    {
      url: `${baseUrl}/docs/cli`,
      lastModified: currentDate,
      changeFrequency: 'daily',
      priority: 0.7,
    },
    {
      url: `${baseUrl}/docs/configuration`,
      lastModified: currentDate,
      changeFrequency: 'daily',
      priority: 0.7,
    },
    {
      url: `${baseUrl}/docs/language-coverage`,
      lastModified: currentDate,
      changeFrequency: 'daily',
      priority: 0.6,
    },
    {
      url: `${baseUrl}/docs/how-it-works`,
      lastModified: currentDate,
      changeFrequency: 'daily',
      priority: 0.6,
    },
    {
      url: `${baseUrl}/docs/builtin-functions`,
      lastModified: currentDate,
      changeFrequency: 'daily',
      priority: 0.6,
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
  ]
}
