import { MetadataRoute } from 'next'

export const dynamic = 'force-static'

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: '*',
      allow: '/',
      disallow: ['/api/', '/admin/', '/login', '/invite/', '/email-verified'],
    },
    sitemap: 'https://runmat.com/sitemap.xml',
    host: 'https://runmat.com',
  }
}
