import { existsSync, readFileSync, readdirSync } from 'fs'
import { basename, join } from 'path'
import matter from 'gray-matter'

export interface BlogPost {
  slug: string
  title: string
  description: string
  date: string
  dateModified?: string
  readTime: string
  author: string
  authors: AuthorInfo[]
  tags: string[]
  image?: string
  imageAlt?: string
  visibility: 'public' | 'unlisted'
}

export interface AuthorInfo {
  name: string
  url?: string
}

type FrontmatterAuthorEntry =
  | string
  | {
      name?: string
      url?: string
    }
  | null
  | undefined

interface BlogFrontmatter {
  title?: string
  description?: string
  excerpt?: string
  date?: string
  readTime?: string
  slug?: string
  author?: string
  authors?: FrontmatterAuthorEntry[]
  tags?: string[]
  image?: string
  imageAlt?: string
  visibility?: 'public' | 'unlisted'
  dateModified?: string
  [key: string]: unknown
}

function slugifyFileName(file: string): string {
  const base = file.replace(/\.md$/, '').toLowerCase()
  const normalized = base
    .replace(/[^a-z0-9]+/g, '-') // keep alnum, replace runs with hyphen
    .replace(/^-+|-+$/g, '') // trim leading/trailing hyphens

  return normalized || 'post'
}

function normalizeAuthors(frontmatter: BlogFrontmatter): AuthorInfo[] {
  const result: AuthorInfo[] = []

  if (Array.isArray(frontmatter.authors)) {
    for (const entry of frontmatter.authors) {
      if (!entry) continue
      if (typeof entry === 'string') {
        result.push({ name: entry })
        continue
      }
      if (typeof entry === 'object' && 'name' in entry && typeof entry.name === 'string') {
        result.push({
          name: entry.name,
          url: typeof entry.url === 'string' ? entry.url : undefined,
        })
      }
    }
  }

  if (result.length === 0 && typeof frontmatter.author === 'string') {
    result.push({ name: frontmatter.author })
  }

  if (result.length === 0) {
    result.push({ name: 'RunMat Team' })
  }

  return result
}

function resolveFrontmatterSlug(frontmatter: BlogFrontmatter, file: string): string {
  if (typeof frontmatter.slug === 'string' && frontmatter.slug.trim().length > 0) {
    return frontmatter.slug.trim()
  }
  return slugifyFileName(file)
}

function readFrontmatter(filePath: string) {
  const fileContent = readFileSync(filePath, 'utf-8')
  const { data } = matter(fileContent)
  const frontmatter = data as BlogFrontmatter
  const slug = resolveFrontmatterSlug(frontmatter, basename(filePath))
  return { frontmatter, slug }
}

export function getAllBlogPosts(): BlogPost[] {
  try {
    const blogDir = join(process.cwd(), 'content/blog')
    const files = readdirSync(blogDir).filter(file => file.endsWith('.md'))

    const posts = files.map(file => {
      const filePath = join(blogDir, file)
      const { frontmatter, slug } = readFrontmatter(filePath)
      const authors = normalizeAuthors(frontmatter)
      const visibility: 'public' | 'unlisted' =
        frontmatter.visibility === 'unlisted' ? 'unlisted' : 'public'

      return {
        slug,
        title: frontmatter.title || 'Untitled',
        description: frontmatter.description || frontmatter.excerpt || '',
        date: frontmatter.date || new Date().toISOString(),
        dateModified: frontmatter.dateModified,
        readTime: frontmatter.readTime || '5 min read',
        author: authors.map(author => author.name).join(', '),
        authors,
        tags: Array.isArray(frontmatter.tags) ? frontmatter.tags : [],
        image: frontmatter.image,
        imageAlt: frontmatter.imageAlt,
        visibility,
      }
    })

    return posts.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
  } catch (error) {
    console.error('Error reading blog posts:', error)
    return []
  }
}

export function getPublicBlogPosts(): BlogPost[] {
  return getAllBlogPosts().filter(post => post.visibility !== 'unlisted')
}

export function resolveBlogFilePath(slug: string): string | null {
  const blogDir = join(process.cwd(), 'content/blog')
  const directPath = join(blogDir, `${slug}.md`)
  if (existsSync(directPath)) {
    return directPath
  }

  try {
    const files = readdirSync(blogDir).filter(file => file.endsWith('.md'))
    for (const file of files) {
      const filePath = join(blogDir, file)
      const { frontmatter } = readFrontmatter(filePath)
      if (typeof frontmatter.slug === 'string' && frontmatter.slug.trim() === slug) {
        return filePath
      }
    }
  } catch (error) {
    console.warn('Error resolving blog file path:', error)
  }

  return null
}
