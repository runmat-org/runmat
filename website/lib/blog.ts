import { readFileSync, readdirSync } from 'fs'
import { join } from 'path'
import matter from 'gray-matter'

export interface BlogPost {
  slug: string
  title: string
  description: string
  date: string
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
  author?: string
  authors?: FrontmatterAuthorEntry[]
  tags?: string[]
  image?: string
  imageAlt?: string
  visibility?: 'public' | 'unlisted'
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

export function getAllBlogPosts(): BlogPost[] {
  try {
    const blogDir = join(process.cwd(), 'content/blog')
    const files = readdirSync(blogDir).filter(file => file.endsWith('.md'))

    const posts = files.map(file => {
      const slug = slugifyFileName(file)
      const filePath = join(blogDir, file)
      const fileContent = readFileSync(filePath, 'utf-8')
      const { data } = matter(fileContent)
      const frontmatter = data as BlogFrontmatter
      const authors = normalizeAuthors(frontmatter)
      const visibility: 'public' | 'unlisted' =
        frontmatter.visibility === 'unlisted' ? 'unlisted' : 'public'

      return {
        slug,
        title: frontmatter.title || 'Untitled',
        description: frontmatter.description || frontmatter.excerpt || '',
        date: frontmatter.date || new Date().toISOString(),
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
