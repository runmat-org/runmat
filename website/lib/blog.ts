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
  tags: string[]
  image?: string
  imageAlt?: string
}

export function getAllBlogPosts(): BlogPost[] {
  try {
    const blogDir = join(process.cwd(), 'content/blog')
    const files = readdirSync(blogDir).filter(file => file.endsWith('.md'))

    const posts = files.map(file => {
      const slug = file.replace(/\.md$/, '')
      const filePath = join(blogDir, file)
      const fileContent = readFileSync(filePath, 'utf-8')
      const { data: frontmatter } = matter(fileContent)

      return {
        slug,
        title: frontmatter.title || 'Untitled',
        description: frontmatter.description || frontmatter.excerpt || '',
        date: frontmatter.date || new Date().toISOString(),
        readTime: frontmatter.readTime || '5 min read',
        author: frontmatter.author || 'RunMat Team',
        tags: frontmatter.tags || [],
        image: frontmatter.image,
        imageAlt: frontmatter.imageAlt,
      }
    })

    return posts.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
  } catch (error) {
    console.error('Error reading blog posts:', error)
    return []
  }
}
