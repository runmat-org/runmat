import { readFileSync, readdirSync, statSync, existsSync } from 'fs'
import { extname, join } from 'path'
import matter from 'gray-matter'

export interface BenchmarkSummary {
  slug: string
  title: string
  description: string
  summary: string
  imageUrl?: string
  date: string
  readTime: string
  author: string
  tags: string[]
}

function extractTitleFromMarkdown(content: string): string {
  const lines = content.split('\n')
  for (const line of lines) {
    const trimmed = line.trim()
    if (trimmed.startsWith('# ')) {
      return trimmed.substring(2).trim()
    }
  }
  return 'Untitled Benchmark'
}

function extractFirstParagraph(content: string): string {
  const lines = content.split('\n')
  let inParagraph = false
  const paragraphLines: string[] = []

  for (const line of lines) {
    const trimmed = line.trim()
    if (trimmed.startsWith('#')) {
      if (inParagraph) break
      continue
    }
    if (!trimmed && !inParagraph) continue
    if (trimmed) {
      inParagraph = true
      paragraphLines.push(trimmed)
    } else if (inParagraph) {
      break
    }
  }

  const paragraph = paragraphLines.join(' ')
  return paragraph || 'Performance benchmark comparing RunMat against alternatives.'
}

function truncateText(text: string, limit = 200): string {
  if (text.length <= limit) {
    return text
  }
  return text.substring(0, limit).trimEnd() + '...'
}

function extractFirstImageUrl(content: string): string | undefined {
  const imageRegex = /!\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)/
  const match = imageRegex.exec(content)
  return match ? match[1] : undefined
}

function getMimeTypeFromExtension(extension: string): string | undefined {
  switch (extension.toLowerCase()) {
    case '.png':
      return 'image/png'
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg'
    case '.svg':
      return 'image/svg+xml'
    case '.webp':
      return 'image/webp'
    default:
      return undefined
  }
}

export function getAllBenchmarks(): BenchmarkSummary[] {
  try {
    const benchmarksDir = join(process.cwd(), '..', 'benchmarks')
    const entries = readdirSync(benchmarksDir, { withFileTypes: true })

    const benchmarks = entries
      .filter(entry => entry.isDirectory() && entry.name !== '.harness' && entry.name !== 'wgpu_profile')
      .map((entry): BenchmarkSummary | null => {
        const slug = entry.name
        const readmePath = join(benchmarksDir, slug, 'README.md')

        try {
          const fileContent = readFileSync(readmePath, 'utf-8')
          const { data: frontmatter, content } = matter(fileContent)

          const title = frontmatter.title || extractTitleFromMarkdown(content)
          const rawDescription = frontmatter.description || frontmatter.excerpt || extractFirstParagraph(content)
          const description = rawDescription || 'Performance benchmark comparing RunMat against alternatives.'
          const summary = truncateText(description)

          const frontmatterImage = typeof frontmatter.image === 'string' ? frontmatter.image : undefined
          const markdownImage = extractFirstImageUrl(content)
          const resolvedImagePath = frontmatterImage || markdownImage

          let imageUrl: string | undefined
          if (resolvedImagePath) {
            if (resolvedImagePath.startsWith('http://') || resolvedImagePath.startsWith('https://')) {
              imageUrl = resolvedImagePath
            } else {
              const sanitizedPath = resolvedImagePath.replace(/^\.?\//, '')
              const absolutePath = join(benchmarksDir, slug, sanitizedPath)
              if (existsSync(absolutePath)) {
                const mimeType = getMimeTypeFromExtension(extname(absolutePath))
                if (mimeType) {
                  const fileBuffer = readFileSync(absolutePath)
                  const base64 = fileBuffer.toString('base64')
                  imageUrl = `data:${mimeType};base64,${base64}`
                }
              }
            }
          }

          const stats = statSync(readmePath)
          const defaultDate = stats.mtime.toISOString()

          return {
            slug,
            title,
            description,
            summary,
            imageUrl,
            date: frontmatter.date || defaultDate,
            readTime: frontmatter.readTime || '5 min read',
            author: frontmatter.author || 'RunMat Team',
            tags: frontmatter.tags || [],
          }
        } catch (error) {
          console.error(`Error reading benchmark ${slug}:`, error)
          return null
        }
      })
      .filter((benchmark): benchmark is BenchmarkSummary => benchmark !== null)

    return benchmarks.sort((a, b) => {
      const dateA = new Date(a.date).getTime()
      const dateB = new Date(b.date).getTime()
      if (dateA !== dateB) {
        return dateB - dateA
      }
      return a.title.localeCompare(b.title)
    })
  } catch (error) {
    console.error('Error reading benchmarks:', error)
    return []
  }
}

export function getBenchmarkSlugs(): string[] {
  return getAllBenchmarks().map(benchmark => benchmark.slug)
}
