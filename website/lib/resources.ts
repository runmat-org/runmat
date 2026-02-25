import { existsSync, readFileSync, readdirSync, statSync } from 'fs'
import { basename, join } from 'path'
import matter from 'gray-matter'

import { getPublicBlogPosts } from './blog'
import { getAllBenchmarks } from './benchmarks'
import { docsTree, flatten as flattenDocs, DocsNode } from '@/content/docs'
import * as curated from '@/content/resources'

export type ResourceType =
  | 'docs'
  | 'guides'
  | 'blogs'
  | 'case-studies'
  | 'webinars'
  | 'benchmarks'

export type ResourceSource = 'resource' | 'blog' | 'doc' | 'benchmark'

export interface ResourceItem {
  id: string
  slug: string
  title: string
  description: string
  href: string
  type: ResourceType
  source: ResourceSource
  date?: string
  dateModified?: string
  readTime?: string
  tags?: string[]
  image?: string
  imageAlt?: string
  featured?: boolean
}

type NativeFrontmatter = {
  title?: string
  description?: string
  excerpt?: string
  date?: string
  readTime?: string
  tags?: string[]
  resourceType?: string
  link?: string
  image?: string
  imageAlt?: string
}

const TYPE_LABELS: Record<ResourceType, string> = {
  docs: 'Docs',
  guides: 'Guides',
  blogs: 'Blogs',
  'case-studies': 'Case Studies',
  webinars: 'Webinars',
  benchmarks: 'Benchmarks',
}

const RESOURCE_TYPE_FOLDERS: Record<ResourceType, string> = {
  docs: 'docs',
  guides: 'guides',
  blogs: 'blogs',
  'case-studies': 'case-studies',
  webinars: 'webinars',
  benchmarks: 'benchmarks',
}

function normalizeResourceType(value?: string): ResourceType | null {
  if (!value) return null
  const v = value.trim().toLowerCase()
  if (v === 'doc' || v === 'docs' || v === 'documentation') return 'docs'
  if (v === 'guide' || v === 'guides') return 'guides'
  if (v === 'blog' || v === 'blogs' || v === 'news' || v === 'qna' || v === 'q&a') return 'blogs'
  if (v === 'case-study' || v === 'case studies' || v === 'case-studies' || v === 'case study')
    return 'case-studies'
  if (v === 'webinar' || v === 'webinars') return 'webinars'
  if (v === 'benchmark' || v === 'benchmarks') return 'benchmarks'
  return null
}

function slugFromFileName(file: string): string {
  return basename(file, '.md')
}

function warn(message: string, meta?: unknown) {
  // Keep warnings non-fatal; Resources should be best-effort.
  console.warn(`[resources] ${message}`, meta ?? '')
}

function buildDocHref(node: DocsNode): string | null {
  if (node.slug) return `/docs/${node.slug.join('/')}`
  if (node.externalHref && node.externalHref.startsWith('/')) return node.externalHref
  return null
}

function resolveDocByPath(path: string): DocsNode | null {
  const normalized = path.replace(/^\/+/, '').replace(/^docs\//, '')
  const segments = normalized.split('/').filter(Boolean)
  const targetPath = `/docs/${segments.join('/')}`
  const nodes = flattenDocs(docsTree)
  for (const node of nodes) {
    const href = buildDocHref(node)
    if (href === targetPath) return node
  }
  return null
}

function docMtime(node: DocsNode): string | undefined {
  if (!node.file) return undefined
  const candidates = [
    join(process.cwd(), node.file),
    join(process.cwd(), '..', node.file),
    join(process.cwd(), '..', '..', node.file),
  ]
  for (const p of candidates) {
    if (existsSync(p)) {
      try {
        return statSync(p).mtime.toISOString()
      } catch {
        return undefined
      }
    }
  }
  return undefined
}

function coerceStringArray(val: unknown): string[] | undefined {
  if (val === undefined) return undefined
  if (!Array.isArray(val)) return undefined
  const out = val.filter((v): v is string => typeof v === 'string').map(v => v.trim()).filter(Boolean)
  return out.length ? out : undefined
}

function loadNativeResources(): ResourceItem[] {
  const baseDir = join(process.cwd(), 'content', 'resources')
  const items: ResourceItem[] = []

  for (const type of Object.keys(RESOURCE_TYPE_FOLDERS) as ResourceType[]) {
    const folder = RESOURCE_TYPE_FOLDERS[type]
    const dir = join(baseDir, folder)
    if (!existsSync(dir)) continue

    const files = readdirSync(dir).filter(f => f.endsWith('.md'))
    for (const file of files) {
      const filePath = join(dir, file)
      try {
        const raw = readFileSync(filePath, 'utf-8')
        const { data, content } = matter(raw)
        const fm = data as NativeFrontmatter

        const fmType = normalizeResourceType(fm.resourceType)
        const resolvedType = fmType ?? type
        if (!resolvedType) {
          warn(`Skipping resource without type`, { file: filePath, resourceType: fm.resourceType })
          continue
        }

        const slug = slugFromFileName(file)
        const title = fm.title || slug.replace(/-/g, ' ')
        const description = fm.description || fm.excerpt || content.trim().slice(0, 200) || ''
        const date =
          fm.date ||
          (() => {
            try {
              return statSync(filePath).mtime.toISOString()
            } catch {
              return undefined
            }
          })()

        items.push({
          id: `resource:${resolvedType}:${slug}`,
          slug,
          title,
          description,
          href: fm.link || `/resources/${resolvedType}/${slug}`,
          type: resolvedType,
          source: 'resource',
          date,
          readTime: fm.readTime,
          tags: fm.tags,
          image: fm.image,
          imageAlt: fm.imageAlt,
        })
      } catch (error) {
        warn(`Failed to read native resource`, { file: filePath, error })
      }
    }
  }

  return items
}

function loadBlogResources(): ResourceItem[] {
  return getPublicBlogPosts().map((post) => {
    const type = normalizeResourceType(post.resourceType || undefined) ?? 'blogs'
    return {
      id: `blog:${post.slug}`,
      slug: post.slug,
      title: post.title,
      description: post.description,
      href: `/blog/${post.slug}`,
      type,
      source: 'blog',
      date: post.date,
      dateModified: post.dateModified,
      readTime: post.readTime,
      tags: post.tags,
      image: post.image,
      imageAlt: post.imageAlt,
    } satisfies ResourceItem
  })
}

function loadCuratedDocs(): ResourceItem[] {
  const entries = Array.isArray(curated.curatedDocs) ? curated.curatedDocs : []
  const items: ResourceItem[] = []
  for (const entry of entries) {
    const path = typeof entry.slug === 'string' ? entry.slug : ''
    const type = normalizeResourceType((entry as { type?: string }).type || 'docs') ?? 'docs'
    const node = resolveDocByPath(path)
    if (!node) {
      warn('Doc entry not found in manifest', entry)
      continue
    }
    const href = buildDocHref(node)
    if (!href) {
      warn('Doc entry lacks href', entry)
      continue
    }
    items.push({
      id: `doc:${href}`,
      slug: href.replace(/^\/docs\//, ''),
      title: node.title,
      description: node.seo?.description || `Documentation: ${node.title}`,
      href,
      type,
      source: 'doc',
      date: docMtime(node),
      tags: node.seo?.keywords,
    })
  }
  return items
}

function loadCuratedBenchmarks(): ResourceItem[] {
  const entries = Array.isArray(curated.curatedBenchmarks) ? curated.curatedBenchmarks : []
  const allBenchmarks = getAllBenchmarks()
  const items: ResourceItem[] = []

  for (const entry of entries) {
    const type = normalizeResourceType((entry as { type?: string }).type || 'benchmarks') ?? 'benchmarks'
    if (!type) {
      warn('Benchmark entry missing type', entry)
      continue
    }
    const benchmark = allBenchmarks.find(b => b.slug === entry.slug)
    if (!benchmark) {
      warn('Benchmark entry not found', entry)
      continue
    }
    items.push({
      id: `benchmark:${benchmark.slug}`,
      slug: benchmark.slug,
      title: benchmark.title,
      description: benchmark.summary || benchmark.description,
      href: `/benchmarks/${benchmark.slug}`,
      type,
      source: 'benchmark',
      date: benchmark.date,
      dateModified: benchmark.date,
      readTime: benchmark.readTime,
      tags: benchmark.tags,
      image: benchmark.imageUrl,
    })
  }

  return items
}

function uniqueById(items: ResourceItem[]): ResourceItem[] {
  const seen = new Set<string>()
  const result: ResourceItem[] = []
  for (const item of items) {
    if (seen.has(item.id)) continue
    seen.add(item.id)
    result.push(item)
  }
  return result
}

function buildResourceIndex(): ResourceItem[] {
  const aggregated = [
    ...loadNativeResources(),
    ...loadBlogResources(),
    ...loadCuratedBenchmarks(),
  ]
  const deduped = uniqueById(aggregated)
  deduped.sort((a, b) => {
    const da = (() => {
      const d = a.dateModified || a.date
      return d ? new Date(d).getTime() : 0
    })()
    const db = (() => {
      const d = b.dateModified || b.date
      return d ? new Date(d).getTime() : 0
    })()
    if (db !== da) return db - da
    return a.title.localeCompare(b.title)
  })
  return deduped
}

function findResourceByRef(ref: { kind: string; slug: string; type?: string }): ResourceItem | null {
  const all = buildResourceIndex()
  const type = normalizeResourceType(ref.type || undefined)
  if (ref.kind === 'blog') {
    const found = all.find(r => r.source === 'blog' && r.slug === ref.slug)
    if (found && type && found.type !== type) {
      return { ...found, type }
    }
    return found || null
  }
  if (ref.kind === 'doc') {
    return (
      all.find(r => r.source === 'doc' && (r.slug === ref.slug || r.href.endsWith(ref.slug))) || null
    )
  }
  if (ref.kind === 'benchmark') {
    const found = all.find(r => r.source === 'benchmark' && r.slug === ref.slug)
    if (found && type && found.type !== type) return { ...found, type }
    return found || null
  }
  if (ref.kind === 'resource') {
    return all.find(r => r.source === 'resource' && r.slug === ref.slug) || null
  }
  return null
}

export function getAllResources(): ResourceItem[] {
  return buildResourceIndex()
}

export function getResourcesByType(type: ResourceType): ResourceItem[] {
  const normalized = normalizeResourceType(type)
  if (!normalized) return []
  return getAllResources().filter(r => r.type === normalized)
}

export function getFeaturedResources(): ResourceItem[] {
  const refs = Array.isArray(curated.featuredResources) ? curated.featuredResources : []
  const curatedItems: ResourceItem[] = []
  for (const ref of refs) {
    const item = findResourceByRef(ref as { kind: string; slug: string; type?: string })
    if (!item) {
      warn('Featured resource missing', ref)
      continue
    }
    curatedItems.push(item)
  }

  const flagged = getAllResources().filter(r => r.featured)
  // Priority: flagged (frontmatter) first by recency, then curated order, all deduped.
  flagged.sort((a, b) => {
    const da = (() => {
      const d = a.dateModified || a.date
      return d ? new Date(d).getTime() : 0
    })()
    const db = (() => {
      const d = b.dateModified || b.date
      return d ? new Date(d).getTime() : 0
    })()
    if (db !== da) return db - da
    return a.title.localeCompare(b.title)
  })

  const combined = uniqueById([...flagged, ...curatedItems])
  return combined
}

export function getLatestResources(limit = 12): ResourceItem[] {
  return getAllResources().slice(0, limit)
}

export const RESOURCE_TYPES: ResourceType[] = ['docs', 'guides', 'blogs', 'case-studies', 'webinars', 'benchmarks']
const HIDDEN_TILE_TYPES: ResourceType[] = ['case-studies', 'webinars']

export function getAvailableResourceTypes(): ResourceType[] {
  const all = getAllResources()
  const seen = new Set<ResourceType>()
  for (const item of all) {
    if (item.type) seen.add(item.type)
  }
  return RESOURCE_TYPES.filter(t => seen.has(t))
}

export function getRoutableResourceTypes(): ResourceType[] {
  // Only native types should have /resources/<type> pages; exclude hidden types too.
  return getAvailableResourceTypes().filter(t => t !== 'blogs' && t !== 'docs' && t !== 'benchmarks' && !HIDDEN_TILE_TYPES.includes(t))
}

export function getResourceTypeLink(type: ResourceType): string {
  if (type === 'blogs') return '/blog'
  if (type === 'docs') return '/docs'
  if (type === 'benchmarks') return '/benchmarks'
  if (HIDDEN_TILE_TYPES.includes(type)) return '#'
  return `/resources/${type}`
}

export function getDisplayResourceTypes(): ResourceType[] {
  // Types shown as tiles; hide case-studies/webinars until destinations exist.
  return getAvailableResourceTypes().filter(t => !HIDDEN_TILE_TYPES.includes(t))
}

function readCollectionsFromFile(filePath: string): string[] | undefined {
  try {
    const raw = readFileSync(filePath, 'utf-8')
    const { data } = matter(raw)
    return coerceStringArray((data as { collections?: unknown }).collections)
  } catch {
    return undefined
  }
}

function loadGuidesFromBlogs(): ResourceItem[] {
  return getPublicBlogPosts()
    .filter(post => Array.isArray(post.collections) && post.collections.map(c => c.toLowerCase()).includes('guides'))
    .map(post => ({
      id: `guides:blog:${post.slug}`,
      slug: post.slug,
      title: post.title,
      description: post.description,
      href: `/blog/${post.slug}`,
      type: 'guides',
      source: 'blog',
      date: post.date,
      dateModified: post.dateModified,
      readTime: post.readTime,
      tags: post.tags,
      image: post.image,
      imageAlt: post.imageAlt,
    }))
}

function loadGuidesFromDocs(): ResourceItem[] {
  const items: ResourceItem[] = []
  const nodes = flattenDocs(docsTree).filter(n => n.file && buildDocHref(n))
  for (const node of nodes) {
    const href = buildDocHref(node)
    if (!href || !node.file) continue
    const candidates = [
      join(process.cwd(), node.file),
      join(process.cwd(), '..', node.file),
      join(process.cwd(), '..', '..', node.file),
    ]
    let collections: string[] | undefined
    let docImage: string | undefined
    let docImageAlt: string | undefined
    for (const p of candidates) {
      if (existsSync(p)) {
        try {
          const raw = readFileSync(p, 'utf-8')
          const { data } = matter(raw)
          collections = coerceStringArray((data as { collections?: unknown }).collections)
          docImage = (data as { image?: string }).image
          docImageAlt = (data as { imageAlt?: string }).imageAlt
        } catch { /* skip */ }
        if (collections) break
      }
    }
    if (!collections || !collections.map(c => c.toLowerCase()).includes('guides')) continue
    items.push({
      id: `guides:doc:${href}`,
      slug: href.replace(/^\/docs\//, ''),
      title: node.title,
      description: node.seo?.description || `Documentation: ${node.title}`,
      href,
      type: 'guides',
      source: 'doc',
      date: docMtime(node),
      tags: node.seo?.keywords,
      image: docImage,
      imageAlt: docImageAlt,
    })
  }
  return items
}

function loadGuidesNative(): ResourceItem[] {
  return getAllResources().filter(r => r.type === 'guides')
}

export function getGuidesCollection(): ResourceItem[] {
  const aggregated = [
    ...loadGuidesNative(),
    ...loadGuidesFromBlogs(),
    ...loadGuidesFromDocs(),
  ]
  const deduped = uniqueById(aggregated)
  return deduped.sort((a, b) => {
    const da = a.date ? new Date(a.date).getTime() : 0
    const db = b.date ? new Date(b.date).getTime() : 0
    if (db !== da) return db - da
    return a.title.localeCompare(b.title)
  })
}

export function resourceTypeLabel(type: ResourceType): string {
  return TYPE_LABELS[type] ?? type
}

