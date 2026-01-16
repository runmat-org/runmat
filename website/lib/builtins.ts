import { readFileSync, readdirSync, realpathSync } from 'fs';
import { join } from 'path';
import type { BuiltinBadge } from './badge-utils';
import { builtinHasGpuSupport, getBuiltinBadges } from './badge-utils';

export type BuiltinSignature = {
  in: string[];
  inTypes?: string[];
  out: string[];
  outTypes?: string[];
  nargin: { min: number; max: number };
  nargout: { min: number; max: number };
};

export type Builtin = {
  name: string;
  slug: string;
  category: string[];
  summary: string;
  description?: string;
  signatures: BuiltinSignature[];
  errors?: string[];
  examples?: { title?: string; code: string }[];
  keywords?: string[];
  internal?: boolean;
  firstParagraph?: string;
};

export type BuiltinDocExample = {
  description: string;
  input: string;
  output?: string;
};

export type BuiltinDocFAQ = {
  question: string;
  answer: string;
};

export type BuiltinDocLink = {
  label: string;
  url: string;
};

export type BuiltinDocSyntax = {
  example: BuiltinDocExample;
  points: string[];
};

export type BuiltinDocJsonEncodeOption = {
  name: string;
  type: string;
  default: string;
  description: string;
};

export type BuiltinDoc = {
  title: string;
  category: string;
  keywords: string[];
  summary: string;
  references: string[];
  description?: string;
  behaviors?: string[];
  examples?: BuiltinDocExample[];
  faqs?: BuiltinDocFAQ[];
  links?: BuiltinDocLink[];
  source?: BuiltinDocLink;
  gpu_residency?: string;
  gpu_behavior?: string[];
  options?: string[];
  syntax?: BuiltinDocSyntax;
  jsonencode_options?: BuiltinDocJsonEncodeOption[];
  gpu_support?: Record<string, unknown>;
  fusion?: Record<string, unknown>;
  requires_feature?: string | null;
  tested?: Record<string, string | string[]>;
};

export type BuiltinDocEntry = BuiltinDoc & { slug: string };

const BUILTINS_DOCS_DIR = realpathSync(join(process.cwd(), 'content', 'builtins-json'));
let builtinDocsCache: BuiltinDocEntry[] | null = null;

function slugFromTitle(title: string): string {
  return title.toLowerCase();
}

function normalizeStringArray(values?: unknown): string[] | undefined {
  if (Array.isArray(values)) {
    const trimmed = values.map((value) => String(value).trim()).filter(Boolean);
    return trimmed.length > 0 ? trimmed : undefined;
  }
  if (typeof values === 'string') {
    const trimmed = values.trim();
    return trimmed ? [trimmed] : undefined;
  }
  return undefined;
}

export function loadBuiltinDocs(): BuiltinDocEntry[] {
  if (builtinDocsCache) return builtinDocsCache;
  const entries = readdirSync(BUILTINS_DOCS_DIR, { withFileTypes: true });
  const docs: BuiltinDocEntry[] = [];
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith('.json')) continue;
    const raw = readFileSync(join(BUILTINS_DOCS_DIR, entry.name), 'utf-8');
    const parsed = JSON.parse(raw) as BuiltinDoc;
    const title = typeof parsed.title === 'string' ? parsed.title : entry.name.replace(/\.json$/, '');
    const normalized: BuiltinDoc = {
      ...parsed,
      title,
      gpu_behavior: normalizeStringArray(parsed.gpu_behavior),
    };
    docs.push({ ...normalized, slug: slugFromTitle(title) });
  }
  builtinDocsCache = docs;
  return docs;
}

export function getBuiltinDocBySlug(slug: string): BuiltinDocEntry | undefined {
  return loadBuiltinDocs().find((doc) => doc.slug === slug);
}

export function loadBuiltins(): Builtin[] {
  return loadBuiltinDocs().map((doc) => ({
    name: doc.title,
    slug: doc.slug,
    category: doc.category ? [doc.category] : [],
    summary: doc.summary ?? '',
    description: doc.description,
    signatures: [],
    keywords: doc.keywords ?? [],
    internal: false,
  }));
}

/**
 * Formats a category array like ["array/sorting_sets"] into a readable label like "Array: Sorting & Sets"
 */
export function formatCategoryLabel(category: string[]): string {
  if (!category || category.length === 0) {
    return 'General';
  }
  
  // Take the first category (most specific)
  const cat = category[0];
  
  // Split by '/' and format each part
  const parts = cat.split('/').map(part => {
    // Convert snake_case or kebab-case to Title Case
    return part
      .split(/[-_]/)
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  });
  
  // Join with ': ' if multiple parts, otherwise just return the formatted part
  if (parts.length > 1) {
    return `${parts[0]}: ${parts.slice(1).join(' & ')}`;
  }
  
  return parts[0];
}

/**
 * Extracts metadata for a builtin function
 */
export type BuiltinMetadata = {
  category: string;
  gpuSupport: boolean;
  badges: BuiltinBadge[];
};

export function getBuiltinMetadata(builtin: Builtin): BuiltinMetadata {
  const badges = getBuiltinBadges(builtin);
  return {
    category: formatCategoryLabel(builtin.category),
    gpuSupport: builtinHasGpuSupport(builtin),
    badges,
  };
}
