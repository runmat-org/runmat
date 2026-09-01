export type BuiltinDocExample = {
  id?: string;
  description?: string;
  label?: string;
  input?: string;
  output?: string;
  compatibility?: "RunMat" | "Matlab" | "Strict";
  harness?:
    | "Portable"
    | "Native"
    | "Browser"
    | "BrowserGraphics"
    | "NativeFilesystem"
    | "NativeLoopbackNetwork"
    | "Wgpu"
    | "NativeForeignRuntime"
    | "InteractiveHost";
  verification?:
    | "Succeeds"
    | { Assertions: { source: string } }
    | { ExpectedError: { identifier: string } }
    | { Figure: { minimum_figures: number; assertions: string } };
  image?: string;
  image_webp?: string;
  matlab_script?: string;
};

export type BuiltinDocFAQ = {
  question: string;
  answer: string;
};

export type BuiltinDocSection = {
  heading: string;
  paragraphs: string[];
};

export type BuiltinDocEvidence = {
  implementation: Array<{ label: string; target: unknown }>;
  verification: Array<{
    kind: "UnitTest" | "IntegrationTest" | "BrowserTest" | "ProviderTest" | "WgpuTest" | "ConformanceTest" | "Validation";
    label: string;
    location: string;
  }>;
  notes: string[];
};

export type BuiltinDocLink = {
  label: string;
  url: string;
  thumbnail?: string;
};

export type BuiltinDocSyntax = {
  example: BuiltinDocExample;
  points?: string[];
} | string[];


export type BuiltinDocJsonEncodeOption = {
  name: string;
  type: string;
  default: string;
  description: string;
};

export type BuiltinDocValidation = {
  summary: string;
  implementation?: BuiltinDocLink;
  parity_test?: BuiltinDocLink;
  tolerance?: string;
};

export type BuiltinDoc = {
  // Reference documents carry domain-specific metadata (for example plotting
  // signatures, graph status, or implementation notes) in addition to this
  // normalized index surface. Preserve those fields for consumers without
  // weakening the types of the normalized fields below.
  [metadata: string]: unknown;
  key: string;
  authority?: "catalog" | "legacy_sidecar";
  title: string;
  slug: string;
  aliases?: string[];
  category: string;
  categoryPath: string[];
  keywords: string[];
  summary: string;
  hero_image?: string | null;
  description?: string;
  behaviors?: string[];
  sections?: BuiltinDocSection[];
  extended_capabilities?: string[];
  examples?: Array<BuiltinDocExample | string>;
  faqs?: BuiltinDocFAQ[];
  links?: BuiltinDocLink[];
  source?: BuiltinDocLink | BuiltinDocLink[];
  evidence?: BuiltinDocEvidence;
  example_exemption?: string | null;
  catalog?: Record<string, unknown>;
  gpu_residency?: string;
  gpu_behavior?: string[];
  options?: string[];
  syntax?: BuiltinDocSyntax;
  jsonencode_options?: BuiltinDocJsonEncodeOption[];
  gpu_support?: Record<string, unknown>;
  fusion?: Record<string, unknown>;
  requires_feature?: string | null;
  tested?: Record<string, string | string[] | null>;
  validation?: BuiltinDocValidation;
};

export type BuiltinManifestEntry = Pick<
  BuiltinDoc,
  "key" | "title" | "slug" | "aliases" | "category" | "categoryPath" | "keywords" | "summary"
> & {
  exampleCount: number;
};

export type BuiltinExampleCatalogEntry = {
  id: string;
  builtinKey: string;
  builtinTitle: string;
  builtinSlug: string;
  category: string;
  categoryPath: string[];
  exampleTitle: string;
  summary: string;
  code: string;
  output?: string;
  keywords: string[];
  suggestedPath: string;
};

export type BuiltinDocLoader = () => Promise<BuiltinDoc>;

import {
  builtinExamplesCatalogLoader,
  builtinDocLoaders,
  builtinManifest
} from "./generated/builtins-manifest.js";

const builtinManifestByKey = new Map<string, BuiltinManifestEntry>(
  builtinManifest.map((entry) => [entry.key, entry])
);

const builtinDocCache = new Map<string, Promise<BuiltinDoc>>();

export function normalizeBuiltinKey(value: string): string {
  return value.trim().toLowerCase();
}

export function slugFromBuiltinTitle(title: string): string {
  return title.trim().toLowerCase();
}

export function categoryPathFromCategory(category?: string | null): string[] {
  if (!category) {
    return [];
  }
  return category
    .split("/")
    .map((part) => part.trim())
    .filter(Boolean);
}

export function getBuiltinManifest(): BuiltinManifestEntry[] {
  return builtinManifest;
}

export function listBuiltinKeys(): string[] {
  return builtinManifest.map((entry) => entry.key);
}

export function getBuiltinManifestEntry(key: string): BuiltinManifestEntry | undefined {
  return builtinManifestByKey.get(normalizeBuiltinKey(key));
}

export async function loadBuiltinDoc(key: string): Promise<BuiltinDoc | null> {
  const normalizedKey = normalizeBuiltinKey(key);
  const loader = builtinDocLoaders[normalizedKey];
  if (!loader) {
    return null;
  }
  let pending = builtinDocCache.get(normalizedKey);
  if (!pending) {
    pending = loader();
    builtinDocCache.set(normalizedKey, pending);
  }
  return pending;
}

export async function loadBuiltinDocs(keys: string[]): Promise<BuiltinDoc[]> {
  const docs = await Promise.all(keys.map((key) => loadBuiltinDoc(key)));
  return docs.filter((doc): doc is BuiltinDoc => doc !== null);
}

export async function loadAllBuiltinDocs(): Promise<BuiltinDoc[]> {
  return loadBuiltinDocs(listBuiltinKeys());
}

export async function loadBuiltinExamplesCatalog(): Promise<BuiltinExampleCatalogEntry[]> {
  return builtinExamplesCatalogLoader();
}
