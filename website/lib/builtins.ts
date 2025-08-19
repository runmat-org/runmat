import { readFileSync } from 'fs';
import { join } from 'path';

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
  status: 'implemented' | 'missing' | 'stubbed';
  signatures: BuiltinSignature[];
  errors?: string[];
  examples?: { title?: string; code: string }[];
  keywords?: string[];
};

export function loadBuiltins(): Builtin[] {
  const p = join(process.cwd(), 'content', 'builtins.json');
  const data = readFileSync(p, 'utf-8');
  const arr = JSON.parse(data) as Builtin[];
  return arr;
}

export function progress(builtins: Builtin[], universeTotal?: number) {
  const implemented = builtins.filter((b) => b.status === 'implemented').length;
  let total = universeTotal ?? builtins.length;
  try {
    const manifestPath = join(process.cwd(), 'docs', 'matlab_core_manifest.json');
    const s = readFileSync(manifestPath, 'utf-8');
    const u = JSON.parse(s) as { name: string }[];
    total = u.length;
  } catch {}
  return { implemented, total, pct: total ? implemented / total : 0 };
}


