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
  signatures: BuiltinSignature[];
  errors?: string[];
  examples?: { title?: string; code: string }[];
  keywords?: string[];
  internal?: boolean;
};

export function loadBuiltins(): Builtin[] {
  const p = join(process.cwd(), 'content', 'builtins.json');
  const data = readFileSync(p, 'utf-8');
  const arr = JSON.parse(data) as Builtin[];
  return arr;
}