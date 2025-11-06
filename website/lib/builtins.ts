import { readFileSync, existsSync } from 'fs';
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
  description?: string;
  signatures: BuiltinSignature[];
  errors?: string[];
  examples?: { title?: string; code: string }[];
  keywords?: string[];
  internal?: boolean;
  mdxPath?: string;
  firstParagraph?: string; // First paragraph extracted from MDX file
};

// Extract the first paragraph from MDX content (before first heading)
function extractFirstParagraphFromMDX(mdxContent: string): string | null {
  const lines = mdxContent.split(/\r?\n/);
  let inFence = false;
  const paraLines: string[] = [];
  
  for (const line of lines) {
    const trimmed = line.trim();
    
    // Skip code fences
    if (trimmed.startsWith('```')) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;
    
    // Stop at first heading
    if (trimmed.startsWith('#')) {
      break;
    }
    
    // Skip empty lines (but allow paragraph to accumulate)
    if (trimmed === '') {
      if (paraLines.length > 0) {
        break; // End of first paragraph
      }
      continue;
    }
    
    // Skip blockquotes, lists, etc.
    if (trimmed.startsWith('>') || trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      continue;
    }
    
    paraLines.push(trimmed);
  }
  
  if (paraLines.length === 0) {
    return null;
  }
  
  // Join lines and clean up
  let paragraph = paraLines.join(' ').trim();
  // Remove markdown code backticks but keep the content
  paragraph = paragraph.replace(/`([^`]+)`/g, '$1');
  // Remove markdown links but keep text
  paragraph = paragraph.replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1');
  
  return paragraph || null;
}

export function loadBuiltins(): Builtin[] {
  const p = join(process.cwd(), 'content', 'builtins.json');
  const data = readFileSync(p, 'utf-8');
  const arr = JSON.parse(data) as Builtin[];
  
  // Enhance builtins with first paragraph from MDX files
  return arr.map(builtin => {
    if (!builtin.mdxPath) {
      return builtin;
    }
    
    try {
      const mdxPath = join(process.cwd(), 'content', builtin.mdxPath);
      if (existsSync(mdxPath)) {
        const mdxContent = readFileSync(mdxPath, 'utf-8');
        const firstParagraph = extractFirstParagraphFromMDX(mdxContent);
        if (firstParagraph) {
          return { ...builtin, firstParagraph };
        }
      }
    } catch {
      // If MDX file doesn't exist or can't be read, continue without it
    }
    
    return builtin;
  });
}