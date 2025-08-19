// Central docs manifest mapping slugs to on-disk files and navigation groups.
// We intentionally reference markdown living both in /docs and in crate subtrees
// to avoid duplicating documentation.

export type DocsNode = {
  title: string;
  slug?: string[]; // URL path segments under /docs
  file?: string; // absolute-ish repo path to markdown to render
  externalHref?: string; // for non-markdown custom pages
  seo?: {
    description?: string;
    keywords?: string[];
    ogTitle?: string;
    ogDescription?: string;
  };
  children?: DocsNode[];
};

// Navigation tree (order matters). Keep groups small and scannable.
export const docsTree: DocsNode[] = [
  {
    title: "Getting Started",
    externalHref: "/docs/getting-started",
  },
  {
    title: "How It Works",
    externalHref: "/docs/how-it-works",
  },
  {
    title: "Architecture",
    children: [
      { 
        title: "High-Level Architecture", 
        slug: ["architecture"], 
        file: "docs/ARCHITECTURE.md",
        seo: {
          description: "RunMat architecture overview: V8-inspired tiered execution (Ignition → Turbine), generational GC, runtime, plotting, and snapshot startup.",
          keywords: ["RunMat", "MATLAB alternative", "Octave alternative", "JIT", "interpreter", "GC", "scientific computing"],
          ogTitle: "RunMat Architecture",
          ogDescription: "Deep dive into RunMat's V8-inspired execution model, GC, runtime, and plotting subsystems.",
        }
      },
      { title: "Compiler Pipeline", slug: ["ignition", "compiler-pipeline"], file: "crates/runmat-ignition/COMPILER_PIPELINE.md", seo: { description: "How runmat-ignition lowers HIR to bytecode with short-circuit lowering, multi-assign shaping, and more.", keywords: ["compiler pipeline", "bytecode", "HIR", "MATLAB interpreter"] } },
      { title: "Instruction Set", slug: ["ignition", "instr-set"], file: "crates/runmat-ignition/INSTR_SET.md", seo: { description: "Complete reference for RunMat's Ignition bytecode opcodes and semantics.", keywords: ["instruction set", "opcodes", "bytecode", "MATLAB runtime"] } },
      { title: "Indexing & Slicing", slug: ["ignition", "indexing-and-slicing"], file: "crates/runmat-ignition/INDEXING_AND_SLICING.md", seo: { description: "MATLAB-compatible indexing and slicing semantics (gather/scatter, end arithmetic, logical masks).", keywords: ["MATLAB indexing", "slicing", "end", "colon", "logical indexing"] } },
      { title: "Error Model", slug: ["ignition", "error-model"], file: "crates/runmat-ignition/ERROR_MODEL.md", seo: { description: "Uniform MException identifiers across runtime failures: indexing, arity, expansion, and OOP.", keywords: ["MException", "error model", "MATLAB errors"] } },
      { title: "OOP Semantics", slug: ["ignition", "oop-semantics"], file: "crates/runmat-ignition/OOP_SEMANTICS.md", seo: { description: "How RunMat handles MATLAB classdef semantics: properties, methods, subsref/subsasgn, operator overloading.", keywords: ["MATLAB OOP", "classdef", "subsref", "operator overloading"] } },
    ],
  },
  {
    title: "Language",
    children: [
      { title: "Language Coverage", slug: ["language-coverage"], file: "docs/LANGUAGE_COVERAGE.md", seo: { description: "MATLAB language feature compatibility in RunMat with an Octave comparison.", keywords: ["MATLAB compatibility", "Octave compatibility", "language coverage", "MATLAB alternative", "Octave alternative"] } },
      { title: "CLI Reference", slug: ["cli"], file: "docs/CLI.md", seo: { description: "RunMat CLI commands, flags, environment variables, and examples.", keywords: ["RunMat CLI", "command line", "flags"] } },
      { title: "Configuration", slug: ["configuration"], file: "docs/CONFIG.md", seo: { description: "Configure RunMat: files, environment overrides, and precedence.", keywords: ["RunMat config", "configuration", "YAML", "TOML", "JSON"] } },
    ],
  },
  {
    title: "Runtime & Builtins",
    children: [
      { title: "Builtin Function Reference", externalHref: "/docs/reference/builtins" },
    ],
  },
  {
    title: "Accelerate (GPU)",
    children: [
      { title: "Accelerate Design", slug: ["accelerate", "overview"], file: "crates/runmat-accelerate/README.md", seo: { description: "RunMat GPU acceleration architecture: gpuArray/gather, planner, and backend design.", keywords: ["GPU", "accelerate", "gpuArray", "gather", "wgpu", "CUDA", "ROCm"] } },
      { title: "Accelerate API", slug: ["accelerate", "api"], file: "crates/runmat-accelerate-api/README.md", seo: { description: "Provider API between RunMat runtime and GPU backends (upload, download, free, device info).", keywords: ["GPU API", "provider", "acceleration"] } },
    ],
  },
];

export function findNodeBySlug(slug: string[]): DocsNode | undefined {
  const walk = (nodes: DocsNode[]): DocsNode | undefined => {
    for (const n of nodes) {
      if (n.slug && eq(n.slug, slug)) return n;
      if (n.children) {
        const f = walk(n.children);
        if (f) return f;
      }
    }
    return undefined;
  };
  return walk(docsTree);
}

export function flatten(nodes: DocsNode[] = docsTree): DocsNode[] {
  const out: DocsNode[] = [];
  const walk = (ns: DocsNode[]) => {
    for (const n of ns) {
      out.push(n);
      if (n.children) walk(n.children);
    }
  };
  walk(nodes);
  return out;
}

// Find the breadcrumb trail for a slug
export function findPathBySlug(slug: string[]): DocsNode[] | undefined {
  const path: DocsNode[] = [];
  const walk = (nodes: DocsNode[]): boolean => {
    for (const n of nodes) {
      path.push(n);
      if (n.slug && eq(n.slug, slug)) return true;
      if (n.children && walk(n.children)) return true;
      path.pop();
    }
    return false;
  };
  const ok = walk(docsTree);
  return ok ? path.slice() : undefined;
}

function eq(a: string[], b: string[]) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}


