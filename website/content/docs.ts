// Central docs manifest mapping slugs to on-disk files and navigation groups.
// We intentionally reference markdown living both in /docs and in crate subtrees
// to avoid duplicating documentation.

export type DocsNode = {
  title: string;
  slug?: string[]; // URL path segments under /docs
  file?: string; // absolute-ish repo path to markdown to render
  externalHref?: string; // for non-markdown custom pages
  hideChildrenInNav?: boolean; // keep children from rendering in sidebar
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
    title: "Overview",
    children: [
      { title: "Install", externalHref: "/download" },
      { title: "Getting Started", externalHref: "/docs/getting-started" },
      {
        title: "Browser Guide",
        slug: ["desktop-browser-guide"],
        file: "docs/DESKTOP_BROWSER_GUIDE.md",
        seo: {
          description: "Use the RunMat sandbox, a browser-based IDE for writing and running MATLAB-style code with GPU acceleration, cloud storage, and file versioning.",
          keywords: ["RunMat sandbox", "RunMat browser", "browser IDE", "sandbox", "WebAssembly", "GPU acceleration", "MATLAB online", "cloud storage", "file versioning"],
          ogTitle: "Browser Guide",
          ogDescription: "Use the RunMat sandbox—a browser-based development environment for MATLAB-style code with automatic GPU acceleration and cloud storage.",
        },
      },
      {
        title: "Versioning & History",
        slug: ["versioning"],
        file: "docs/VERSIONING.md",
        seo: {
          description: "How RunMat's automatic versioning and snapshots work.",
          keywords: ["version control", "git export", "snapshots", "history", "RunMat versioning"],
          ogTitle: "Versioning & History",
          ogDescription: "RunMat versions your work automatically. Learn about snapshots, history, and git export.",
        },
      },
      {
        title: "Collaboration & Teams",
        slug: ["collaboration"],
        file: "docs/COLLABORATION.md",
        seo: {
          description: "How to collaborate with teams in RunMat: organizations, projects, roles, real-time sync, and enterprise SSO.",
          keywords: ["collaboration", "teams", "organizations", "sharing", "real-time", "SSO", "RunMat Cloud"],
          ogTitle: "Collaboration & Teams",
          ogDescription: "Share projects, manage teams, and collaborate in real time with RunMat Cloud.",
        },
      },
      {
        title: "Design Philosophy",
        slug: ["design-philosophy"],
        file: "docs/DESIGN_PHILOSOPHY.md",
        seo: {
          description: "RunMat's slim core, package-first philosophy, and why we don't chase 100% MATLAB parity.",
          keywords: ["design philosophy", "slim core", "packages", "MATLAB alternative"],
          ogTitle: "RunMat Design Philosophy",
          ogDescription: "Why RunMat focuses on a minimal core and a powerful package system.",
        },
      },
      {
        title: "Correctness & Trust",
        slug: ["correctness"],
        file: "docs/CORRECTNESS.md",
      },
      { title: "How It Works", externalHref: "/docs/how-it-works" },
      { title: "Roadmap", slug: ["roadmap"], file: "docs/ROADMAP.md", seo: { description: "RunMat's development roadmap and progress.", keywords: ["roadmap", "development", "progress"] } },
    ],
  },
  {
    title: "Using RunMat",
    children: [
      { title: "Builtin Function Reference", externalHref: "/docs/matlab-function-reference" },
      { title: "Library Reference", slug: ["library"], file: "docs/LIBRARY.md", seo: { description: "How RunMat's built-in MATLAB functions are implemented and organized.", keywords: ["standard library", "builtins", "MATLAB functions"] } },
      { title: "CLI Reference", slug: ["cli"], file: "docs/CLI.md", seo: { description: "RunMat CLI commands, flags, environment variables, and examples.", keywords: ["RunMat CLI", "command line", "flags"] } },
      { title: "Configuration", slug: ["configuration"], file: "docs/CONFIG.md", seo: { description: "Configure RunMat: files, environment overrides, and precedence.", keywords: ["RunMat config", "configuration", "YAML", "TOML", "JSON"] } },
      {
        title: "Plotting",
        slug: ["plotting"],
        file: "docs/PLOTTING.md",
        seo: { description: "How RunMat plotting works: MATLAB-familiar API, GPU-first rendering, drawnow/pause semantics.", keywords: ["plotting", "GPU plotting", "MATLAB plot", "3D visualization", "drawnow"] },
        children: [
          {
            title: "Plotting in RunMat",
            slug: ["plotting", "plotting-in-runmat"],
            file: "docs/plotting/PLOTTING_IN_RUNMAT.md",
            seo: {
              description: "Building figures in RunMat: axes, plot families, subplot workflows, and refining figures from first command to finished output.",
              keywords: ["RunMat plotting", "figure", "axes", "subplot", "MATLAB plot workflow"],
              ogTitle: "Plotting in RunMat",
              ogDescription: "How to build, organize, and refine figures using RunMat's MATLAB-compatible plotting system.",
            },
          },
          {
            title: "Choosing the Right Plot Type",
            slug: ["plotting", "choosing-the-right-plot-type"],
            file: "docs/plotting/CHOOSING_THE_RIGHT_PLOT_TYPE.md",
            seo: {
              description: "When to use plot vs scatter, surf vs mesh, imagesc vs contourf, and how to match plot families to the structure of your data.",
              keywords: ["plot types", "scatter", "surf", "contour", "bar", "histogram", "MATLAB chart types"],
              ogTitle: "Choosing the Right Plot Type",
              ogDescription: "Pick the right plot family for your data: continuity, categories, fields, distributions, and more.",
            },
          },
          {
            title: "Styling Plots and Axes",
            slug: ["plotting", "styling-plots-and-axes"],
            file: "docs/plotting/STYLING_PLOTS_AND_AXES.md",
            seo: {
              description: "Labels, limits, legends, colormaps, grids, and view configuration for readable RunMat figures.",
              keywords: ["plot styling", "labels", "legend", "colormap", "axis limits", "MATLAB figure styling"],
              ogTitle: "Styling Plots and Axes",
              ogDescription: "Make figures readable with labels, legends, colormaps, and coordinated styling across plot objects and axes.",
            },
          },
          {
            title: "Graphics Handles",
            slug: ["plotting", "graphics-handles"],
            file: "docs/plotting/GRAPHICS_HANDLES.md",
            seo: {
              description: "How RunMat's graphics object model works: figures, axes, plot objects, handles, get/set, and subplot state.",
              keywords: ["graphics handles", "get", "set", "gcf", "gca", "figure handle", "MATLAB graphics objects"],
              ogTitle: "Graphics Handles and Plot Objects",
              ogDescription: "Inspect and update plot objects with handles. Understand figures, axes, legends, and the stateful graphics system.",
            },
          },
          {
            title: "Plot Replay and Export",
            slug: ["plotting", "plot-replay-and-export"],
            file: "docs/plotting/PLOT_REPLAY_AND_EXPORT.md",
            seo: {
              description: "How RunMat figures persist as scene state, how replay differs from recomputation, and how to export figures.",
              keywords: ["plot export", "figure replay", "scene state", "save figure", "MATLAB figure export"],
              ogTitle: "Plot Replay and Export",
              ogDescription: "Persist, replay, and export RunMat figures. Understand the difference between scene state and rendered output.",
            },
          },
        ],
      },
    ],
  },
  {
    title: "Language",
    children: [
      {
        title: "MATLAB Compatibility",
        slug: ["compatibility"],
        file: "docs/COMPATIBILITY.md",
        seo: {
          description: "RunMat MATLAB compatibility: language coverage, 400+ built-in functions, plotting, toolbox status, and known limitations.",
          keywords: ["MATLAB compatibility", "MATLAB alternative", "RunMat compatibility", "Octave alternative", "MATLAB migration"],
          ogTitle: "MATLAB Compatibility",
          ogDescription: "How compatible is RunMat with MATLAB? Language coverage, function reference, toolbox status, and GPU acceleration.",
        },
      },
      { title: "Compatibility Modes", slug: ["language"], file: "docs/LANGUAGE.md", seo: { description: "RunMat compatibility modes: runmat, matlab, and strict — controlling command syntax and error namespaces.", keywords: ["compatibility modes", "MATLAB syntax", "strict mode", "command syntax"] } },
      { title: "Language Coverage", slug: ["language-coverage"], file: "docs/LANGUAGE_COVERAGE.md", seo: { description: "MATLAB language feature compatibility in RunMat with an Octave comparison.", keywords: ["MATLAB compatibility", "Octave compatibility", "language coverage", "MATLAB alternative", "Octave alternative"] } },
    ],
  },
  {
    title: "Architecture & Internals",
    children: [
      {
        title: "High-Level Architecture",
        slug: ["architecture"],
        file: "docs/ARCHITECTURE.md",
        seo: {
          description: "RunMat architecture overview: tiered execution (VM → Turbine), generational GC, runtime, plotting, and snapshot startup.",
          keywords: ["RunMat", "MATLAB alternative", "Octave alternative", "JIT", "interpreter", "GC", "scientific computing"],
          ogTitle: "RunMat Architecture",
          ogDescription: "Deep dive into RunMat's V8-inspired execution model, GC, runtime, and plotting subsystems.",
        },
      },
      {
        title: "Large Dataset Persistence",
        slug: ["large-dataset-persistence"],
        file: "docs/LARGE_DATASET_PERSISTENCE.md",
        seo: {
          description: "How RunMat stores and operates on large numerical datasets using chunked, content-addressed objects with the data.* API.",
          keywords: ["large datasets", "chunked storage", "content-addressed", "data API", "multi-terabyte arrays", "scientific data", "RunMat data"],
          ogTitle: "Large Dataset Persistence",
          ogDescription: "Read and write subregions of multi-terabyte arrays without loading the full file. Same API on local and cloud projects.",
        },
      },
      {
        title: "Filesystem",
        slug: ["filesystem"],
        file: "docs/FILESYSTEM.md",
        seo: {
          description: "RunMat filesystem strategy across CLI, browser, and desktop environments.",
          keywords: ["filesystem", "storage", "sandbox", "WASM", "runtime"],
          ogTitle: "RunMat Filesystem Strategy",
          ogDescription: "How RunMat unifies filesystem access across native and browser runtimes.",
        },
      },
      // Lexer entry removed pending compiler docs rewrite; crate README was deleted in fbd1d97f (lexer crate split).
      { title: "Parser", slug: ["internals", "parser"], file: "crates/runmat-parser/README.md", seo: { description: "Precedence-based parser for MATLAB/Octave with statements, OOP, and command-form.", keywords: ["parser", "AST", "MATLAB", "Octave"] } },
      // HIR entry removed pending compiler docs rewrite; crate README was deleted during the HIR refactor.
      // /docs/internals/hir redirects to /docs/architecture in next.config.ts.
      // Additional VM internals pages can be added here when dedicated docs are promoted.
      { title: "Garbage Collector", slug: ["internals", "garbage-collector"], file: "crates/runmat-gc/README.md", seo: { description: "Generational mark-and-sweep GC with handles, barriers, and promotion.", keywords: ["garbage collector", "GC", "generational", "write barrier"] } },
    ],
  },
  {
    title: "Accelerate (GPU)",
    children: [
      { title: "Introduction to RunMat Fusion", slug: ["accelerate", "fusion-intro"], file: "docs/INTRODUCTION_TO_RUNMAT_GPU.md", seo: { description: "How RunMat manages GPU data residency: keeping arrays on device, minimizing transfers.", keywords: ["GPU", "residency", "data residency", "device memory"] } },
      { title: "GPU Residency & Precision", slug: ["accelerate", "gpu-behavior"], file: "docs/GPU_BEHAVIOR_NOTES.md", seo: { description: "GPU residency rules and precision guarantees: when data moves to/from device.", keywords: ["GPU residency", "precision", "f32", "f64", "device memory"] } },
      {
        title: "Fusion Guide",
        externalHref: "/docs/fusion-guide",
        hideChildrenInNav: true,
        children: [
          { title: "Elementwise Chains", slug: ["fusion", "elementwise"], file: "docs/fusion/ELEMENTWISE.md", seo: { description: "How RunMat fuses straight-line arithmetic and transcendental expressions into a single GPU kernel, reducing memory bandwidth and eliminating repeated reads/writes.", keywords: ["elementwise fusion", "GPU kernel", "arithmetic fusion", "transcendental", "operator fusion"] } },
          { title: "Reductions", slug: ["fusion", "reduction"], file: "docs/fusion/REDUCTION.md", seo: { description: "How RunMat keeps column and row reductions like sum and mean on the GPU, avoiding CPU round-trips for tall-slice workloads.", keywords: ["reduction fusion", "GPU reduction", "sum", "mean", "column reduction"] } },
          { title: "Matmul Epilogues", slug: ["fusion", "matmul-epilogue"], file: "docs/fusion/MATMUL_EPILOGUE.md", seo: { description: "How RunMat folds scalar and vector epilogue operations after matrix multiplication into the same GPU call, keeping affine transforms and normalisation on-device.", keywords: ["matmul epilogue", "matrix multiplication fusion", "GPU matmul", "affine transform", "epilogue fusion"] } },
          { title: "Centered Gram / Covariance", slug: ["fusion", "centered-gram"], file: "docs/fusion/CENTERED_GRAM.md", seo: { description: "How RunMat fuses mean-centering, transpose multiplication, and scalar division into a single GPU covariance or Gram-matrix computation.", keywords: ["covariance fusion", "Gram matrix", "mean-centering", "GPU covariance", "PCA"] } },
          { title: "Power-Step Normalisation", slug: ["fusion", "power-step-normalize"], file: "docs/fusion/POWER_STEP_NORMALIZE.md", seo: { description: "How RunMat fuses matrix multiplication with L2-style renormalization for iterative solvers like power iteration and Krylov updates.", keywords: ["power iteration fusion", "normalisation fusion", "Krylov", "iterative solver", "GPU renormalization"] } },
          { title: "Explained Variance", slug: ["fusion", "explained-variance"], file: "docs/fusion/EXPLAINED_VARIANCE.md", seo: { description: "How RunMat accelerates diag(Q' * G * Q) diagnostics on the GPU, measuring how much energy an orthogonal basis captures against a covariance matrix.", keywords: ["explained variance", "GPU diagnostics", "eigenvalue", "PCA variance", "covariance diagnostics"] } },
          { title: "Image Normalisation", slug: ["fusion", "image-normalize"], file: "docs/fusion/IMAGE_NORMALIZE.md", seo: { description: "How RunMat fuses tensor whitening, gain/bias application, and optional gamma correction into a single GPU dispatch for imaging and sensor pipelines.", keywords: ["image normalisation", "tensor whitening", "GPU normalisation", "gain bias", "gamma correction"] } },
        ],
      },
    ],
  },
  {
    title: "Meta",
    children: [
      {
        title: "Changelog",
        slug: ["changelog"],
        file: "docs/CHANGELOG.md",
        seo: {
          description: "What's new across the RunMat runtime, cloud, and sandbox — release notes, bug fixes, and new features.",
          keywords: ["changelog", "release notes", "what's new", "RunMat updates", "version history"],
          ogTitle: "RunMat Changelog",
          ogDescription: "Track every update to the RunMat runtime, cloud, and sandbox.",
        },
      },
      { title: "License", slug: ["license"], file: "LICENSE.md", seo: { description: "RunMat software license." } },
      { title: "Terms and Conditions", slug: ["terms"], file: "docs/TERMS.md", seo: { description: "Dystr Terms and Conditions governing the use of the RunMat platform and related services.", keywords: ["terms and conditions", "terms of service", "legal", "RunMat", "Dystr"] } },
      { title: "Telemetry", slug: ["telemetry"], file: "docs/TELEMETRY.md", seo: { description: "RunMat telemetry: what information is collected and how it is used." } },
      { title: "Contributing", slug: ["contributing"], file: "docs/CONTRIBUTING.md", seo: { description: "How to contribute to RunMat: branch workflow, PR scope, and coding style." } },
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

