//! Tuned multi-pass Codex prompts for builtin generation, in execution order.

use crate::context::types::AuthoringContext;

#[derive(Copy, Clone)]
pub struct PassSpec {
    pub name: &'static str,
    pub build: fn(&AuthoringContext) -> String,
}

// 1) Packaging
pub const PACKAGING: PassSpec = PassSpec {
    name: "packaging",
    build: |_: &AuthoringContext| packaging_extra(),
};

// 2) WGPU provider integration
pub const WGPU: PassSpec = PassSpec {
    name: "wgpu",
    build: |_: &AuthoringContext| wgpu_extra(),
};

// 3) Completion (resolve placeholders and argument handling), scoped to builtin name
pub const COMPLETION: PassSpec = PassSpec {
    name: "completion",
    build: |ctx: &AuthoringContext| completion_extra(ctx),
};

// 4) Documentation polish
pub const DOCS: PassSpec = PassSpec {
    name: "docs",
    build: |_: &AuthoringContext| docs_extra(),
};

/// Ordered pass list used by headless and interactive flows.
pub static PASS_ORDER: &[PassSpec] = &[PACKAGING, WGPU, COMPLETION, DOCS];

fn packaging_extra() -> String {
    let mut s = String::new();
    s.push_str("Re-check the implementation against the Builtin Packaging template. ");
    s.push_str("Paste the entire template and ensure all required sections (DOC_MD, GPU/Fusion specs, tests, error semantics) are present and correct. ");
    s.push_str("Apply changes via apply_patch when necessary. Then stop.\n\n");
    s.push_str("Template reference: crates/runmat-runtime/BUILTIN_PACKAGING.md (already included above in references).\n");
    s
}

fn wgpu_extra() -> String {
    let mut s = String::new();
    s.push_str("Ensure the WGPU provider backend is fully implemented and wired for this builtin: ");
    s.push_str("(1) implement or extend provider hooks in crates/runmat-accelerate/src/backend/wgpu/provider_impl.rs matching GPU_SPEC provider_hooks; ");
    s.push_str("(2) add dispatch and shader wiring as needed under crates/runmat-accelerate/src/backend/wgpu/{dispatch,shaders,params,types}; ");
    s.push_str("(3) add #[cfg(feature=\\\"wgpu\\\")] tests in the builtin module verifying GPU parity (use provider registration helper); ");
    s.push_str("(4) if you add WGPU-only tests, set requires_feature: 'wgpu' in the DOC_MD frontmatter so runmatfunc can run with the correct Cargo features; ");
    s.push_str("(5) keep CPU semantics identical; (6) re-run tests until all pass. Apply changes via apply_patch.\n\n");
    s.push_str("References: crates/runmat-accelerate/src/backend/wgpu/provider_impl.rs, .../dispatch, .../shaders, and existing GPU tests (math/reduction/{sum,mean}.rs, math/trigonometry/sin.rs).\n");
    s
}

fn completion_extra(ctx: &AuthoringContext) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "Within the builtin '{}', eliminate any remaining placeholders and complete the implementation: ",
        ctx.builtin.name
    ));
    s.push_str("(1) search changed files and immediate dependencies for TODO, FIXME, unimplemented!, todo!(), panic(\\\"unimplemented\\\"), \\\"for later\\\", \\\"next revision\\\" and replace with working code or proper error paths that match MATLAB semantics; ");
    s.push_str("(2) ensure the full argument matrix supported by the spec is implemented (positional, keyword-like strings, size vectors, dtype/logical toggles, 'like' prototype, GPU residency cases); ");
    s.push_str("(3) add or extend unit + #[cfg(feature=\\\"wgpu\\\")] tests to cover all supported argument forms and edge cases (including error conditions); ");
    s.push_str("(4) ensure DOC_MD frontmatter 'tested' paths reflect unit/integration tests; (5) keep CPU/GPU parity. Apply changes via apply_patch, then stop.\n\n");
    s.push_str("References: crates/runmat-runtime/Library.md and crates/runmat-runtime/BUILTIN_PACKAGING.md; use sum, mean, sin, zeros/ones/rand as templates.\n");
    s
}

fn docs_extra() -> String {
    let mut s = String::new();
    s.push_str("Review and improve the DOC_MD for this builtin to match the tone, structure, and completeness of the best existing builtins. ");
    s.push_str("Ensure: (1) headlines/sections match style; (2) GPU Execution reflects provider hooks/fusion residency; (3) realistic, runnable Examples; (4) See Also with correct names. ");
    s.push_str("Headings are curated for SEO—do not change them. If anything is missing or inconsistent, update DOC_MD via apply_patch. Then stop.\n\n");
    s.push_str("Reference examples: math/reduction/{sum,mean}.rs, math/trigonometry/sin.rs, array/{zeros,ones,rand}.rs.\n");
    s
}