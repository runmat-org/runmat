use std::fmt::Write;

use crate::app::config::AppConfig;
use crate::builtin::metadata::BuiltinRecord;

/// Build a textual prompt describing the builtin authoring task.
pub fn render_prompt(record: &BuiltinRecord, config: &AppConfig) -> String {
    let mut prompt = String::new();
    let _ = writeln!(prompt, "Builtin: {}", record.name);
    if let Some(summary) = &record.summary {
        let _ = writeln!(prompt, "Summary: {}", summary);
    }
    if let Some(category) = &record.category {
        let _ = writeln!(prompt, "Category: {}", category);
    }
    if !record.keywords.is_empty() {
        let _ = writeln!(prompt, "Keywords: {}", record.keywords.join(", "));
    }
    if !record.accel_tags.is_empty() {
        let _ = writeln!(prompt, "GPU Tags: {}", record.accel_tags.join(", "));
    }
    if let Some(model) = &config.default_model {
        let _ = writeln!(prompt, "Preferred Codex model: {}", model);
    }

    prompt.push_str("\nAuthoring Checklist:\n");
    prompt.push_str("1. Create a dedicated builtin module (DOC_MD, GPU/Fusion specs, runtime registration, helper routines, tests).\n");
    prompt.push_str("2. Mirror RunMat semantics exactly; raise MATLAB-compatible errors and keep GPU fallbacks in sync with host code.\n");
    prompt.push_str("3. Update DOC_MD YAML frontmatter with title/category/keywords/summary, GPU + fusion metadata, references, tested blocks, and any required features.\n");
    prompt.push_str("4. Register GPU + fusion specs via register_builtin_gpu_spec!/register_builtin_fusion_spec! with concise notes.\n");
    prompt.push_str("5. Cover scalars, broadcasting, dimension arguments, gpuArray residency, and doc examples in #[cfg(test)] including native-accel providers.\n");
    prompt.push_str("6. Keep the module self-contained; share helpers via builtins/common rather than referencing legacy monoliths.\n");

    prompt.push_str("\nDocs & References:\n");
    if let Some(path) = config.generation_plan_path() {
        let _ = writeln!(prompt, "- Blueprint: {}", path.display());
    } else {
        prompt.push_str("- Blueprint: generation-plan-2.md (ensure YAML + GPU guidance)\n");
    }
    if let Some(path) = config.fusion_design_doc() {
        let _ = writeln!(prompt, "- Fusion/GPU design: {}", path.display());
    }
    prompt.push_str(
        "- Keep docs/builtins.d.ts and docs/generated/builtins.json aligned via runmatfunc docs.\n",
    );

    prompt.push_str("\nImplementation Notes:\n");
    prompt.push_str("- Include TODO markers for any unsupported GPU hooks; document fallbacks.\n");
    prompt.push_str(
        "- Prefer helper methods from builtins/common for tensor conversions and GPU gather.\n",
    );
    prompt.push_str("- Avoid importing legacy aggregated modules once per-builtin file exists; update globs/snippets accordingly.\n");

    prompt.push_str("\nTesting Expectations:\n");
    prompt.push_str("- cargo test -p runmat-runtime --lib -- <builtin> for unit coverage.\n");
    prompt.push_str(
        "- cargo test -p runmat-runtime --tests -- <builtin> for integration/fusion harnesses.\n",
    );
    prompt.push_str("- Include doc example smoke tests (test_support::doc_examples).\n");

    prompt
}
