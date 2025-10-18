use std::fmt::Write;

use crate::app::config::AppConfig;
use crate::builtin::metadata::BuiltinRecord;
use crate::context::reference;

const IMPLEMENTATION_NOTES: &[&str] = &[
    "Create a dedicated builtin module (DOC_MD, GPU/Fusion specs, runtime registration, helper routines, tests).",
    "Mirror function semantics exactly to MATLAB, as this is a MATLAB compatible runtime; ",
    "raise MATLAB-compatible errors and keep GPU fallbacks in sync with host code.",
    "Share helpers via builtins/common rather than referencing legacy modules.",
    "Document GPU fallbacks and complete any provider hooks that are incomplete that are needed ",
    "for the function to work correctly and completely on the GPU.",
];

const TESTING_EXPECTATIONS: &[&str] = &[
    "cargo test -p runmat-runtime --lib -- <builtin>",
    "cargo test -p runmat-runtime --tests -- <builtin>",
    "Include doc example smoke tests (test_support::doc_examples)",
];

/// Build a textual prompt describing the builtin authoring task.
pub fn render_prompt(record: &BuiltinRecord, config: &AppConfig) -> String {
    let mut prompt = String::new();
    let _ = writeln!(prompt, "Builtin: {}", record.name);
    let default_model = config.default_model.as_deref().unwrap_or("gpt-5-codex");
    let _ = writeln!(prompt, "Preferred Codex model: {}", default_model);
    prompt.push('\n');

    for (label, body) in reference::reference_sections() {
        let _ = writeln!(prompt, "===== {label} =====");
        prompt.push_str(body);
        if !prompt.ends_with('\n') {
            prompt.push('\n');
        }
        prompt.push('\n');
    }

    prompt.push_str("===== Task =====\n");
    if let Some(category) = &record.category {
        let module_hint = format_module_path(category, &record.name);
        let _ = writeln!(
            prompt,
            "- Implement the `{}` builtin at `{}`.",
            record.name, module_hint
        );
    } else {
        let _ = writeln!(
            prompt,
            "- Implement the `{}` builtin and choose an appropriate category under `crates/runmat-runtime/src/builtins/`.",
            record.name
        );
    }
    prompt.push_str(
        "- Use the references above to follow the RunMat builtin template (DOC_MD, GPU/Fusion specs, tests).\n",
    );
    prompt.push_str(
        "- Ensure MATLAB-compatible semantics and document GPU fallbacks when provider hooks are incomplete.\n\n",
    );

    prompt.push_str("===== Implementation Notes =====\n");
    for note in IMPLEMENTATION_NOTES {
        let _ = writeln!(prompt, "- {}", note);
    }
    prompt.push('\n');

    prompt.push_str("===== Testing Expectations =====\n");
    for note in TESTING_EXPECTATIONS {
        let _ = writeln!(prompt, "- {}", note.replace("<builtin>", &record.name));
    }

    prompt
}

fn format_module_path(category: &str, name: &str) -> String {
    let mut segments: Vec<String> = category
        .split('/')
        .map(|segment| segment.trim().to_ascii_lowercase().replace([' ', '-'], "_"))
        .collect();
    let file = format!("{}.rs", name.trim().to_ascii_lowercase());
    segments.push(file);
    format!("crates/runmat-runtime/src/builtins/{}", segments.join("/"))
}
