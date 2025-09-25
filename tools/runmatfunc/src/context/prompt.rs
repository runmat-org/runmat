use crate::builtin::metadata::BuiltinRecord;

/// Build a textual prompt describing the builtin authoring task.
pub fn render_prompt(record: &BuiltinRecord) -> String {
    let mut prompt = String::new();
    prompt.push_str(&format!("Builtin: {}\n", record.name));
    if let Some(summary) = &record.summary {
        prompt.push_str(&format!("Summary: {}\n", summary));
    }
    if let Some(category) = &record.category {
        prompt.push_str(&format!("Category: {}\n", category));
    }
    if !record.keywords.is_empty() {
        prompt.push_str(&format!("Keywords: {}\n", record.keywords.join(", ")));
    }
    if !record.accel_tags.is_empty() {
        prompt.push_str(&format!("GPU Tags: {}\n", record.accel_tags.join(", ")));
    }
    prompt.push_str("\nTask:\n");
    prompt.push_str("- Review existing implementation and documentation.\n");
    prompt.push_str("- Ensure documentation (DOC_MD) is exhaustive and includes GPU semantics.\n");
    prompt.push_str("- Verify tests cover scalar, tensor, GPU, and variadic cases.\n");
    prompt.push_str("- Maintain MATLAB-compatible behaviour while leveraging Accelerate/Fusion.\n");
    prompt
}
