use anyhow::Result;
use similar::{ChangeTag, TextDiff};

/// Produce a unified diff between two strings using +/- prefixes.
pub fn unified_diff(old: &str, new: &str) -> Result<String> {
    let diff = TextDiff::from_lines(old, new);
    let mut output = String::new();
    for change in diff.iter_all_changes() {
        let prefix = match change.tag() {
            ChangeTag::Delete => '-',
            ChangeTag::Insert => '+',
            ChangeTag::Equal => ' ',
        };
        output.push(prefix);
        output.push_str(change.value());
    }
    Ok(output)
}

/// Diff helper when you already have file contents available.
pub fn diff_contents(original: &str, updated: &str) -> Result<String> {
    unified_diff(original, updated)
}
