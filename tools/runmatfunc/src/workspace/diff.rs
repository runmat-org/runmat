use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{anyhow, Context, Result};
use similar::TextDiff;

/// Produce a unified diff with file headers.
pub fn unified_diff(old: &str, new: &str, old_label: &str, new_label: &str) -> Result<String> {
    let diff = TextDiff::from_lines(old, new);
    let unified = diff
        .unified_diff()
        .context_radius(3)
        .header(old_label, new_label)
        .to_string();
    Ok(unified)
}

/// Diff helper when you already have file contents available.
pub fn diff_contents(original: &str, updated: &str) -> Result<String> {
    unified_diff(original, updated, "original", "updated")
}

/// Run `git diff` against the provided paths and return the textual diff when present.
pub fn git_diff(paths: &[PathBuf]) -> Result<Option<String>> {
    if paths.is_empty() {
        return Ok(None);
    }

    let mut cmd = Command::new("git");
    cmd.arg("diff").arg("--");
    for path in paths {
        cmd.arg(path);
    }

    let output = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("executing git diff")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("git diff failed: {}", stderr.trim()));
    }

    let diff = String::from_utf8_lossy(&output.stdout).into_owned();
    if diff.trim().is_empty() {
        Ok(None)
    } else {
        Ok(Some(diff))
    }
}
