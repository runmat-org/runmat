use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Read file contents into a string.
pub fn read_file(path: &Path) -> Result<String> {
    fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))
}

/// Write string contents to a path, creating parent directories as needed.
pub fn write_file(path: &Path, contents: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating directory {}", parent.display()))?;
    }
    fs::write(path, contents).with_context(|| format!("writing {}", path.display()))
}

/// Convenience helper to check for file existence.
pub fn path_exists(path: &Path) -> bool {
    path.exists()
}

/// Resolve a relative path against a workspace root.
pub fn resolve(root: &Path, relative: &str) -> PathBuf {
    root.join(relative)
}
