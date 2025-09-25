use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::builtin::metadata::BuiltinRecord;

/// Return source paths relevant for the builtin (implementation, tests, docs).
/// For now we simply point to the runtime source tree and expect a standard layout.
pub fn source_paths(record: &BuiltinRecord) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();

    // Implementation path (best-effort guess against new builtins layout; falls back to legacy module).
    let candidate = format!("crates/runmat-runtime/src/builtins/**/*{}*.rs", record.name);
    paths.push(Path::new("crates/runmat-runtime/src/lib.rs").to_path_buf());
    paths.push(PathBuf::from(candidate));

    Ok(paths)
}
