use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::Result;
use glob::glob;

use crate::builtin::metadata::BuiltinRecord;

/// Return source paths relevant for the builtin (implementation, docs, tests).
pub fn source_paths(record: &BuiltinRecord) -> Result<Vec<PathBuf>> {
    let mut unique: HashSet<PathBuf> = HashSet::new();

    unique.insert(PathBuf::from("crates/runmat-runtime/src/lib.rs"));

    let lowered = record.name.to_ascii_lowercase();
    add_glob(
        &mut unique,
        format!("crates/runmat-runtime/src/builtins/**/*{lowered}*.rs"),
    );
    add_glob(
        &mut unique,
        format!("crates/runmat-runtime/src/**/*{lowered}*.rs"),
    );

    add_if_exists(&mut unique, "crates/runmat-runtime/src/mathematics.rs");
    add_if_exists(&mut unique, "crates/runmat-runtime/src/arrays.rs");
    add_if_exists(&mut unique, "crates/runmat-runtime/src/elementwise.rs");
    add_if_exists(&mut unique, "crates/runmat-runtime/src/accel.rs");

    add_if_exists(&mut unique, "docs/fusion-runtime-design.md");

    let mut paths: Vec<PathBuf> = unique.into_iter().collect();
    paths.sort();
    Ok(paths)
}

fn add_glob(set: &mut HashSet<PathBuf>, pattern: String) {
    if let Ok(entries) = glob(&pattern) {
        for entry in entries.flatten() {
            if entry.is_file() {
                set.insert(entry);
            }
        }
    }
}

fn add_if_exists(set: &mut HashSet<PathBuf>, path: &str) {
    let p = Path::new(path);
    if p.exists() {
        set.insert(p.to_path_buf());
    }
}
