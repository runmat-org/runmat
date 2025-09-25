use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::Result;
use glob::{glob, Pattern};
use tracing::debug;

use crate::app::config::AppConfig;
use crate::builtin::metadata::BuiltinRecord;

/// Return source paths relevant for the builtin (implementation, docs, tests).
pub fn source_paths(record: &BuiltinRecord, config: &AppConfig) -> Result<Vec<PathBuf>> {
    let mut unique: HashSet<PathBuf> = HashSet::new();
    let excludes = compile_patterns(&config.snippet_excludes);

    add_candidate(
        &mut unique,
        PathBuf::from("crates/runmat-runtime/src/lib.rs"),
        &excludes,
    );

    if let Some(candidate) = expected_builtin_module(record) {
        add_candidate(&mut unique, candidate, &excludes);
    }

    // Legacy fallback for yet-to-be migrated modules.
    let lowered = record.name.to_ascii_lowercase();
    add_glob_results(
        &mut unique,
        format!("crates/runmat-runtime/src/**/*{lowered}*.rs"),
        &excludes,
    );

    add_candidate(
        &mut unique,
        PathBuf::from("crates/runmat-runtime/src/mathematics.rs"),
        &excludes,
    );
    add_candidate(
        &mut unique,
        PathBuf::from("crates/runmat-runtime/src/arrays.rs"),
        &excludes,
    );
    add_candidate(
        &mut unique,
        PathBuf::from("crates/runmat-runtime/src/elementwise.rs"),
        &excludes,
    );
    add_candidate(
        &mut unique,
        PathBuf::from("crates/runmat-runtime/src/accel.rs"),
        &excludes,
    );

    if let Some(plan) = config.generation_plan_path() {
        add_candidate(&mut unique, plan.to_path_buf(), &excludes);
    }
    if let Some(doc) = config.fusion_design_doc() {
        add_candidate(&mut unique, doc.to_path_buf(), &excludes);
    }

    for pattern in &config.snippet_includes {
        add_glob_results(&mut unique, pattern.clone(), &excludes);
    }

    let mut paths: Vec<PathBuf> = unique.into_iter().collect();
    paths.sort();
    Ok(paths)
}

fn add_candidate(set: &mut HashSet<PathBuf>, path: PathBuf, excludes: &[Pattern]) {
    if should_exclude(&path, excludes) {
        return;
    }
    if path.exists() {
        set.insert(path);
    } else {
        debug!("snippet candidate {} missing; skipping", path.display());
    }
}

fn add_glob_results(set: &mut HashSet<PathBuf>, pattern: String, excludes: &[Pattern]) {
    match glob(&pattern) {
        Ok(entries) => {
            for entry in entries.flatten() {
                if entry.is_file() && !should_exclude(&entry, excludes) {
                    set.insert(entry);
                }
            }
        }
        Err(err) => debug!("invalid snippet glob '{}': {}", pattern, err),
    }
}

fn compile_patterns(patterns: &[String]) -> Vec<Pattern> {
    patterns
        .iter()
        .filter_map(|raw| match Pattern::new(raw) {
            Ok(pattern) => Some(pattern),
            Err(err) => {
                debug!("invalid snippet exclude pattern '{}': {}", raw, err);
                None
            }
        })
        .collect()
}

fn should_exclude(path: &Path, patterns: &[Pattern]) -> bool {
    patterns.iter().any(|pat| pat.matches_path(path))
}

fn expected_builtin_module(record: &BuiltinRecord) -> Option<PathBuf> {
    record.category.as_ref().map(|category| {
        let mut path = PathBuf::from("crates/runmat-runtime/src/builtins");
        for segment in category.split('/') {
            path.push(normalize_segment(segment));
        }
        path.push(format!("{}.rs", normalize_segment(&record.name)));
        path
    })
}

fn normalize_segment(segment: &str) -> String {
    segment.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}
