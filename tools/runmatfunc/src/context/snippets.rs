use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::Result;

use crate::app::config::AppConfig;
use crate::builtin::metadata::BuiltinRecord;
use crate::context::reference::REFERENCE_FILE_PATHS;

/// Return source paths relevant for the builtin (implementation, docs, tests).
pub fn source_paths(record: &BuiltinRecord, config: &AppConfig) -> Result<Vec<PathBuf>> {
    let mut paths: BTreeSet<PathBuf> = REFERENCE_FILE_PATHS.iter().map(PathBuf::from).collect();

    if let Some(plan) = config.generation_plan_path() {
        paths.insert(plan.to_path_buf());
    }
    if let Some(doc) = config.fusion_design_doc() {
        paths.insert(doc.to_path_buf());
    }

    if let Some(candidate) = expected_builtin_module(record) {
        paths.insert(candidate);
    }

    Ok(paths.into_iter().collect())
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
