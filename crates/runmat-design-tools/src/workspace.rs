//! A design workspace read from disk, for command-line callers (`runmat-design`, `runmat check`,
//! `runmat test`): every design file and model library under a folder.

use crate::Files;
use std::path::Path;

/// The extensions read: design files, the circuit lock, STEP inputs, the model libraries
/// (`.lib`, `.cir`, `.sp`) that simulation sections include, the `.m` functions design values
/// call, and the SVG logos title blocks draw.
pub const DESIGN_EXTENSIONS: &[&str] = &[
    ".part",
    ".assembly",
    ".drawing",
    ".circuit",
    ".circuit.layout",
    ".circuit.lock",
    ".step",
    ".stp",
    ".lib",
    ".cir",
    ".sp",
    ".m",
    ".svg",
];

/// Every design file under `dir` (hidden folders, `target` and `node_modules` skipped), keyed by
/// its path as `dir` spells it.
pub fn read_workspace(dir: &Path) -> Files {
    let mut files = Files::new();
    walk(dir, &mut files, 0);
    files
}

fn walk(dir: &Path, files: &mut Files, depth: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || name == "target" || name == "node_modules" {
            continue;
        }
        if p.is_dir() && depth < 12 {
            walk(&p, files, depth + 1);
        } else if DESIGN_EXTENSIONS.iter().any(|x| name.ends_with(x)) {
            if let Ok(text) = std::fs::read_to_string(&p) {
                files.insert(p.to_string_lossy().into_owned(), text);
            }
        }
    }
}

/// A `.circuit` or `.circuit.layout` path: the files the circuit checks run on.
pub fn is_circuit_file(path: &Path) -> bool {
    let name = path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default();
    name.ends_with(".circuit") || name.ends_with(".circuit.layout")
}
