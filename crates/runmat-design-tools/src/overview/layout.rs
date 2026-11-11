//! The project's layout, for `design.overview`: its top-level folders, whether they follow the
//! standard layout, and its library `.circuit` files.
//!
//! The standard layout is `mechanical/<subsystem>/`, `electrical/<board>/` (each `.circuit`
//! beside its `.circuit.layout`), `electrical/library/` (`footprints/`, `parts/`),
//! `simulation/` and `drawings/`. A project with folders of its own keeps them; the overview
//! says which, so an agent adds files where the project already puts them.

use crate::Files;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// The standard layout's top-level folders.
pub const STANDARD_FOLDERS: &[&str] = &["mechanical", "electrical", "simulation", "drawings"];

/// Where footprint libraries go in the standard layout, under the project root.
pub const FOOTPRINT_LIBRARY: &str = "electrical/library/footprints";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProjectLayout {
    /// The project's root folder: where its `runmat.toml` is, else the folder above its
    /// `mechanical/` or `electrical/` folder, else the folder the supplied files share.
    pub root: String,
    /// The top-level folders the supplied files are in, by name.
    pub folders: Vec<String>,
    /// Whether every top-level folder is one of the standard layout's.
    pub standard: bool,
    /// The library `.circuit` files (no `circuit` block) among the supplied files.
    pub libraries: Vec<String>,
    /// Where new files go, in one line.
    pub guidance: String,
}

fn parent(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(d, _)| d)
}

fn root_of(path: &str, files: &Files) -> String {
    if let Some(toml) = files
        .keys()
        .filter(|p| p.ends_with("/runmat.toml") && path.starts_with(&format!("{}/", parent(p))))
        .min_by_key(|p| p.len())
    {
        return parent(toml).to_owned();
    }
    let segments: Vec<&str> = path.split('/').collect();
    if let Some(i) = segments.iter().position(|s| STANDARD_FOLDERS.contains(s)) {
        return segments[..i].join("/");
    }
    let mut common = parent(path).to_owned();
    for p in files.keys() {
        while !common.is_empty() && !p.starts_with(&format!("{common}/")) {
            common = parent(&common).to_owned();
        }
    }
    common
}

/// The layout of the project `path` is in, from the supplied files.
pub fn layout(path: &str, files: &Files) -> ProjectLayout {
    let root = root_of(path, files);
    let prefix = format!("{root}/");
    let mut folders = BTreeSet::new();
    let mut libraries = Vec::new();
    for p in files.keys().map(String::as_str).chain([path]) {
        let Some(rest) = p.strip_prefix(&prefix) else { continue };
        if let Some((top, _)) = rest.split_once('/') {
            if !top.starts_with('.') && !top.starts_with('+') {
                folders.insert(top.to_owned());
            }
        }
        if p.to_ascii_lowercase().ends_with(".circuit") && !libraries.iter().any(|l| l == p) {
            let text = files.get(p).map(String::as_str).unwrap_or("");
            if runmat_circuit::syntax::parse_circuit(p, text).design.is_some_and(|d| d.library) {
                libraries.push(p.to_owned());
            }
        }
    }
    let folders: Vec<String> = folders.into_iter().collect();
    let standard = folders.iter().all(|f| STANDARD_FOLDERS.contains(&f.as_str()));
    let guidance = if standard {
        format!(
            "Standard layout: mechanical/<subsystem>/, electrical/<board>/ (each .circuit beside its .circuit.layout), electrical/library/ (footprints/, parts/), simulation/, drawings/. Custom footprints go in {FOOTPRINT_LIBRARY}/<name>.circuit."
        )
    } else {
        format!("This project keeps its own folders ({}); add new files where it already puts files of that kind.", folders.join(", "))
    };
    ProjectLayout { root, folders, standard, libraries, guidance }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_standard_project_reports_its_folders_and_libraries() {
        let files: Files = [
            ("/w/drone/electrical/fc/fc.circuit", "runmat 0.7\nstd 1\n\ncircuit fc {\n}\n"),
            ("/w/drone/electrical/library/parts.circuit", "runmat 0.7\nstd 1\n"),
            ("/w/drone/mechanical/frame/frame.part", ""),
        ]
        .into_iter()
        .map(|(p, t)| (p.to_string(), t.to_string()))
        .collect();
        let l = layout("/w/drone/electrical/fc/fc.circuit", &files);
        assert_eq!(l.root, "/w/drone");
        assert_eq!(l.folders, ["electrical", "mechanical"]);
        assert!(l.standard);
        assert_eq!(l.libraries, ["/w/drone/electrical/library/parts.circuit"]);
        assert!(l.guidance.contains("electrical/library/footprints/<name>.circuit"));
        let mut own = files.clone();
        own.insert("/w/drone/hw/board.circuit".into(), String::new());
        own.insert("/w/drone/runmat.toml".into(), String::new());
        let l = layout("/w/drone/hw/board.circuit", &own);
        assert!(!l.standard);
        assert!(l.guidance.contains("electrical, hw, mechanical"), "{}", l.guidance);
    }
}
