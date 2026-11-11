//! A design workspace read from disk, for command-line callers (`runmat-design`, `runmat check`,
//! `runmat test`): every design file and model library under a folder.

use crate::Files;
use std::io;
use std::path::{Component, Path, PathBuf};

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

/// Every design file under `dir`, keyed by [`source_path`]. Hidden folders,
/// `target` and `node_modules` are skipped.
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
                files.insert(source_path(&p), text);
            }
        }
    }
}

/// Serialize a native path into the design compilers' normalized, slash-separated namespace.
pub fn source_path(path: &Path) -> String {
    #[cfg(windows)]
    {
        runmat_cad::modules::normalize(&path.to_string_lossy().replace('\\', "/"))
    }
    #[cfg(not(windows))]
    {
        path.to_string_lossy().into_owned()
    }
}

/// Resolve an existing workspace file's logical path for native settings publication.
pub fn physical_path(root: &Path, source: &str) -> io::Result<PathBuf> {
    let root_source = source_path(root);
    let relative = Path::new(source)
        .strip_prefix(Path::new(&root_source))
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "design path is outside its workspace",
            )
        })?;
    output_path(root, relative)
}

/// Keep exported files beneath the caller's output folder.
pub fn output_path(root: &Path, relative: &Path) -> io::Result<PathBuf> {
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "design path must name a file within its workspace",
        ));
    }
    Ok(root.join(relative))
}

/// A `.circuit` or `.circuit.layout` path: the files the circuit checks run on.
pub fn is_circuit_file(path: &Path) -> bool {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy())
        .unwrap_or_default();
    name.ends_with(".circuit") || name.ends_with(".circuit.layout")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_workspace_paths_round_trip_and_confine_settings_writes() {
        #[cfg(windows)]
        let roots = [r"\\?\C:\project", r"\\?\UNC\server\share\project"];
        #[cfg(not(windows))]
        let roots = ["/project"];
        for root in roots {
            let root = Path::new(root);
            let file = root.join("models/board.circuit.layout");
            let source = source_path(&file);
            assert_eq!(physical_path(root, &source).unwrap(), file);
            assert!(physical_path(
                root,
                &format!("{}/../escape.circuit.layout", source_path(root))
            )
            .is_err());
            assert_eq!(
                output_path(root, Path::new("gerber/layer.gbr")).unwrap(),
                root.join("gerber/layer.gbr")
            );
            for path in ["/escape.step", "../escape.step"] {
                assert!(output_path(root, Path::new(path)).is_err());
            }
            #[cfg(windows)]
            assert!(output_path(root, Path::new(r"C:\escape.step")).is_err());
            assert!(physical_path(
                root,
                &format!("{}-other/board.circuit.layout", source_path(root))
            )
            .is_err());
        }
    }
}
