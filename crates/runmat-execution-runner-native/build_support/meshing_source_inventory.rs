//! File and Cargo-watch visibility for an owning dependency package. Another
//! package's code is inventoried through the dependency graph, never by nesting.
use std::{
    collections::BTreeSet,
    fs, io,
    path::{Path, PathBuf},
};

pub struct SourceInventory {
    pub files: Vec<PathBuf>,
    pub watches: Vec<PathBuf>,
}

pub fn collect(
    package_root: &Path,
    package_roots: &BTreeSet<PathBuf>,
) -> io::Result<SourceInventory> {
    let mut inventory = SourceInventory {
        files: Vec::new(),
        watches: Vec::new(),
    };
    visit(package_root, package_root, package_roots, &mut inventory)?;
    Ok(inventory)
}

// The return value says whether Cargo can recursively watch this entire subtree
// without including excluded output or another package. Files are always watched.
fn visit(
    package_root: &Path,
    path: &Path,
    package_roots: &BTreeSet<PathBuf>,
    inventory: &mut SourceInventory,
) -> io::Result<bool> {
    let mut complete_subtree = path != package_root;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let name = entry.file_name();
        let package_output = path == package_root
            && matches!(name.to_str(), Some("target" | "artifacts" | ".artifacts"));
        if package_output || matches!(name.to_str(), Some(".git" | "node_modules" | ".DS_Store")) {
            complete_subtree = false;
            continue;
        }
        let kind = entry.file_type()?;
        // Omission is not source identity. Linked or nonregular inputs require
        // richer closure evidence; until then they cannot authorize cached work.
        if kind.is_symlink() || (!kind.is_dir() && !kind.is_file()) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("nonregular source entry {}", entry.path().display()),
            ));
        }
        if kind.is_dir() {
            if package_roots.contains(&entry.path()) {
                complete_subtree = false;
            } else {
                complete_subtree &= visit(package_root, &entry.path(), package_roots, inventory)?;
            }
        } else {
            // Code generation may consume JSON, tables or other non-Rust inputs.
            // Preserve every regular package input rather than guessing by suffix.
            inventory.files.push(entry.path());
            inventory.watches.push(entry.path());
        }
    }
    if complete_subtree {
        inventory.watches.push(path.to_path_buf());
    }
    Ok(complete_subtree)
}
