use super::{is_descendant, not_found, MemoryEntry, MemoryTree};
use std::io::{self, ErrorKind};
use std::path::Path;

/// Caller holds the tree lock throughout validation and publication.
pub(super) fn execute(tree: &mut MemoryTree, from: &Path, to: &Path) -> io::Result<()> {
    let source = tree.entries.get(from).ok_or_else(|| not_found(from))?;
    if from == to {
        return Ok(());
    }
    if to == Path::new("/") || is_descendant(from, to) {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "invalid rename destination",
        ));
    }
    reject_readonly(source)?;
    let parent = to
        .parent()
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidInput, "destination has no parent"))?;
    match tree.entries.get(parent) {
        Some(MemoryEntry::Directory) => {}
        Some(_) => {
            return Err(io::Error::new(
                ErrorKind::NotADirectory,
                "destination parent is not a directory",
            ))
        }
        None => return Err(not_found(parent)),
    }
    if let Some(destination) = tree.entries.get(to) {
        reject_readonly(destination)?;
        match (source, destination) {
            (MemoryEntry::Directory, MemoryEntry::Directory) => {
                if tree.entries.keys().any(|path| is_descendant(to, path)) {
                    return Err(io::Error::new(
                        ErrorKind::DirectoryNotEmpty,
                        "destination directory is not empty",
                    ));
                }
            }
            (MemoryEntry::File { .. }, MemoryEntry::File { .. }) => {}
            _ => {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    "source and destination types differ",
                ))
            }
        }
    }
    // Collect every change before modifying the map. No await or fallible
    // filesystem operation occurs between removal and publication.
    let moves = tree
        .entries
        .iter()
        .filter(|(path, _)| path.as_path() == from || is_descendant(from, path))
        .map(|(path, entry)| {
            let suffix = path.strip_prefix(from).expect("selected descendant");
            (path.clone(), to.join(suffix), entry.clone())
        })
        .collect::<Vec<_>>();
    for (old, _, _) in &moves {
        tree.entries.remove(old);
    }
    for (_, new, entry) in moves {
        tree.entries.insert(new, entry);
    }
    Ok(())
}

fn reject_readonly(entry: &MemoryEntry) -> io::Result<()> {
    if matches!(entry, MemoryEntry::File { readonly: true, .. }) {
        Err(io::Error::new(
            ErrorKind::PermissionDenied,
            "rename path is readonly",
        ))
    } else {
        Ok(())
    }
}
