//! Release: a design file's text saved as its next released revision in the project's release
//! store ([`runmat_cad::refs::release_store`]), as a change to the store file (created on the
//! first release). Assemblies pin components to these revisions.

use crate::edit::model;
use crate::runtime::Files;
use crate::xref_repairs::FileEdit;
use runmat_cad::ids::RevisionId;
use runmat_cad::refs::release_store::{store_path, ReleaseStore};
use runmat_cad::syntax::edit::{EditFailure, SourceChange};
use runmat_design_syntax::edit::TextEdit;
use runmat_types::Span;

/// Releases `file` (a workspace path) at `text`: the store's change and the new revision.
pub fn release(file: &str, text: &str, files: &Files) -> Result<(FileEdit, RevisionId), EditFailure> {
    let at = store_path(file);
    let old = files.get(&at).cloned().unwrap_or_default();
    let mut store = if old.is_empty() {
        ReleaseStore::new(file)
    } else {
        ReleaseStore::parse(&old).map_err(model)?
    };
    let revision = store.release(text).map_err(model)?;
    let new = store.to_text();
    let change = SourceChange {
        edits: vec![TextEdit::replace(
            Span {
                start: 0,
                end: old.len(),
            },
            new.clone(),
        )],
        source: new,
        reprinted: true,
    };
    Ok(((at, old, change), revision))
}

/// The latest released revision of `file`, or why there is none.
pub fn latest(file: &str, files: &Files) -> Result<RevisionId, EditFailure> {
    let name = file.rsplit('/').next().unwrap_or(file);
    crate::runtime::release_store(file, files)
        .and_then(|s| s.latest().map(|r| r.revision.clone()))
        .ok_or_else(|| model(format!("{name} has no released revision: release it first")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_release_creates_the_store_and_the_next_extends_it() {
        let mut files = Files::new();
        let ((path, old, change), r1) = release("/p/bracket.part", "one", &files).unwrap();
        assert_eq!(path, "/p/.runmat/releases/bracket.part.json");
        assert!(old.is_empty());
        assert_eq!(r1.0, "R1");
        files.insert(path.clone(), change.source);
        assert_eq!(latest("/p/bracket.part", &files).unwrap().0, "R1");
        let ((_, old, change), r2) = release("/p/bracket.part", "two", &files).unwrap();
        assert!(!old.is_empty());
        assert_eq!(r2.0, "R2");
        files.insert(path, change.source);
        assert!(release("/p/bracket.part", "two", &files).is_err());
        assert!(latest("/p/other.part", &files)
            .unwrap_err()
            .to_string()
            .contains("other.part has no released revision"));
    }
}
