//! `assembly.release` and `assembly.update-pins`: released revisions of the components' files.
//!
//! - Release saves the selected components' files (or, with none selected, the assembly itself)
//!   as their next released revision in the project's release store ([`crate::edit::release`]).
//! - Update pins sets the chosen components (all of them when none is chosen) to follow their
//!   file live (`latest`), or pins each to its file's latest released revision (`released`).

use super::{find, model, selected_names, text_value, Compiled};
use crate::edit::release;
use crate::edit::EditRequest;
use crate::runtime::{resolve_path, Files};
use crate::xref_repairs::FileEdit;
use runmat_cad::assembly::model::AssemblyDocument;
use runmat_cad::refs::XrefContext;
use runmat_cad::syntax::edit::EditFailure;

/// The components a command names, or every component when it names none.
fn chosen_or_all(
    document: &AssemblyDocument,
    compiled: &Compiled,
    request: &EditRequest,
) -> Result<Vec<String>, EditFailure> {
    let named = !crate::features::picks_of(&request.values, "components").is_empty()
        || !request.selection.is_empty();
    if named {
        return selected_names(compiled, request);
    }
    if document.occurrences.is_empty() {
        return Err(model("the assembly has no components"));
    }
    Ok(document.occurrences.iter().map(|o| o.name.clone()).collect())
}

/// `assembly.update-pins`.
pub(super) fn update(
    document: &mut AssemblyDocument,
    path: &str,
    files: &Files,
    compiled: &Compiled,
    request: &EditRequest,
) -> Result<String, EditFailure> {
    let names = chosen_or_all(document, compiled, request)?;
    let released = match text_value(&request.values, "to") {
        "" | "latest" => false,
        "released" => true,
        other => return Err(model(format!("update to `{other}`? Choose latest or released"))),
    };
    let mut changed = Vec::new();
    for name in &names {
        let at = find(document, name)?;
        let o = &mut document.occurrences[at];
        let context = if released {
            let file = resolve_path(path, &o.source.definition.file.0, files);
            XrefContext::Pinned {
                revision: release::latest(&file, files)?,
            }
        } else {
            XrefContext::Live
        };
        if o.source.context != context {
            changed.push(match &context {
                XrefContext::Pinned { revision } => format!("{name} to {revision}"),
                XrefContext::Live => name.clone(),
            });
            o.source.context = context;
        }
    }
    if changed.is_empty() {
        return Err(model(if released {
            "those components are already pinned to their latest release"
        } else {
            "those components already follow their files live"
        }));
    }
    Ok(if released {
        format!("Pinned {}", changed.join(", "))
    } else {
        format!("{} now follow the latest", changed.join(", "))
    })
}

/// `assembly.release`: the store changes go to `others`.
pub(super) fn release(
    document: &AssemblyDocument,
    path: &str,
    text: &str,
    files: &Files,
    compiled: &Compiled,
    request: &EditRequest,
    others: &mut Vec<FileEdit>,
) -> Result<String, EditFailure> {
    let named = !crate::features::picks_of(&request.values, "components").is_empty()
        || !request.selection.is_empty();
    let mut targets: Vec<(String, String)> = Vec::new();
    if named {
        for name in selected_names(compiled, request)? {
            let o = &document.occurrences[find(document, &name)?];
            let file = resolve_path(path, &o.source.definition.file.0, files);
            if targets.iter().any(|(f, _)| f == &file) {
                continue;
            }
            let content = files
                .get(&file)
                .ok_or_else(|| model(format!("{file} isn't in the project")))?;
            targets.push((file, content.clone()));
        }
    } else {
        targets.push((path.to_owned(), text.to_owned()));
    }
    let mut made = Vec::new();
    for (file, content) in targets {
        let (edit, revision) = release::release(&file, &content, files)?;
        others.push(edit);
        let name = file.rsplit('/').next().unwrap_or(&file).to_owned();
        made.push(format!("{name} as {revision}"));
    }
    Ok(format!("Released {}", made.join(", ")))
}
