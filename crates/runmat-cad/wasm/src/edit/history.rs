//! The feature tree's own commands on a part:
//!
//! - `part.rename` (`values.name`): the selected feature, sketch, plane, mate connector, input
//!   or region takes the new name, in its declaration and every reference in the file
//!   ([`runmat_cad::syntax::rename`]). Among `files`, drawings whose views show this part follow
//!   a feature rename, assemblies that set the input on an occurrence of the part (directly or
//!   by override) follow an input rename, and `.fea` studies whose geometry reads the part
//!   follow a region rename (`select name "<region>"`). A taken or invalid name is refused with
//!   the reason; a body is named after its feature and refuses with that reason.
//! - `part.rollback-here`: the rollback bar moves to just after the selected feature
//!   (`rollback_after = <feature>` in the part); later features are not regenerated.
//!   `part.rollback-end` takes the bar back to the end (the line goes).
//!
//! Deleting the feature the bar sits after, or renaming it, keeps the bar with it
//! ([`super::part_edit`] and the rename).

use super::{feature_of, model, text_value, EditRequest};
use crate::runtime::Files;
use runmat_cad::modules::ModuleScope;
use runmat_cad::syntax::edit::{edit_part, EditFailure, SourceChange};
use runmat_cad::syntax::rename::{
    assemblies_in, rename_in_drawing, rename_input_in_assembly, rename_part_item,
    rename_region_in_study, RenameKind,
};

/// What an Outline id names, for a rename.
fn rename_target(id: &str) -> Result<(RenameKind, &str), EditFailure> {
    if let Some(feature) = feature_of(id) {
        return Ok((RenameKind::Feature, feature));
    }
    if let Some(input) = id.strip_prefix("input:") {
        return Ok((RenameKind::Input, input));
    }
    if let Some(region) = id.strip_prefix("region:") {
        return Ok((RenameKind::Region, region));
    }
    if id.starts_with("body:") {
        return Err(model(
            "a body is named after the feature that makes it: rename that feature",
        ));
    }
    if id.starts_with("origin:") {
        return Err(model("the origin's planes, axes and point keep their names"));
    }
    Err(model(format!("`{id}` can't be renamed")))
}

/// A file that follows a rename: its path, its source before the change, and the change.
type FollowingChange = (String, String, SourceChange);

/// `part.rename`: the part's own change, then each drawing that follows it.
pub(super) fn rename(
    path: &str,
    text: &str,
    files: &Files,
    request: &EditRequest,
) -> Result<(SourceChange, String, Vec<FollowingChange>), EditFailure> {
    let [selected] = request.selection.as_slice() else {
        return Err(model("select one item to rename"));
    };
    let (kind, from) = rename_target(selected)?;
    let to = text_value(&request.values, "name").trim();
    let change = rename_part_item(text, path, kind, from, to)?;
    let mut others = Vec::new();
    let scope = ModuleScope::from_files(
        files
            .iter()
            .map(|(p, t)| (p.as_str(), t.as_str()))
            .filter(|(p, _)| *p != path)
            .chain([(path, text)]),
    );
    let assemblies = match kind {
        RenameKind::Input => assemblies_in(
            files
                .iter()
                .map(|(p, t)| (p.as_str(), t.as_str()))
                .filter(|(p, _)| *p != path),
            &scope,
        ),
        _ => Default::default(),
    };
    for (other, source) in files.iter().filter(|(p, _)| p.as_str() != path) {
        let follow = match kind {
            RenameKind::Feature if other.ends_with(".drawing") => {
                rename_in_drawing(source, other, &scope, path, from, to)?
            }
            RenameKind::Input if other.ends_with(".assembly") => {
                rename_input_in_assembly(source, other, &scope, &assemblies, path, from, to)?
            }
            RenameKind::Region if other.to_ascii_lowercase().ends_with(".fea") => {
                rename_region_in_study(source, other, path, from, to)?
            }
            _ => None,
        };
        if let Some(follow) = follow {
            others.push((other.clone(), source.clone(), follow));
        }
    }
    let noun = match kind {
        RenameKind::Feature => ("drawing", "drawings"),
        RenameKind::Input => ("assembly", "assemblies"),
        RenameKind::Region => ("study", "studies"),
    };
    let followed = match others.len() {
        0 => String::new(),
        1 => format!(" and in 1 {}", noun.0),
        n => format!(" and in {n} {}", noun.1),
    };
    Ok((change, format!("Renamed {from} to {to}{followed}"), others))
}

/// `part.rollback-here` and `part.rollback-end`.
pub(super) fn rollback(
    path: &str,
    text: &str,
    request: &EditRequest,
) -> Result<(SourceChange, String), EditFailure> {
    let here = request.command == "part.rollback-here";
    let feature = if here {
        let selected: Vec<&str> = request
            .selection
            .iter()
            .filter_map(|s| feature_of(s))
            .collect();
        let [feature] = selected.as_slice() else {
            return Err(model("select the feature to roll back to"));
        };
        Some((*feature).to_owned())
    } else {
        None
    };
    let mut summary = String::new();
    let change = edit_part(text, path, |document| {
        let at = match &feature {
            Some(id) => Some(
                document
                    .features
                    .iter()
                    .position(|f| f.id.0 == *id)
                    .ok_or_else(|| model(format!("the part has no feature `{id}`")))?,
            ),
            None => None,
        };
        // After the last feature is the end: the bar's line goes.
        let last = document.features.len().checked_sub(1);
        document.rollback_after = match at {
            Some(at) if Some(at) != last => Some(document.features[at].id.clone()),
            _ => None,
        };
        summary = match &document.rollback_after {
            Some(id) => format!("Rolled back to {id}"),
            None => "Rolled to the end".to_owned(),
        };
        Ok(())
    })?;
    if change.edits.is_empty() {
        return Err(model(match &feature {
            Some(id) if summary.starts_with("Rolled back") => {
                format!("the model is rolled back to {id} already")
            }
            _ => "the rollback bar is at the end already".to_owned(),
        }));
    }
    Ok((change, summary))
}

/// While the model is rolled back, a feature a command adds goes in at the rollback bar, and
/// the bar moves past it, so the new feature is built on the model as it stands.
pub(super) fn insert_at_bar(
    path: &str,
    text: &str,
    change: SourceChange,
) -> Result<SourceChange, EditFailure> {
    let before = runmat_cad::syntax::parse_part(path, text);
    let Some(bar) = before.document.and_then(|d| d.rollback_after) else {
        return Ok(change);
    };
    let after = runmat_cad::syntax::parse_part(path, &change.source);
    let Some(mut document) = after.document else {
        return Ok(change);
    };
    let old: std::collections::BTreeSet<String> = runmat_cad::syntax::parse_part(path, text)
        .document
        .map(|d| d.features.into_iter().map(|f| f.id.0).collect())
        .unwrap_or_default();
    let Some(at) = document.features.iter().position(|f| f.id == bar) else {
        return Ok(change);
    };
    let added: Vec<usize> = (at + 1..document.features.len())
        .filter(|&i| !old.contains(&document.features[i].id.0))
        .collect();
    if added.is_empty() {
        return Ok(change);
    }
    let mut moved = Vec::new();
    for i in added.iter().rev() {
        moved.push(document.features.remove(*i));
    }
    moved.reverse();
    document.rollback_after = moved.last().map(|f| f.id.clone());
    for (k, feature) in moved.into_iter().enumerate() {
        document.features.insert(at + 1 + k, feature);
    }
    edit_part(text, path, |d| {
        *d = document;
        Ok(())
    })
}

#[cfg(test)]
mod tests;
