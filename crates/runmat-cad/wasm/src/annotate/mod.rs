//! Annotate on a part: 3D annotations (model-based definition).
//!
//! A part's annotations share the drawing's annotation model ([`AnnotationKind`]): a datum
//! symbol, GD&T frame, toleranced dimension, surface finish or coating shows a semantic PMI item
//! the part declares ([`runmat_cad::pmi::PmiSet`]), the same item a drawing's frame or symbol
//! names, and a weld symbol or note carries its own content. Each is written in the `.part` as
//! an `annotation` statement on the part's canonical face and edge references, so it follows the
//! geometry through every rebuild; [`display`] places it in the model for the canvas and lists it
//! for the Outline.
//!
//! Commands: `annotate.datum`, `annotate.gdt`, `annotate.dimension`, `annotate.surface-finish`,
//! `annotate.coating`, `annotate.weld-symbol`, `annotate.note`, `annotate.auto-dimension`
//! ([`auto`]) and `annotate.annotation-view` ([`view`]). On a selected annotation
//! (`annotation:<id>`): `part.delete` removes it with the PMI item it shows, and
//! `part.set-property` moves its label (`at`), files it under a view (`view`), or changes a
//! field of its definition, which re-applies its command in place under the same names.

mod auto;
mod build;
pub(crate) mod display;
pub(crate) mod form;
#[cfg(test)]
mod tests;
mod view;

pub use display::{annotation_display, annotation_rows};

use crate::drawing_edits::Values;
use crate::edit::{model, EditRequest};
use crate::picks::Picks;
use runmat_cad::drawing::AnnotationKind;
use runmat_cad::model::PartDocument;
use runmat_cad::pmi::{ModelAnnotation, PmiSet};
use runmat_cad::syntax::edit::{edit_part, EditFailure, SourceChange};
use runmat_cad::syntax::parse_part;
use serde_json::Value;

/// The Outline id prefix of an annotation.
pub const ANNOTATION: &str = "annotation:";
/// The Outline id prefix of an annotation view.
pub const VIEW: &str = "annotation_view:";

type Edit = Result<(SourceChange, String), EditFailure>;

/// An Annotate command, or a delete or property change on selected annotations; `None` for
/// any other request.
pub(crate) fn edit(path: &str, text: &str, request: &EditRequest) -> Option<Edit> {
    let annotations: Vec<&str> = request
        .selection
        .iter()
        .filter_map(|s| s.strip_prefix(ANNOTATION))
        .collect();
    let views: Vec<&str> = request
        .selection
        .iter()
        .filter_map(|s| s.strip_prefix(VIEW))
        .collect();
    Some(match request.command.as_str() {
        "part.delete" if !annotations.is_empty() || !views.is_empty() => {
            delete(path, text, &annotations, &views)
        }
        "part.set-property" if annotations.len() == 1 => {
            set_property(path, text, annotations[0], &request.values)
        }
        "part.set-property" if views.len() == 1 => {
            view::set_property(path, text, views[0], &request.values)
        }
        command if command.starts_with("annotate.") => {
            run(path, text, command, &request.values, None)
        }
        _ => return None,
    })
}

fn document(path: &str, text: &str) -> Result<PartDocument, EditFailure> {
    let parsed = parse_part(path, text);
    parsed
        .document
        .ok_or(EditFailure::Invalid(parsed.diagnostics))
}

/// Runs `command` with `values`; with `replace`, in place of that annotation (its id, view and
/// label position, and the names of the PMI items it showed, are kept).
fn run(path: &str, text: &str, command: &str, values: &Values, replace: Option<&str>) -> Edit {
    let parsed = document(path, text)?;
    let picks = Picks::new(&parsed, path, text);
    match command {
        "annotate.auto-dimension" => return auto::run(path, text, &parsed, values, &picks),
        "annotate.annotation-view" => return view::run(path, text, &parsed, values),
        _ => {}
    }
    let mut added: Vec<String> = Vec::new();
    let change = edit_part(text, path, |doc| {
        let (old, name, place) = match replace {
            Some(id) => {
                let name = doc
                    .pmi
                    .annotations
                    .iter()
                    .find(|a| a.id == id)
                    .and_then(|a| semantic_name(&doc.pmi, a));
                let (old, place) = take(&mut doc.pmi, id, false)?;
                (Some(old), name, place)
            }
            None => (None, None, doc.pmi.annotations.len()),
        };
        let kinds = build::declare(command, values, &picks, &mut doc.pmi, name.as_deref())?;
        for (k, kind) in kinds.into_iter().enumerate() {
            let annotation = match (&old, k) {
                // The edited annotation keeps its id, view, label position and place.
                (Some(a), 0) => ModelAnnotation {
                    id: a.id.clone(),
                    kind,
                    view: a.view.clone(),
                    at: a.at,
                },
                _ => ModelAnnotation {
                    id: fresh_id(&doc.pmi, stem(&kind)),
                    kind,
                    view: old.as_ref().and_then(|a| a.view.clone()),
                    at: None,
                },
            };
            added.push(annotation.id.clone());
            let at = (place + k).min(doc.pmi.annotations.len());
            doc.pmi.annotations.insert(at, annotation);
        }
        Ok(())
    })?;
    let verb = if replace.is_some() {
        "Changed"
    } else {
        "Added"
    };
    Ok((change, format!("{verb} {}", added.join(", "))))
}

/// The PMI item an annotation shows, by name: its tolerance, size or location, finish, or
/// treatment (a datum keeps its label from the form).
fn semantic_name(pmi: &PmiSet, a: &ModelAnnotation) -> Option<String> {
    match &a.kind {
        AnnotationKind::Gdt { tolerance } => Some(tolerance.clone()),
        AnnotationKind::Dimension { pmi: name, .. } => name.clone(),
        AnnotationKind::Coating { treatment } => Some(treatment.clone()),
        AnnotationKind::SurfaceFinish { on } => pmi
            .finishes
            .iter()
            .find(|f| f.on.contains(on))
            .map(|f| f.name.clone()),
        _ => None,
    }
}

/// Removes annotation `id` and the PMI item only it shows, answering it and where it stood.
/// A datum another frame uses can't be deleted; an edit (`delete` false) keeps it declared.
fn take(pmi: &mut PmiSet, id: &str, delete: bool) -> Result<(ModelAnnotation, usize), EditFailure> {
    let at = pmi
        .annotations
        .iter()
        .position(|a| a.id == id)
        .ok_or_else(|| model(format!("the part has no annotation `{id}`")))?;
    let old = pmi.annotations.remove(at);
    // Shown by another annotation too: the item stays.
    let shared = pmi.annotations.iter().any(|a| a.kind == old.kind);
    let name = semantic_name(pmi, &old);
    if !shared {
        match &old.kind {
            AnnotationKind::Gdt { tolerance } => pmi.tolerances.retain(|t| &t.name != tolerance),
            AnnotationKind::Dimension { pmi: Some(n), .. } => {
                pmi.sizes.retain(|s| &s.name != n);
                pmi.locations.retain(|l| &l.name != n);
            }
            AnnotationKind::Coating { treatment } => {
                pmi.treatments.retain(|t| &t.name != treatment)
            }
            AnnotationKind::SurfaceFinish { .. } => {
                if let Some(n) = &name {
                    pmi.finishes.retain(|f| &f.name != n);
                }
            }
            AnnotationKind::DatumSymbol { label } => {
                let users: Vec<&str> = pmi
                    .tolerances
                    .iter()
                    .filter(|t| t.datums.iter().any(|d| &d.label == label))
                    .map(|t| t.name.as_str())
                    .collect();
                if delete && !users.is_empty() {
                    return Err(model(format!(
                        "datum {label} is used by {}; delete or change those frames first",
                        users.join(", ")
                    )));
                }
                if users.is_empty() {
                    pmi.datums.retain(|d| &d.label != label);
                }
            }
            _ => {}
        }
    }
    Ok((old, at))
}

/// The id stem of a new annotation of `kind`.
fn stem(kind: &AnnotationKind) -> &'static str {
    match kind {
        AnnotationKind::DatumSymbol { .. } => "datum",
        AnnotationKind::Gdt { .. } => "gdt",
        AnnotationKind::Dimension { .. } => "dim",
        AnnotationKind::SurfaceFinish { .. } => "finish",
        AnnotationKind::Coating { .. } => "coating",
        AnnotationKind::Weld { .. } => "weld",
        _ => "note",
    }
}

fn fresh_id(pmi: &PmiSet, stem: &str) -> String {
    (1..)
        .map(|n| format!("{stem}{n}"))
        .find(|id| !pmi.annotations.iter().any(|a| &a.id == id))
        .expect("a free annotation id")
}

fn delete(path: &str, text: &str, annotations: &[&str], views: &[&str]) -> Edit {
    let change = edit_part(text, path, |doc| {
        for id in annotations {
            take(&mut doc.pmi, id, true)?;
        }
        for name in views {
            view::remove(&mut doc.pmi, name)?;
        }
        Ok(())
    })?;
    let names: Vec<&str> = annotations.iter().chain(views).copied().collect();
    Ok((change, format!("Deleted {}", names.join(", "))))
}

/// One field of a selected annotation, as Properties edits it.
fn set_property(path: &str, text: &str, id: &str, values: &Values) -> Edit {
    let (field, value) = values
        .iter()
        .find(|(k, _)| k.as_str() != "configuration")
        .ok_or_else(|| model("no property to set"))?;
    match field.as_str() {
        // The label dragged in the view: its offset from the attachment (mm), or back to the
        // default place.
        "at" => {
            // From the canvas as text (`12, 0, 30`), from the agent as an array.
            let listed: Value = match value {
                Value::String(t) if t.trim().is_empty() => Value::Null,
                Value::String(t) => Value::Array(
                    t.split(',')
                        .map(|x| x.trim().parse::<f64>().map_or(Value::Null, Value::from))
                        .collect(),
                ),
                other => other.clone(),
            };
            let at = match &listed {
                Value::Null => None,
                Value::Array(a) if a.len() == 3 => {
                    let v: Vec<f64> = a.iter().filter_map(Value::as_f64).collect();
                    if v.len() != 3 || v.iter().any(|x| !x.is_finite()) {
                        return Err(model("a label position is three numbers (mm)"));
                    }
                    Some([v[0], v[1], v[2]])
                }
                _ => return Err(model("a label position is three numbers (mm)")),
            };
            let change = edit_part(text, path, |doc| {
                let a = doc
                    .pmi
                    .annotations
                    .iter_mut()
                    .find(|a| a.id == id)
                    .ok_or_else(|| model(format!("the part has no annotation `{id}`")))?;
                a.at = at;
                Ok(())
            })?;
            Ok((change, format!("Moved {id}")))
        }
        "view" => {
            let name = value.as_str().unwrap_or("").trim().to_owned();
            let change = edit_part(text, path, |doc| {
                if !name.is_empty() && !doc.pmi.views.iter().any(|v| v.name == name) {
                    return Err(model(format!("the part has no annotation view `{name}`")));
                }
                let a = doc
                    .pmi
                    .annotations
                    .iter_mut()
                    .find(|a| a.id == id)
                    .ok_or_else(|| model(format!("the part has no annotation `{id}`")))?;
                a.view = (!name.is_empty()).then_some(name.clone());
                Ok(())
            })?;
            Ok((change, format!("Set {id} view")))
        }
        _ => {
            // The annotation's definition as its command's values, with this field changed.
            let doc = document(path, text)?;
            let a = doc
                .pmi
                .annotations
                .iter()
                .find(|a| a.id == id)
                .ok_or_else(|| model(format!("the part has no annotation `{id}`")))?;
            let (command, mut current) = form::values(&doc, a).ok_or_else(|| {
                model(format!("`{id}` is edited in the source, not in Properties"))
            })?;
            current.insert(field.clone(), value.clone());
            run(path, text, command, &current, Some(id))
        }
    }
}
