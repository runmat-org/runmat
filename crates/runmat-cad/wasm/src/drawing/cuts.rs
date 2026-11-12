//! Checks on the section views' cuts once their lines are projected.

use crate::runtime::Diagnostic;
use runmat_cad::diagnostics::SpanMap;
use runmat_cad::drawing::{DrawingDocument, ViewKind};
use serde_json::{Map, Value};

/// A warning for each section view whose cutting plane cuts no material: the plane misses the
/// model or lies on one of its faces, so the view would draw the model whole with no hatch.
/// The warning points at the view's `offset` when it has one, else at the view.
pub(super) fn sections_that_cut_nothing(
    path: &str,
    document: &DrawingDocument,
    hlr: &Map<String, Value>,
    spans: &SpanMap,
) -> Vec<Diagnostic> {
    document
        .views
        .iter()
        .filter(|v| matches!(v.kind, ViewKind::Section { .. }))
        .filter_map(|v| {
            let entry = hlr.get(&v.name)?;
            let cut = entry["hatch"].as_array().is_some_and(|h| !h.is_empty());
            if cut {
                return None;
            }
            let span = spans
                .spans
                .get(&format!("view.{}#offset", v.name))
                .or_else(|| spans.spans.get(&format!("view.{}", v.name)));
            Some(Diagnostic {
                path: path.to_owned(),
                start: span.map_or(0, |s| s.start as usize),
                end: span.map_or(0, |s| s.end as usize),
                severity: "warning".into(),
                message: format!(
                    "section {} cuts no material: its cutting plane misses the model or lies on a face, so nothing is hatched. Move the plane or its offset into the model",
                    v.name
                ),
            })
        })
        .collect()
}
