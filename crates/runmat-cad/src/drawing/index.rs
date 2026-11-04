//! The drawing JSON projection: the document (sheets, views by reference, annotations,
//! tables) with what checking it resolved (diagnostics, title-block text, balloon numbers,
//! annotations that found their entities). The Desktop's drawing Outline reads this.

use super::{check_drawing, DrawingDocument, ModelReferences};
use crate::diagnostics::Diagnostic;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Bumped on any change to the projection's shape.
pub const DRAWING_INDEX_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DrawingIndex {
    pub format_version: u32,
    pub document: DrawingDocument,
    pub diagnostics: Vec<Diagnostic>,
    /// Sheet → title-block rows (label, text).
    pub title_blocks: BTreeMap<String, Vec<(String, String)>>,
    /// Balloon id → item number as shown.
    pub balloons: BTreeMap<String, String>,
    /// Annotation id → whether it resolved to model entities.
    pub resolved: BTreeMap<String, bool>,
}

pub fn build_drawing_index(doc: &DrawingDocument, model: &dyn ModelReferences) -> DrawingIndex {
    let report = check_drawing(doc, model);
    DrawingIndex {
        format_version: DRAWING_INDEX_FORMAT_VERSION,
        document: doc.clone(),
        diagnostics: report.diagnostics.diagnostics.clone(),
        title_blocks: report.title_blocks.clone(),
        balloons: report.balloons.clone(),
        resolved: doc
            .annotations
            .iter()
            .map(|a| (a.id.clone(), report.resolved.contains_key(&a.id)))
            .collect(),
    }
}

pub fn to_json(index: &DrawingIndex) -> String {
    let mut json = serde_json::to_string_pretty(index).expect("drawing index serializes");
    json.push('\n');
    json
}
