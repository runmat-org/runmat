//! The drawing JSON projection: the document (sheets, views by reference, annotations,
//! tables) with what checking it resolved (diagnostics, title-block text, balloon numbers,
//! annotations that found their entities, and each annotation's text as the sheet shows it).
//! The Desktop's drawing Outline reads this.

use super::{
    check_drawing, AnnotationKind, DimensionStyle, DrawingDocument, ModelRef, ModelReferences,
};
use crate::diagnostics::Diagnostic;
use crate::ids::PropertyName;
use crate::refs::name::EntityName;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Bumped on any change to the projection's shape.
pub const DRAWING_INDEX_FORMAT_VERSION: u32 = 2;

/// What the drawn model measures and declares, for annotation text (`40 mm`, `⏥ 0.05 A`).
pub trait AnnotationValues {
    /// A dimension's measured value with its unit, from the entities it resolved to.
    fn dimension(
        &self,
        _model: &ModelRef,
        _style: DimensionStyle,
        _entities: &[Vec<EntityName>],
    ) -> Option<String> {
        None
    }
    /// A PMI tolerance's feature control frame as text: symbol, value and datums.
    fn tolerance_frame(&self, _model: &ModelRef, _tolerance: &str) -> Option<String> {
        None
    }
}

/// No measurements: dimensions and frames fall back to their PMI names.
pub struct NoAnnotationValues;

impl AnnotationValues for NoAnnotationValues {}

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
    /// Annotation id → its text as the sheet shows it (`40 mm`, `Balloon 3`, a note with its
    /// fields filled).
    pub text: BTreeMap<String, String>,
}

pub fn build_drawing_index(
    doc: &DrawingDocument,
    model: &dyn ModelReferences,
    values: &dyn AnnotationValues,
) -> DrawingIndex {
    let report = check_drawing(doc, model);
    let text = doc
        .annotations
        .iter()
        .filter_map(|a| {
            let view = doc.views.iter().find(|v| v.name == a.view)?;
            let at = &view.model;
            let entities = report.resolved.get(&a.id).map(Vec::as_slice).unwrap_or(&[]);
            let shown = match &a.kind {
                AnnotationKind::Dimension {
                    style,
                    pmi,
                    reference,
                    ..
                } => {
                    let value = values
                        .dimension(at, *style, entities)
                        .or_else(|| pmi.clone())
                        .unwrap_or_else(|| "Dimension".into());
                    if *reference {
                        format!("({value})")
                    } else {
                        value
                    }
                }
                AnnotationKind::HoleCallout { .. } => "Hole callout".into(),
                AnnotationKind::Gdt { tolerance } => values
                    .tolerance_frame(at, tolerance)
                    .unwrap_or_else(|| tolerance.clone()),
                AnnotationKind::DatumSymbol { label } => format!("Datum {label}"),
                AnnotationKind::DatumTarget { label } => format!("Datum target {label}"),
                AnnotationKind::SurfaceFinish { .. } => "Surface finish".into(),
                AnnotationKind::Weld { symbol, .. } => format!("Weld {symbol}"),
                AnnotationKind::Note { text, .. } => {
                    fill_fields(text, |field| model.property(at, &PropertyName::new(field)))
                }
                AnnotationKind::Balloon { .. } => match report.balloons.get(&a.id) {
                    Some(number) => format!("Balloon {number}"),
                    None => "Balloon".into(),
                },
                AnnotationKind::CenterMark { .. } => "Center mark".into(),
                AnnotationKind::Centerline { .. } => "Centerline".into(),
            };
            Some((a.id.clone(), shown))
        })
        .collect();
    DrawingIndex {
        text,
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

/// `{field}` replaced by the property's text; a field the model lacks stays as written.
fn fill_fields(text: &str, property: impl Fn(&str) -> Option<String>) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(i) = rest.find('{') {
        let Some(j) = rest[i..].find('}') else { break };
        out.push_str(&rest[..i]);
        let field = &rest[i + 1..i + j];
        match property(field) {
            Some(value) => out.push_str(&value),
            None => out.push_str(&rest[i..=i + j]),
        }
        rest = &rest[i + j + 1..];
    }
    out.push_str(rest);
    out
}

pub fn to_json(index: &DrawingIndex) -> String {
    let mut json = serde_json::to_string_pretty(index).expect("drawing index serializes");
    json.push('\n');
    json
}
