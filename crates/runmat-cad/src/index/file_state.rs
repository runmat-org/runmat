//! What a design file holds, as the agent's guides, its next steps and the Desktop's suggestions
//! choose by it. A part says what kind of part it is from its features and materials; every
//! other file is empty or written.
//!
//! This is the one classifier. The part index carries it ([`PartIndex::file_state`]), the
//! Desktop reads it from the projection it compiles, and `design.overview` reports it, so the
//! Guide tab, the overview's open decisions and the panel's suggestions agree.
//!
//! [`PartIndex::file_state`]: super::PartIndex::file_state

use crate::model::{FeatureKind, PartDocument};
use serde::{Deserialize, Serialize};

/// The open file's content, as a guide or a suggestion needs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum FileState {
    /// Nothing written yet: whitespace, or a part with no inputs, values, features, materials
    /// or outputs (what the Files menu writes for a new part).
    Empty,
    /// Some content.
    Written,
    /// A part with sheet-metal features: its own checks (flat pattern, bend reliefs) replace
    /// a machined part's.
    SheetMetal,
    /// A part with no body, only datum planes, axes, points, frames, curves and connectors: a
    /// skeleton other parts and assemblies place themselves by.
    Reference,
    /// A part of runs swept along paths in a pipe, duct, conduit or cable material: routing
    /// questions replace a machined part's load and machining checks.
    Piping,
    /// A part of electrical boxes and devices.
    Electrical,
    /// A part of structural members (lumber, engineered wood, steel sections, rebar): member
    /// lists and design-load checks.
    Structural,
}

impl FileState {
    /// Every file state, for checks that walk them all.
    pub const ALL: [FileState; 7] = [
        FileState::Empty,
        FileState::Written,
        FileState::SheetMetal,
        FileState::Reference,
        FileState::Piping,
        FileState::Electrical,
        FileState::Structural,
    ];
}

/// Words in a run's material name: pipe, tube, duct, conduit and cable.
const RUN_MATERIALS: &[&str] = &[
    "pex", "pipe", "dwv", "tube", "tubing", "duct", "conduit", "cable",
];
/// Words in an electrical box's or device's material name.
const ELECTRICAL_MATERIALS: &[&str] = &["electrical", "device", "receptacle", "panelboard"];
/// Words in a structural member's material name.
const STRUCTURAL_MATERIALS: &[&str] = &[
    "lumber",
    "joist",
    "lvl",
    "lsl",
    "psl",
    "glulam",
    "timber",
    "rebar",
    "steel_beam",
    "steel_section",
    "steel_tube",
    "steel_channel",
    "steel_angle",
];
/// A part with this many sweeps is a set of runs whatever its material.
const RUN_SWEEPS: usize = 3;

/// What a part is, from its features and materials, in this order:
/// 1. sheet metal, when any feature is a sheet-metal feature;
/// 2. a reference skeleton, when every feature besides sketches is a datum;
/// 3. runs, with three or more sweeps, or a sweep and a pipe, duct, conduit or cable material;
/// 4. electrical, with an electrical box or device material;
/// 5. structural, with a lumber, engineered-wood, steel-section or rebar material;
/// 6. else a written part, or empty when it declares nothing.
pub fn part_file_state(doc: &PartDocument) -> FileState {
    if doc.features.is_empty()
        && doc.inputs.is_empty()
        && doc.lets.is_empty()
        && doc.materials.is_empty()
        && doc.parts.is_empty()
    {
        return FileState::Empty;
    }
    let kinds = || doc.features.iter().map(|f| &f.kind);
    if kinds().any(|k| matches!(k, FeatureKind::SheetMetal(_))) {
        return FileState::SheetMetal;
    }
    let mut modeled = kinds()
        .filter(|k| !matches!(k, FeatureKind::Sketch(_)))
        .peekable();
    if modeled.peek().is_some() && modeled.all(is_datum) {
        return FileState::Reference;
    }
    let materials: Vec<&str> = doc
        .materials
        .iter()
        .map(|m| m.material.0.as_str())
        .chain(doc.parts.iter().filter_map(|p| p.material.as_ref()).map(|m| m.0.as_str()))
        .collect();
    let any_material = |words: &[&str]| materials.iter().any(|m| names_any(m, words));
    let sweeps = kinds().filter(|k| matches!(k, FeatureKind::Sweep(_))).count();
    if sweeps >= RUN_SWEEPS || (sweeps > 0 && any_material(RUN_MATERIALS)) {
        return FileState::Piping;
    }
    if any_material(ELECTRICAL_MATERIALS) {
        return FileState::Electrical;
    }
    if any_material(STRUCTURAL_MATERIALS) {
        return FileState::Structural;
    }
    FileState::Written
}

/// Datum geometry: places other features, makes no body.
fn is_datum(kind: &FeatureKind) -> bool {
    matches!(
        kind,
        FeatureKind::Plane(_)
            | FeatureKind::Axis(_)
            | FeatureKind::Point(_)
            | FeatureKind::Frame(_)
            | FeatureKind::Curve(_)
            | FeatureKind::MateConnector(_)
    )
}

/// Whether a built-in material reference (`material.pex`) names one of `words`. Library
/// materials (`library.…`) say nothing about what the part is.
fn names_any(material: &str, words: &[&str]) -> bool {
    material
        .strip_prefix("material.")
        .is_some_and(|name| words.iter().any(|w| name.contains(w)))
}
