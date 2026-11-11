//! AP242 semantic PMI, written into and read from a STEP file whose faces carry names (the
//! kernel writes each named face's reference text as its `ADVANCED_FACE` name).
//!
//! Each item is linked to its faces the way the CAx-IF Recommended Practices for PMI describe:
//! a `shape_aspect` (or `datum_feature`) of the part's `product_definition_shape`, and one
//! `geometric_item_specific_usage` per face. What is written:
//!
//! - **Datums:** `datum_feature` on the faces, related to its `datum` by a
//!   `shape_aspect_relationship`.
//! - **Geometric tolerances:** the characteristic's entity (`flatness_tolerance`,
//!   `position_tolerance`, …) with its magnitude as a `length_measure_with_unit`; a
//!   `datum_system` of `datum_reference_compartment`s (with material-condition modifiers);
//!   `geometric_tolerance_with_modifiers` for MMC and LMC; a `tolerance_zone` of form
//!   `cylindrical or circular` for a diametral zone.
//! - **Sizes:** `dimensional_size` with its nominal value (`dimensional_characteristic_representation`)
//!   and a `plus_minus_tolerance` of a `tolerance_value` (lower and upper limit) or of
//!   `limits_and_fits` for a fit.
//! - **Locations:** `dimensional_location` between two shape aspects, valued and toleranced
//!   the same way.
//!
//! Values are in millimetres here; the writer and reader convert to and from the file's
//! length unit.

pub mod read;
pub mod schema;
pub mod write;

pub use read::read;
pub use write::embed;

use crate::pmi::{Characteristic, DatumReference, MaterialCondition};

#[derive(Clone, Debug, PartialEq)]
pub struct StepDatum {
    pub label: String,
    /// Face names.
    pub faces: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StepTolerance {
    pub name: String,
    pub characteristic: Characteristic,
    pub faces: Vec<String>,
    /// The zone's width or diameter (mm).
    pub value: f64,
    pub diametral: bool,
    pub modifier: MaterialCondition,
    pub datums: Vec<DatumReference>,
}

/// A toleranced value (mm): the nominal, and the limits as deviations from it, or a fit.
#[derive(Clone, Debug, PartialEq)]
pub struct Limits {
    pub nominal: f64,
    pub lower: f64,
    pub upper: f64,
    /// An ISO 286 fit such as `H7` (instead of the deviations).
    pub fit: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StepSize {
    pub name: String,
    /// The CAx-IF size kind: `diameter`, `radius`, `thickness`, …
    pub kind: String,
    pub faces: Vec<String>,
    pub limits: Limits,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StepLocation {
    pub name: String,
    pub from: Vec<String>,
    pub to: Vec<String>,
    pub limits: Limits,
}

/// The PMI of one part, by face names.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepPmi {
    pub datums: Vec<StepDatum>,
    pub tolerances: Vec<StepTolerance>,
    pub sizes: Vec<StepSize>,
    pub locations: Vec<StepLocation>,
}

impl StepPmi {
    pub fn is_empty(&self) -> bool {
        self.datums.is_empty()
            && self.tolerances.is_empty()
            && self.sizes.is_empty()
            && self.locations.is_empty()
    }
}

/// AP242 entity names of the characteristics.
pub const CHARACTERISTICS: &[(Characteristic, &str)] = &[
    (Characteristic::Straightness, "STRAIGHTNESS_TOLERANCE"),
    (Characteristic::Flatness, "FLATNESS_TOLERANCE"),
    (Characteristic::Circularity, "ROUNDNESS_TOLERANCE"),
    (Characteristic::Cylindricity, "CYLINDRICITY_TOLERANCE"),
    (Characteristic::ProfileOfLine, "LINE_PROFILE_TOLERANCE"),
    (
        Characteristic::ProfileOfSurface,
        "SURFACE_PROFILE_TOLERANCE",
    ),
    (Characteristic::Parallelism, "PARALLELISM_TOLERANCE"),
    (
        Characteristic::Perpendicularity,
        "PERPENDICULARITY_TOLERANCE",
    ),
    (Characteristic::Angularity, "ANGULARITY_TOLERANCE"),
    (Characteristic::Position, "POSITION_TOLERANCE"),
    (Characteristic::Concentricity, "CONCENTRICITY_TOLERANCE"),
    (Characteristic::Symmetry, "SYMMETRY_TOLERANCE"),
    (Characteristic::CircularRunout, "CIRCULAR_RUNOUT_TOLERANCE"),
    (Characteristic::TotalRunout, "TOTAL_RUNOUT_TOLERANCE"),
];

/// Characteristics whose AP242 entity is a subtype of `geometric_tolerance_with_datum_reference`.
pub fn entity_has_datums(c: Characteristic) -> bool {
    matches!(
        c,
        Characteristic::Parallelism
            | Characteristic::Perpendicularity
            | Characteristic::Angularity
            | Characteristic::Concentricity
            | Characteristic::Symmetry
            | Characteristic::CircularRunout
            | Characteristic::TotalRunout
    )
}

pub const CYLINDRICAL_ZONE: &str = "cylindrical or circular";

/// `geometric_tolerance_modifier` and `simple_datum_reference_modifier` values.
pub fn modifier_enum(m: MaterialCondition) -> Option<&'static str> {
    match m {
        MaterialCondition::Rfs => None,
        MaterialCondition::Mmc => Some("MAXIMUM_MATERIAL_REQUIREMENT"),
        MaterialCondition::Lmc => Some("LEAST_MATERIAL_REQUIREMENT"),
    }
}
