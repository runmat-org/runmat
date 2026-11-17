//! A part's analysis representations: the simplified geometry a study meshes in place of the
//! design, declared in the part's own source and built from it on demand. The design itself is
//! never changed.
//!
//! ```text
//! representation simple : defeatured = { max_fillet = 2 mm, max_hole = 3 mm, max_chamfer = 1 mm }
//! representation shell : midsurface = { body = feature.plate.body }
//! representation spar : beam = { body = feature.spar.body }
//! ```
//!
//! - `defeatured`: the bodies with fillets, holes and chamfers at or below the thresholds
//!   removed, and the removed features listed. Faces in a region stay when `keep_regions`.
//! - `midsurface`: the mid-surface of a thin body, from its pairs of opposite parallel faces,
//!   with each pair's wall thickness.
//! - `beam`: the centerline of a prismatic body and its cross-section's properties.

use crate::assembly::model::Representation;
use crate::refs::Reference;
use crate::value::ValueExpr;
use serde::{Deserialize, Serialize};

fn yes() -> bool {
    true
}

fn is_true(b: &bool) -> bool {
    *b
}

/// One `representation <name> : <kind> = { … }` statement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartRepresentation {
    pub name: String,
    /// `defeatured`, `midsurface` or `beam`.
    pub kind: Representation,
    /// The body a mid-surface or beam is made from; absent: the part's only body.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<Reference>,
    /// Fillets at or below this radius are removed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_fillet: Option<ValueExpr>,
    /// Holes at or below this diameter are removed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_hole: Option<ValueExpr>,
    /// Chamfers at or below this distance are removed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_chamfer: Option<ValueExpr>,
    /// Faces in one of the part's regions stay, so a study's loads and supports keep their faces.
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub keep_regions: bool,
}

impl PartRepresentation {
    /// The kinds a part can declare: the analysis idealisations.
    pub const KINDS: [Representation; 3] = [
        Representation::Defeatured,
        Representation::Midsurface,
        Representation::Beam,
    ];
}
