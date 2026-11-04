//! Mold tooling: parting line, shut-off surfaces, the parting surface, the core/cavity split,
//! the shrink scale, and undercut detection.

use crate::model::datum::{AxisRef, PlaneRef};
use crate::model::visit::{Params, Visit};
use crate::refs::Reference;
use crate::value::ValueExpr;
use serde::{Deserialize, Serialize};

// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mold")]
pub enum MoldFeature {
    /// Scale the part about a point for the material's shrink (per axis, or uniform).
    Shrink {
        bodies: Vec<Reference>,
        /// Shrink allowances as fractions (0.005 = 0.5 %); the part grows by `1 + s`.
        x: ValueExpr,
        y: ValueExpr,
        z: ValueExpr,
    },
    /// The parting line: edges where faces change from drafted towards the pull to drafted
    /// away from it. Given explicitly, or found by draft analysis when `edges` is empty.
    PartingLine {
        body: Reference,
        pull: AxisRef,
        #[serde(default)]
        edges: Vec<Reference>,
        /// Draft below which a face is reported as needing draft.
        min_draft: ValueExpr,
    },
    /// Patches closing through-openings so core and cavity separate.
    ShutOff { loops: Vec<Vec<Reference>> },
    /// The parting surface: extends the parting line outward, here planar.
    PartingSurface { plane: PlaneRef, extent: ValueExpr },
    /// Split a mold block into core and cavity by the parting surface around the part.
    CoreCavity {
        part: Reference,
        parting: PlaneRef,
        /// The block's margin around the part.
        margin: ValueExpr,
    },
    /// Report faces that cannot release along the pull direction.
    Undercuts { body: Reference, pull: AxisRef },
}

impl Params for MoldFeature {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        match self {
            MoldFeature::Shrink { bodies, x, y, z } => {
                v.references("bodies", bodies);
                v.value(x);
                v.value(y);
                v.value(z);
            }
            MoldFeature::PartingLine {
                body,
                pull,
                edges,
                min_draft,
            } => {
                v.reference("body", body);
                v.axis("pull", pull);
                v.references("edges", edges);
                v.value(min_draft);
            }
            MoldFeature::ShutOff { loops } => {
                for (i, l) in loops.iter().enumerate() {
                    v.references(&format!("loops[{}]", i), l);
                }
            }
            MoldFeature::PartingSurface { plane, extent } => {
                v.plane("plane", plane);
                v.value(extent);
            }
            MoldFeature::CoreCavity {
                part,
                parting,
                margin,
            } => {
                v.reference("part", part);
                v.plane("parting", parting);
                v.value(margin);
            }
            MoldFeature::Undercuts { body, pull } => {
                v.reference("body", body);
                v.axis("pull", pull);
            }
        }
    }
    fn kind_name(&self) -> &'static str {
        match self {
            MoldFeature::Shrink { .. } => "mold.shrink",
            MoldFeature::PartingLine { .. } => "mold.parting_line",
            MoldFeature::ShutOff { .. } => "mold.shut_off",
            MoldFeature::PartingSurface { .. } => "mold.parting_surface",
            MoldFeature::CoreCavity { .. } => "mold.core_cavity",
            MoldFeature::Undercuts { .. } => "mold.undercuts",
        }
    }
    fn new_body(&self) -> bool {
        matches!(
            self,
            MoldFeature::PartingSurface { .. } | MoldFeature::CoreCavity { .. }
        )
    }
    fn body_feature(&self) -> bool {
        !matches!(
            self,
            MoldFeature::PartingLine { .. } | MoldFeature::Undercuts { .. }
        )
    }
}
