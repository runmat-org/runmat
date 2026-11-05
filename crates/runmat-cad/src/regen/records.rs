//! Typed records regeneration keeps beside the bodies: what geometry-free features evaluate
//! to (reference curves, cosmetic threads, mold analyses) and what body features derive
//! (sheet metal rules and developments, weldment cut lists).

use crate::model::curves::CurveValue;
use crate::model::detail::ThreadSpec;
use crate::model::sheet_metal::{BendRuleValue, Development, FlatBendLine, FlatLayout, Section};
use crate::model::weldment::{CutListRow, MemberCut};
use crate::refs::EntityName;
use serde::{Deserialize, Serialize};

/// The evaluated sheet metal rules of a sheet body (SI).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SheetRules {
    pub thickness: f64,
    pub bend_radius: f64,
    pub rule: BendRuleValue,
}

/// Faces a mold analysis reports, by name, with the measured draft (radians, signed: positive
/// releases along the pull).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceDraft {
    pub face: Vec<EntityName>,
    pub draft: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "record")]
pub enum FeatureRecord {
    Curve(CurveValue),
    /// The holes a hole feature made (by key), for drawings and the hole table.
    Holes {
        holes: Vec<crate::drawing::hole_table::HoleInstance>,
    },
    /// A cosmetic thread on a face (names of the face it resolved to).
    CosmeticThread {
        face: Vec<EntityName>,
        spec_designation: String,
        pitch: f64,
        depth: f64,
        length: Option<f64>,
    },
    /// A sheet body's rules and its section (base flanges), for later flanges and the flat.
    Sheet {
        rules: SheetRules,
        section: Option<Section>,
        depth: f64,
    },
    /// A flat pattern: the layout (outline and bend lines, for drawings and cutting) and, for
    /// a bent section, its development along the strip.
    FlatPattern {
        layout: FlatLayout,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        development: Option<Development>,
    },
    /// The bends a sketched bend or jog made in a flat plate (the base flange `base`), as bend
    /// lines in the plate's own frame: the plate keeps its developed outline.
    PlateBend {
        base: crate::ids::FeatureId,
        lines: Vec<FlatBendLine>,
        /// The part of the plate the bends moved, in the plate's frame: the half-plane
        /// `(p − origin)·normal ≥ 0`, for later bends to keep clear of.
        moved: ([f64; 2], [f64; 2]),
        /// The rigid motion that carries the moved part from the developed plate to where the
        /// bends put it (world coordinates).
        motion: crate::regen::kernel::Transform,
        /// The flat the bend moved, as a profile whose frame puts the sheet between `z = 0`
        /// and `z = t` (world coordinates): a later bend of that flat bends this profile.
        flat: crate::regen::kernel::CanonicalProfile,
        /// The bend whose moved flat this bend bent (none: the plate's unmoved part).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent: Option<crate::ids::FeatureId>,
    },
    /// The flanges an edge flange or hem grew on the edges of a flat plate (the base flange
    /// `base`), one per edge, developed onto the plate's flat pattern.
    PlateFlange {
        base: crate::ids::FeatureId,
        edges: Vec<PlateFlangeEdge>,
    },
    /// What a corner relief or closed corner changed at flange corners of a flat plate (the
    /// base flange `base`): each edit replaces a stretch of the developed outline.
    PlateCorner {
        base: crate::ids::FeatureId,
        edits: Vec<OutlineEdit>,
    },
    /// Structural members and their cut list.
    Weldment {
        cuts: Vec<MemberCut>,
        cut_list: Vec<CutListRow>,
    },
    /// Parting line and undercut analysis.
    Mold {
        /// Faces drafted towards the pull, away from it, and needing draft.
        positive: Vec<FaceDraft>,
        negative: Vec<FaceDraft>,
        insufficient: Vec<FaceDraft>,
        /// Faces that cannot release along the pull (blocked by other material).
        undercuts: Vec<Vec<EntityName>>,
        /// Edges between a positive and a negative face: the parting line.
        parting_edges: Vec<[EntityName; 2]>,
    },
}

/// One flange of a [`FeatureRecord::PlateFlange`], in the developed plate's frame (the base
/// flange's profile frame, metres). The flange grows from the outline's edge `start..end`
/// along `out`: the plate stops at `shift` past that edge line (negative when the flange sits
/// inside the edge), its bend zone spans the next `allowance`, and its flat the `flat` after.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlateFlangeEdge {
    /// The flange's flat region: `<feature>.<edge ordinal>`.
    pub key: String,
    pub start: [f64; 2],
    pub end: [f64; 2],
    pub out: [f64; 2],
    pub shift: f64,
    pub allowance: f64,
    pub flat: f64,
    pub line: FlatBendLine,
    /// The rigid motion that carries the developed flange flat to where the flange put it
    /// (world coordinates).
    pub motion: crate::regen::kernel::Transform,
    /// The flat region the flange grew from: a sketched bend's feature id, another flange's
    /// `key`, or none for the plate's unmoved part.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    /// The flange's section frame in the part (world coordinates): `x` out of the thickness
    /// face, `y` out of the sheet face the flange turns towards, `z` along the bend axis, the
    /// origin on the bend's plate-side tangent line.
    pub frame: runmat_geometry_kernel::frame::Frame,
}

/// A stretch of a plate's developed outline replaced: the outline runs from `from` to `to`
/// along `path` instead (in the base flange's profile frame, metres).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OutlineEdit {
    pub from: [f64; 2],
    pub to: [f64; 2],
    pub path: Vec<runmat_geometry_kernel::op::ProfileSegment>,
}

impl FeatureRecord {
    /// The thread spec of a cosmetic thread record, for drawings.
    pub fn thread_spec(&self, spec: &ThreadSpec) -> Option<String> {
        match self {
            FeatureRecord::CosmeticThread { .. } => Some(spec.designation.clone()),
            _ => None,
        }
    }
}
