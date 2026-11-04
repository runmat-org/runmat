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

impl FeatureRecord {
    /// The thread spec of a cosmetic thread record, for drawings.
    pub fn thread_spec(&self, spec: &ThreadSpec) -> Option<String> {
        match self {
            FeatureRecord::CosmeticThread { .. } => Some(spec.designation.clone()),
            _ => None,
        }
    }
}
