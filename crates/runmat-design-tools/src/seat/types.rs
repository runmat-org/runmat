//! design.seat_part's input and its report.

use crate::edit::StagedChange;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SeatInput {
    /// Workspace path of the assembly.
    pub path: String,
    /// The occurrence to seat: one the assembly has, or the name of a new one when `part` is
    /// given.
    pub component: String,
    /// A part file to insert as `component` (its workspace path, `/mechanical/skid_clamp.part`).
    #[serde(default)]
    pub part: Option<String>,
    /// The inserted part's input values (`{ "name": "length", "value": "180 mm" }`).
    #[serde(default)]
    pub inputs: Option<Vec<InputValue>>,
    /// Where it goes. Leave it out with `through`: the member is fastened to the first holder.
    #[serde(default)]
    pub target: Option<SeatTarget>,
    /// Holders whose bores a straight member passes through, two or more: `leg_fore_left` or
    /// `leg_fore_left.<bore face>` each. The member goes on their common axis with its middle
    /// midway between the outermost bores (`offset` shifts it along the axis) and is fastened to
    /// the first; bores that aren't on one axis are refused with the seating that lines them up.
    #[serde(default)]
    pub through: Option<Vec<String>>,
    /// A component already seated whose mirror image this one becomes, across the target's
    /// plane of symmetry: a right leg as the left leg's mirror. Give target.component only.
    #[serde(default)]
    pub mirror: Option<String>,
    /// With `mirror`: the normal of the mirror plane (default `[0, 1, 0]`); the plane passes
    /// through the middle of the target's box.
    #[serde(default)]
    pub mirror_normal: Option<Vec<f64>>,
    /// The components the assembly stands on (the skid tubes): the result then carries the
    /// stance check of the assembly after this seating.
    #[serde(default)]
    pub contact: Option<Vec<String>>,
    /// The component's own face, side or frame that meets the target.
    #[serde(default)]
    pub own: Option<SeatOwn>,
    /// Gap left between the two, mm (negative presses in). Along an axis: the shift along it.
    #[serde(default)]
    pub offset: Option<f64>,
    /// Seat it the other way round (its face along the target's normal, or its axis reversed).
    #[serde(default)]
    pub flip: Option<bool>,
    /// Turn about the contact normal (or axis), degrees, after `along`.
    #[serde(default)]
    pub rotation: Option<f64>,
    /// The assembly direction the component's long axis runs along (`[1, 0, 0]`). A part with
    /// a bore or round face across the contact runs along its axis (a clamp along the tube it
    /// holds); otherwise its box's longest side across the contact.
    #[serde(default)]
    pub along: Option<Vec<f64>>,
    /// `fasten` (the default), `coincident` (planes) or `concentric` (axes).
    #[serde(default)]
    pub mate: Option<String>,
    /// A component already seated (a clamp, or the tube it holds) whose round face's axis this
    /// one's bore goes on: the component keeps the station along that axis that the target picks
    /// and stops short of the target instead of touching it, so a straight member runs through
    /// a set of clamps under a curved body. Its long axis runs along that axis unless `along`
    /// says otherwise. Only with a target face or ray.
    #[serde(default)]
    pub level_with: Option<String>,
    /// Neighbours to report the clearance to (occurrence names).
    #[serde(default)]
    pub clearance: Option<Vec<String>>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InputValue {
    pub name: String,
    pub value: String,
}

/// The target: `component` and one of `face`, `connector`, or `from` with `direction`.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SeatTarget {
    pub component: String,
    /// A face of the target's part: its name for people (`Body side 2`), reference or region.
    #[serde(default)]
    pub face: Option<String>,
    /// One of the assembly's connectors.
    #[serde(default)]
    pub connector: Option<String>,
    /// A point beside the target, mm, assembly coordinates; the ray from it along `direction`
    /// meets the target's surface where the component goes.
    #[serde(default)]
    pub from: Option<Vec<f64>>,
    /// The way the component approaches the target (`[0, 0, 1]`: up into a belly).
    #[serde(default)]
    pub direction: Option<Vec<f64>>,
}

/// The component's side of the contact: one of `face`, `side` or `connector`.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SeatOwn {
    #[serde(default)]
    pub face: Option<String>,
    /// A side of the part's box: `+x`, `-x`, `+y`, `-y`, `+z` (its top) or `-z`.
    #[serde(default)]
    pub side: Option<String>,
    /// A frame the part publishes (a `mate_connector` feature).
    #[serde(default)]
    pub connector: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub struct SeatOutput {
    #[serde(flatten)]
    pub change: StagedChange,
    pub seat: SeatReport,
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub struct SeatReport {
    /// One line: where it went, the mate, interference and the nearest neighbour.
    pub summary: String,
    pub component: String,
    pub target: String,
    /// The mate written and its kind (`fasten`, `planar`, `cylindrical`).
    pub mate: String,
    pub mate_kind: String,
    /// The mate after the solve: `satisfied`, `redundant`, `conflicting`, `not_converged`,
    /// `inactive`, `failed`, or `not solved` when the mate didn't resolve. The mates it clashes
    /// with, or why it failed, are in `notes`.
    pub mate_status: String,
    pub pose: Pose,
    /// Where the two meet, mm, assembly coordinates, with the target's normal (or axis) there.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contact: Option<ContactReport>,
    /// Overlaps with any other component; empty when it overlaps nothing it was measured
    /// against (a pair the kernel couldn't measure is named in the summary and the notes).
    pub interference: Vec<OverlapRow>,
    /// Gaps to the named neighbours (or the nearest few), mm.
    pub clearance: Vec<GapRow>,
    /// With `through`: each bore, its distance off the member's axis and its station along it
    /// from the member's middle, mm.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub through: Vec<ThroughRow>,
    /// With `contact`: design.inspect's stance section after the seating (`ok`, `summary`,
    /// `findings`, ground clearance, centre of mass, support, stance).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stance: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub struct ThroughRow {
    pub holder: String,
    pub bore: String,
    pub off_axis_mm: f64,
    pub station_mm: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub struct Pose {
    pub translation_mm: [f64; 3],
    pub rotation_axis: [f64; 3],
    pub rotation_deg: f64,
    /// `solved`, `floating`, `failed`, …
    pub status: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dof: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub struct ContactReport {
    pub point_mm: [f64; 3],
    pub normal: [f64; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub struct OverlapRow {
    pub with: String,
    pub volume_mm3: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub struct GapRow {
    pub with: String,
    /// The gap, when it is within `beyond_mm`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distance_mm: Option<f64>,
    /// Farther than this.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beyond_mm: Option<f64>,
    /// The kernel couldn't measure this pair (the reason is in `notes`): neither its gap nor
    /// its overlap is known.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub not_measured: bool,
}
