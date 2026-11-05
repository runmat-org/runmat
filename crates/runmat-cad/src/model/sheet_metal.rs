//! Sheet metal: a body kind with thickness, bend radius, a bend allowance rule (K-factor,
//! bend table, deduction or allowance) and gauges; flanges, hems, jogs, bends, corners,
//! reliefs, tabs and slots, form tools, unfold/fold, and the flat pattern.
//!
//! # Geometry
//!
//! A sheet is described by its *section*: a chain of flats and bends in a plane, swept along
//! the plane's normal. The sketch chain is one mold line of the sheet; the material lies on one
//! side of it ([`MaterialSide`]). At each corner of the chain a bend of inner radius `R` turns
//! the sheet; the flats are what is left of each segment between the bends' tangent lines.
//! [`section`] computes flats, bends and the closed outline; the engine lowers the outline to
//! labelled tools (faces tagged by the sketch curve or bend they come from, so references
//! survive edits and unfold/fold), and [`develop`] derives the flat pattern.
//!
//! # Developed length
//!
//! For a bend of angle `θ` (the change of direction), inner radius `R`, thickness `t` and
//! K-factor `K` (the neutral axis at `R + K t`):
//!
//! - bend allowance `BA = θ (R + K t)`;
//! - outside setback `OSSB = tan(θ/2) (R + t)` (θ ≤ 180°);
//! - bend deduction `BD = 2 OSSB − BA`.
//!
//! A bend table gives `BA` directly (interpolated in angle for the sheet's thickness and
//! radius); a deduction rule gives `BD`, from which `BA = 2 OSSB − BD`.

use crate::ids::{FeatureId, SketchEntityId};
use crate::model::datum::{AxisRef, PointRef};
use crate::model::feature::ProfileRef;
use crate::model::visit::{Params, Visit};
use crate::refs::Reference;
use crate::value::ValueExpr;
use serde::{Deserialize, Serialize};

/// How bends develop.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "rule")]
pub enum BendRule {
    KFactor {
        k: ValueExpr,
    },
    /// Bend allowance from a table (by thickness, radius and angle).
    Table {
        table: BendTable,
    },
    /// A fixed bend deduction per 90° bend, scaled by angle.
    Deduction {
        value: ValueExpr,
    },
    /// A fixed bend allowance per 90° bend, scaled by angle.
    Allowance {
        value: ValueExpr,
    },
}

/// A bend table: allowance rows, SI.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BendTable {
    pub name: String,
    pub rows: Vec<BendTableRow>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BendTableRow {
    pub thickness: f64,
    pub radius: f64,
    /// Bend angle (radians).
    pub angle: f64,
    pub allowance: f64,
}

impl BendTable {
    /// The allowance for this thickness and radius at `angle`, interpolated linearly between
    /// the two nearest tabulated angles. `None` when the table has no row for the pair.
    pub fn allowance(&self, thickness: f64, radius: f64, angle: f64) -> Option<f64> {
        let tol = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1e-6);
        let mut rows: Vec<&BendTableRow> = self
            .rows
            .iter()
            .filter(|r| tol(r.thickness, thickness) && tol(r.radius, radius))
            .collect();
        rows.sort_by(|a, b| a.angle.partial_cmp(&b.angle).expect("finite"));
        match rows.len() {
            0 => None,
            1 => Some(rows[0].allowance * angle / rows[0].angle),
            _ => {
                let i = rows
                    .windows(2)
                    .position(|w| angle <= w[1].angle)
                    .unwrap_or(rows.len() - 2);
                let (a, b) = (rows[i], rows[i + 1]);
                let t = (angle - a.angle) / (b.angle - a.angle);
                Some(a.allowance + t * (b.allowance - a.allowance))
            }
        }
    }
}

/// A gauge table: thickness, default bend radius and K-factor by gauge name.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GaugeTable {
    pub name: String,
    pub rows: Vec<GaugeRow>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GaugeRow {
    pub gauge: String,
    pub thickness: f64,
    pub bend_radius: f64,
    pub k_factor: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "relief")]
pub enum ReliefKind {
    Rectangular { width: ValueExpr, depth: ValueExpr },
    Obround { width: ValueExpr, depth: ValueExpr },
    Tear,
    None,
}

/// The sheet metal rules a body carries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SheetMetalRules {
    pub thickness: ValueExpr,
    /// Inner bend radius.
    pub bend_radius: ValueExpr,
    pub rule: BendRule,
    /// The gauge chosen from a table (thickness and radius then come from it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gauge: Option<(GaugeTable, String)>,
    #[serde(default = "default_relief")]
    pub relief: ReliefKind,
}

fn default_relief() -> ReliefKind {
    ReliefKind::Tear
}

/// Which side of the sketch chain the material lies on (by the chain's left normal).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialSide {
    #[default]
    Left,
    Right,
    Mid,
}

/// Where a flange sits relative to the edge it grows from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlangePosition {
    /// The flange's outer face is flush with the original edge (the sheet trims back).
    MaterialInside,
    /// The flange's inner face is flush with the original edge.
    MaterialOutside,
    /// The bend starts at the original edge (nothing trims).
    #[default]
    BendOutside,
}

/// How a flange length is measured.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LengthReference {
    /// To the virtual sharp of the outer faces.
    #[default]
    OuterVirtualSharp,
    InnerVirtualSharp,
    /// From the bend's tangent line (the flat length).
    Tangent,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "hem")]
pub enum HemKind {
    Closed,
    Open {
        gap: ValueExpr,
    },
    /// A roll of more than 180° at inner radius `radius`, then the hem's `length` as a
    /// straight leg running back towards the sheet.
    Teardrop {
        radius: ValueExpr,
        angle: ValueExpr,
    },
    /// A roll of `angle` at inner radius `radius`; the hem's `length` is a straight leg after
    /// it (zero for a plain curl).
    Rolled {
        radius: ValueExpr,
        angle: ValueExpr,
    },
}

/// How a closed corner's two flats meet. The picked flange `A` and the flange `B` it meets
/// both stand at 90°; each flat extends along its bend line towards the other:
///
/// - `overlap`: `A` runs to `B`'s outer face, covering `B`'s end; `B` stops `gap` short of
///   `A`'s inner face;
/// - `underlap`: `A` stops `gap` short of `B`'s inner face; `B` runs to `A`'s outer face;
/// - `butt`: each stops `gap` short of the other's inner face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CornerOverlap {
    Overlap,
    Underlap,
    Butt,
}

/// A form tool's shape, in the frame of the face it is stamped on: `x` along the tool
/// (the face frame's `x` turned by the feature's rotation), `z` out of the face, the origin
/// at the stamp point. Each form keeps the sheet's thickness `t`:
///
/// - `louver`: a lance `length` × `width` (`0 ≤ y ≤ width`) whose material rises as a hood
///   hinged at `y = 0` to `height` at `y = width`, open at the lance and at both ends;
/// - `dimple`: a round cup of outer `diameter` rising `height` out of the face;
/// - `bead`: a rectangular rib `length` × `width` rising `height` out of the face, closed at
///   its ends.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "form")]
pub enum FormToolShape {
    Louver {
        length: ValueExpr,
        width: ValueExpr,
        height: ValueExpr,
    },
    Dimple {
        diameter: ValueExpr,
        height: ValueExpr,
    },
    Bead {
        length: ValueExpr,
        width: ValueExpr,
        height: ValueExpr,
    },
}

/// A corner relief, placed at the corner `P` where the plate-side tangent lines of two bends
/// meet on the flat pattern. Along each bend the relief removes the bend zone's last `size`
/// next to the corner; on the plate it removes:
///
/// - `rectangular`: the `size` × `size` square at `P`;
/// - `round`: the quarter disc of radius `size` about `P`, so the relief's inner end is a
///   rounded and tangent to both bend zones' cuts;
/// - `tear`: nothing. The bends are ripped along their tangent lines up to `P`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "relief")]
pub enum CornerReliefKind {
    Rectangular { size: ValueExpr },
    Round { size: ValueExpr },
    Tear,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "break")]
pub enum BreakKind {
    Chamfer { distance: ValueExpr },
    Fillet { radius: ValueExpr },
}

// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "sheet_metal")]
pub enum SheetMetalFeature {
    /// The first flange: a closed profile makes a flat plate; an open chain makes a bent
    /// section swept `depth` along the sketch normal (one flat per segment, one bend per
    /// corner).
    BaseFlange {
        profile: ProfileRef,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        depth: Option<ValueExpr>,
        #[serde(default)]
        side: MaterialSide,
        rules: SheetMetalRules,
    },
    /// Flanges on sheet edges: each grows from the thickness face along the edge.
    EdgeFlange {
        edges: Vec<Reference>,
        length: ValueExpr,
        angle: ValueExpr,
        #[serde(default)]
        length_reference: LengthReference,
        #[serde(default)]
        position: FlangePosition,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        radius: Option<ValueExpr>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        relief: Option<ReliefKind>,
    },
    /// A profile swept along a chain of sheet edges, mitred at the corners.
    MiterFlange {
        profile: ProfileRef,
        edges: Vec<Reference>,
        #[serde(default = "zero")]
        gap: ValueExpr,
    },
    Hem {
        edges: Vec<Reference>,
        kind: HemKind,
        length: ValueExpr,
    },
    /// Two bends offsetting part of a flat along a sketch line.
    Jog {
        face: Reference,
        line: (FeatureId, SketchEntityId),
        offset: ValueExpr,
        angle: ValueExpr,
    },
    /// A bend of a flat along a sketch line.
    SketchedBend {
        face: Reference,
        line: (FeatureId, SketchEntityId),
        angle: ValueExpr,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        radius: Option<ValueExpr>,
        /// Which side of the line stays put.
        fixed: crate::refs::name::PlaneSide,
    },
    /// A transition between two open profiles on parallel planes, formed in `facets` bends.
    LoftedBend {
        profiles: [ProfileRef; 2],
        facets: u32,
        rules: SheetMetalRules,
    },
    /// A thin solid recognised as sheet metal: the fixed face stays, bend edges become bends,
    /// rip edges become gaps.
    ConvertToSheetMetal {
        body: Reference,
        fixed_face: Reference,
        #[serde(default)]
        bend_edges: Vec<Reference>,
        #[serde(default)]
        rip_edges: Vec<Reference>,
        rules: SheetMetalRules,
    },
    /// Extends the flats of the flanges whose faces are picked, and of the flange each one
    /// meets at a corner, to close that corner (see [`CornerOverlap`]). A face of the plate
    /// closes every corner of the plate, each with its earlier flange as the picked one.
    ClosedCorner {
        faces: Vec<Reference>,
        gap: ValueExpr,
        overlap: CornerOverlap,
    },
    BreakCorner {
        edges: Vec<Reference>,
        kind: BreakKind,
    },
    /// Reliefs at corners where two flanges' bends meet. A vertex or edge picks the corner
    /// nearest it, a face of a flange each corner of that flange, and a face of the plate
    /// every corner of the plate.
    CornerRelief {
        corners: Vec<Reference>,
        relief: CornerReliefKind,
    },
    /// Tabs on one sheet's edge and matching slots in another's face.
    TabAndSlot {
        edge: Reference,
        slot_face: Reference,
        count: ValueExpr,
        length: ValueExpr,
        #[serde(default = "zero")]
        clearance: ValueExpr,
    },
    /// A form tool stamped at a point of a flat of a plate, rising out of that face.
    FormTool {
        tool: FormToolShape,
        face: Reference,
        at: PointRef,
        #[serde(default = "zero_angle")]
        rotation: ValueExpr,
    },
    /// Flatten the named bends (all, when empty) about the fixed face.
    Unfold {
        fixed_face: Reference,
        #[serde(default)]
        bends: Vec<Reference>,
    },
    /// Re-fold bends an earlier unfold flattened.
    Fold {
        #[serde(default)]
        bends: Vec<Reference>,
    },
    /// The flat pattern: a derived body plus bend lines and the bend table.
    FlatPattern {
        fixed_face: Reference,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        grain: Option<AxisRef>,
    },
}

fn zero() -> ValueExpr {
    ValueExpr::mm(0.0)
}

fn zero_angle() -> ValueExpr {
    ValueExpr::deg(0.0)
}

impl SheetMetalRules {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        v.value(&self.thickness);
        v.value(&self.bend_radius);
        match &self.rule {
            BendRule::KFactor { k } => v.value(k),
            BendRule::Deduction { value } | BendRule::Allowance { value } => v.value(value),
            BendRule::Table { .. } => {}
        }
        relief_values(&self.relief, v);
    }
}

fn relief_values<'a>(r: &'a ReliefKind, v: &mut Visit<'a>) {
    if let ReliefKind::Rectangular { width, depth } | ReliefKind::Obround { width, depth } = r {
        v.value(width);
        v.value(depth);
    }
}

impl Params for SheetMetalFeature {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        use SheetMetalFeature as S;
        match self {
            S::BaseFlange {
                profile,
                depth,
                rules,
                ..
            } => {
                v.profile(profile);
                v.opt_value(depth);
                rules.visit(v);
            }
            S::EdgeFlange {
                edges,
                length,
                angle,
                radius,
                relief,
                ..
            } => {
                v.references("edges", edges);
                v.value(length);
                v.value(angle);
                v.opt_value(radius);
                if let Some(r) = relief {
                    relief_values(r, v);
                }
            }
            S::MiterFlange {
                profile,
                edges,
                gap,
            } => {
                v.profile(profile);
                v.references("edges", edges);
                v.value(gap);
            }
            S::Hem {
                edges,
                kind,
                length,
            } => {
                v.references("edges", edges);
                v.value(length);
                match kind {
                    HemKind::Closed => {}
                    HemKind::Open { gap } => v.value(gap),
                    HemKind::Teardrop { radius, angle } | HemKind::Rolled { radius, angle } => {
                        v.value(radius);
                        v.value(angle);
                    }
                }
            }
            S::Jog {
                face,
                line,
                offset,
                angle,
            } => {
                v.reference("face", face);
                v.feature(&line.0);
                v.value(offset);
                v.value(angle);
            }
            S::SketchedBend {
                face,
                line,
                angle,
                radius,
                ..
            } => {
                v.reference("face", face);
                v.feature(&line.0);
                v.value(angle);
                v.opt_value(radius);
            }
            S::LoftedBend {
                profiles, rules, ..
            } => {
                v.profile(&profiles[0]);
                v.profile(&profiles[1]);
                rules.visit(v);
            }
            S::ConvertToSheetMetal {
                body,
                fixed_face,
                bend_edges,
                rip_edges,
                rules,
            } => {
                v.reference("body", body);
                v.reference("fixed_face", fixed_face);
                v.references("bend_edges", bend_edges);
                v.references("rip_edges", rip_edges);
                rules.visit(v);
            }
            S::ClosedCorner { faces, gap, .. } => {
                v.references("faces", faces);
                v.value(gap);
            }
            S::BreakCorner { edges, kind } => {
                v.references("edges", edges);
                match kind {
                    BreakKind::Chamfer { distance } => v.value(distance),
                    BreakKind::Fillet { radius } => v.value(radius),
                }
            }
            S::CornerRelief { corners, relief } => {
                v.references("corners", corners);
                if let CornerReliefKind::Rectangular { size } | CornerReliefKind::Round { size } =
                    relief
                {
                    v.value(size);
                }
            }
            S::TabAndSlot {
                edge,
                slot_face,
                count,
                length,
                clearance,
            } => {
                v.reference("edge", edge);
                v.reference("slot_face", slot_face);
                v.value(count);
                v.value(length);
                v.value(clearance);
            }
            S::FormTool {
                tool,
                face,
                at,
                rotation,
            } => {
                match tool {
                    FormToolShape::Louver {
                        length,
                        width,
                        height,
                    }
                    | FormToolShape::Bead {
                        length,
                        width,
                        height,
                    } => {
                        v.value(length);
                        v.value(width);
                        v.value(height);
                    }
                    FormToolShape::Dimple { diameter, height } => {
                        v.value(diameter);
                        v.value(height);
                    }
                }
                v.reference("face", face);
                v.point("at", at);
                v.value(rotation);
            }
            S::Unfold { fixed_face, bends } => {
                v.reference("fixed_face", fixed_face);
                v.references("bends", bends);
            }
            S::Fold { bends } => v.references("bends", bends),
            S::FlatPattern { fixed_face, grain } => {
                v.reference("fixed_face", fixed_face);
                if let Some(g) = grain {
                    v.axis("grain", g);
                }
            }
        }
    }

    fn kind_name(&self) -> &'static str {
        use SheetMetalFeature as S;
        match self {
            S::BaseFlange { .. } => "sheet_metal.base_flange",
            S::EdgeFlange { .. } => "sheet_metal.edge_flange",
            S::MiterFlange { .. } => "sheet_metal.miter_flange",
            S::Hem { .. } => "sheet_metal.hem",
            S::Jog { .. } => "sheet_metal.jog",
            S::SketchedBend { .. } => "sheet_metal.sketched_bend",
            S::LoftedBend { .. } => "sheet_metal.lofted_bend",
            S::ConvertToSheetMetal { .. } => "sheet_metal.convert",
            S::ClosedCorner { .. } => "sheet_metal.closed_corner",
            S::BreakCorner { .. } => "sheet_metal.break_corner",
            S::CornerRelief { .. } => "sheet_metal.corner_relief",
            S::TabAndSlot { .. } => "sheet_metal.tab_and_slot",
            S::FormTool { .. } => "sheet_metal.form_tool",
            S::Unfold { .. } => "sheet_metal.unfold",
            S::Fold { .. } => "sheet_metal.fold",
            S::FlatPattern { .. } => "sheet_metal.flat_pattern",
        }
    }

    fn new_body(&self) -> bool {
        matches!(
            self,
            SheetMetalFeature::BaseFlange { .. }
                | SheetMetalFeature::LoftedBend { .. }
                | SheetMetalFeature::FlatPattern { .. }
        )
    }
}

mod develop;
pub use develop::*;
