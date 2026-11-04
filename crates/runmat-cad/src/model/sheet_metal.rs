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
use std::f64::consts::PI;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CornerOverlap {
    Overlap,
    Underlap,
    Butt,
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
    ClosedCorner {
        faces: Vec<Reference>,
        gap: ValueExpr,
        overlap: CornerOverlap,
    },
    BreakCorner {
        edges: Vec<Reference>,
        kind: BreakKind,
    },
    CornerRelief {
        corners: Vec<Reference>,
        kind: ReliefKind,
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
    /// A library form tool (louvre, lance, dimple) stamped at a point of a face.
    FormTool {
        tool: String,
        face: Reference,
        at: PointRef,
        #[serde(default = "zero")]
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
            S::CornerRelief { corners, kind } => {
                v.references("corners", corners);
                relief_values(kind, v);
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
                face, at, rotation, ..
            } => {
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

// ------------------------------------------------------------------------- the section

/// A flat of a section: segment `curve` of the chain, from `start` to `end` along it (the
/// stretch between the neighbouring bends' tangent lines), in sketch coordinates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SectionFlat {
    pub curve: SketchEntityId,
    /// The flat's reference-side line (on the chain) and its direction.
    pub start: [f64; 2],
    pub end: [f64; 2],
    /// Unit normal towards the material's far side.
    pub normal: [f64; 2],
    /// Offsets of the two faces from the chain, along `normal` (`near ≤ far`).
    pub near: f64,
    pub far: f64,
}

impl SectionFlat {
    pub fn length(&self) -> f64 {
        (self.end[0] - self.start[0]).hypot(self.end[1] - self.start[1])
    }
}

/// A bend of a section between two consecutive flats.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SectionBend {
    /// `"<curve before>+<curve after>"`: the bend's key from the definition.
    pub key: String,
    pub center: [f64; 2],
    pub inner_radius: f64,
    pub thickness: f64,
    /// The change of direction, signed (positive turns left), radians.
    pub angle: f64,
    /// Direction angles (from the centre) where the bend starts and ends, counter-clockwise
    /// order.
    pub from: f64,
    pub to: f64,
}

impl SectionBend {
    pub fn bend_key(a: &SketchEntityId, b: &SketchEntityId) -> String {
        format!("{}+{}", a, b)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Section {
    pub flats: Vec<SectionFlat>,
    pub bends: Vec<SectionBend>,
    pub thickness: f64,
}

/// The section of a sheet along an open polyline chain: `points[i]..points[i+1]` is curve
/// `curves[i]`. Material on `side` of the chain, thickness `t`, inner bend radius `r`.
pub fn section(
    points: &[[f64; 2]],
    curves: &[SketchEntityId],
    side: MaterialSide,
    t: f64,
    r: f64,
) -> Result<Section, String> {
    if points.len() < 2 || curves.len() + 1 != points.len() {
        return Err("a sheet section needs a chain of at least one line".into());
    }
    if t <= 0.0 || r < 0.0 {
        return Err("sheet thickness must be positive and the bend radius non-negative".into());
    }
    let (o0, o1) = match side {
        MaterialSide::Left => (0.0, t),
        MaterialSide::Right => (-t, 0.0),
        MaterialSide::Mid => (-t / 2.0, t / 2.0),
    };
    let dir = |i: usize| {
        let (a, b) = (points[i], points[i + 1]);
        let l = (b[0] - a[0]).hypot(b[1] - a[1]);
        [(b[0] - a[0]) / l, (b[1] - a[1]) / l]
    };
    let left = |d: [f64; 2]| [-d[1], d[0]];
    let n = curves.len();
    // Parameters along each segment where its flat starts and ends.
    let mut s0 = vec![0.0; n];
    let mut s1: Vec<f64> = (0..n)
        .map(|i| (points[i + 1][0] - points[i][0]).hypot(points[i + 1][1] - points[i][1]))
        .collect();
    let mut bends = Vec::new();
    for k in 1..n {
        let (da, db) = (dir(k - 1), dir(k));
        let cross = da[0] * db[1] - da[1] * db[0];
        let dotp = da[0] * db[0] + da[1] * db[1];
        let phi = cross.atan2(dotp);
        if phi.abs() < 1e-12 {
            continue; // collinear segments: no bend
        }
        if phi.abs() >= PI - 1e-9 {
            return Err("a sheet cannot fold back on itself in one bend; use a hem".into());
        }
        // The centre lies on the inner side, at offset c from both segment lines.
        let c = if phi > 0.0 { o1 + r } else { o0 - r };
        // Corner of the two lines offset by c.
        let v = points[k];
        let (na, nb) = (left(da), left(db));
        // Solve v + c·na + s·da = v + c·nb + u·db for the intersection.
        let pa = [v[0] + c * na[0], v[1] + c * na[1]];
        let pb = [v[0] + c * nb[0], v[1] + c * nb[1]];
        let den = da[0] * db[1] - da[1] * db[0];
        let w = [pb[0] - pa[0], pb[1] - pa[1]];
        let s = (w[0] * db[1] - w[1] * db[0]) / den;
        let center = [pa[0] + s * da[0], pa[1] + s * da[1]];
        // Tangent parameters: the centre's foot on each segment line.
        let foot = |i: usize| {
            let d = dir(i);
            (center[0] - points[i][0]) * d[0] + (center[1] - points[i][1]) * d[1]
        };
        s1[k - 1] = foot(k - 1);
        s0[k] = foot(k);
        let inner = c.abs() - (if phi > 0.0 { o1 } else { -o0 }).abs();
        let inner = if phi > 0.0 {
            c - o1
        } else {
            let _ = inner;
            o0 - c
        };
        // From the centre, the chain point of segment k-1's end lies at angle of -c·na.
        let ang = |n: [f64; 2], c: f64| (-c * n[1]).atan2(-c * n[0]);
        let (a0, a1) = (ang(na, c), ang(nb, c));
        let (from, to) = if phi > 0.0 { (a0, a1) } else { (a1, a0) };
        let to = if to < from { to + 2.0 * PI } else { to };
        bends.push(SectionBend {
            key: SectionBend::bend_key(&curves[k - 1], &curves[k]),
            center,
            inner_radius: inner,
            thickness: t,
            angle: phi,
            from,
            to,
        });
    }
    let mut flats = Vec::new();
    for i in 0..n {
        if s1[i] < s0[i] - 1e-12 {
            return Err(format!(
                "segment `{}` is shorter than its bends' setbacks",
                curves[i]
            ));
        }
        let d = dir(i);
        let p = points[i];
        flats.push(SectionFlat {
            curve: curves[i].clone(),
            start: [p[0] + s0[i] * d[0], p[1] + s0[i] * d[1]],
            end: [p[0] + s1[i] * d[0], p[1] + s1[i] * d[1]],
            normal: left(d),
            near: o0,
            far: o1,
        });
    }
    Ok(Section {
        flats,
        bends,
        thickness: t,
    })
}

// ------------------------------------------------------------------- developed lengths

/// The bend rule evaluated (SI).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "rule")]
pub enum BendRuleValue {
    KFactor {
        k: f64,
    },
    Table {
        table: BendTable,
    },
    /// Deduction for a 90° bend.
    Deduction {
        value: f64,
    },
    /// Allowance for a 90° bend.
    Allowance {
        value: f64,
    },
}

/// Bend allowance for one bend (`angle` unsigned, radians).
pub fn bend_allowance(rule: &BendRuleValue, t: f64, r: f64, angle: f64) -> Result<f64, String> {
    let a = angle.abs();
    Ok(match rule {
        BendRuleValue::KFactor { k } => {
            if !(0.0..=1.0).contains(k) {
                return Err(format!("K-factor {} is outside [0, 1]", k));
            }
            a * (r + k * t)
        }
        BendRuleValue::Table { table } => table.allowance(t, r, a).ok_or_else(|| {
            format!(
                "bend table `{}` has no row for thickness {} m and radius {} m",
                table.name, t, r
            )
        })?,
        BendRuleValue::Allowance { value } => value * a / (PI / 2.0),
        BendRuleValue::Deduction { value } => {
            2.0 * outside_setback(t, r, a) - value * a / (PI / 2.0)
        }
    })
}

/// `tan(θ/2)(R + t)`.
pub fn outside_setback(t: f64, r: f64, angle: f64) -> f64 {
    (angle.abs() / 2.0).tan() * (r + t)
}

/// `2·OSSB − BA`.
pub fn bend_deduction(rule: &BendRuleValue, t: f64, r: f64, angle: f64) -> Result<f64, String> {
    Ok(2.0 * outside_setback(t, r, angle) - bend_allowance(rule, t, r, angle)?)
}

/// Whether a bend turns up or down when the flat is viewed from the reference side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BendDirection {
    Up,
    Down,
}

/// A bend line of the flat pattern: the centre of the bend zone, at `position` along the
/// developed length from the first flat's start.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BendLine {
    pub key: String,
    pub position: f64,
    /// The bend zone: `[position − allowance/2, position + allowance/2]`.
    pub allowance: f64,
    pub deduction: f64,
    pub angle: f64,
    pub inner_radius: f64,
    pub direction: BendDirection,
}

/// One region of the developed strip, in order: a flat or a bend zone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "region")]
pub enum StripRegion {
    Flat {
        curve: SketchEntityId,
        start: f64,
        length: f64,
    },
    Bend {
        key: String,
        start: f64,
        length: f64,
    },
}

/// The developed section: total length, regions in order, bend lines and the bend table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Development {
    pub length: f64,
    pub regions: Vec<StripRegion>,
    pub bend_lines: Vec<BendLine>,
}

/// Develop a section under a bend rule.
pub fn develop(section: &Section, rule: &BendRuleValue) -> Result<Development, String> {
    let mut x = 0.0;
    let mut regions = Vec::new();
    let mut bend_lines = Vec::new();
    for (i, f) in section.flats.iter().enumerate() {
        let len = f.length();
        regions.push(StripRegion::Flat {
            curve: f.curve.clone(),
            start: x,
            length: len,
        });
        x += len;
        if let Some(b) = section.bends.get(i) {
            let ba = bend_allowance(rule, section.thickness, b.inner_radius, b.angle)?;
            let bd = bend_deduction(rule, section.thickness, b.inner_radius, b.angle)?;
            regions.push(StripRegion::Bend {
                key: b.key.clone(),
                start: x,
                length: ba,
            });
            bend_lines.push(BendLine {
                key: b.key.clone(),
                position: x + ba / 2.0,
                allowance: ba,
                deduction: bd,
                angle: b.angle.abs(),
                inner_radius: b.inner_radius,
                direction: if b.angle > 0.0 {
                    BendDirection::Up
                } else {
                    BendDirection::Down
                },
            });
            x += ba;
        }
    }
    if section.bends.len() + 1 != section.flats.len() {
        return Err("a section needs one bend between each pair of flats".into());
    }
    Ok(Development {
        length: x,
        regions,
        bend_lines,
    })
}
