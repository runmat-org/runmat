//! Sketches: entities, constraints and the `solved { }` cache, behind a solver seam.

use crate::ids::{ConstraintId, SketchEntityId};
use crate::model::datum::PlaneRef;
use crate::refs::reference::Reference;
use crate::value::ValueExpr;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sketch {
    pub plane: PlaneRef,
    pub entities: Vec<SketchEntity>,
    #[serde(default)]
    pub constraints: Vec<SketchConstraint>,
    /// The `solved { }` block: the last solver output, so a file reloads without re-solving.
    #[serde(default)]
    pub solved: SolvedCache,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SketchEntity {
    pub id: SketchEntityId,
    pub geometry: SketchGeometry,
    #[serde(default)]
    pub construction: bool,
    /// Converted geometry (Use edges, Offset of a component edge): re-read from what it was
    /// converted from on each regeneration and held fixed by the solver. The written geometry
    /// is its last position.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projected: Option<SketchProjection>,
}

/// Where a converted entity's geometry comes from, read again on every regeneration and
/// carried onto the sketch's plane along its normal.
///
/// In an assembly: an edge set a component publishes. In a part: one of the part's own edges or
/// faces, as it stands when the sketch regenerates (features before the sketch only).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "from")]
pub enum SketchProjection {
    /// The edges component `occurrence` (a dotted occurrence path in the assembly) publishes as
    /// `published`. In the file: `from = published(occurrence = upper, name = upper_e1)`.
    Published {
        occurrence: String,
        published: String,
    },
    /// A part edge projected onto the sketch plane: `from = project(<edge>)`.
    Project { edge: Reference },
    /// One outline chain of a curved face seen along the sketch normal (where the surface turns
    /// edge-on): `from = silhouette(<face>, chain = 0)`.
    Silhouette {
        face: Reference,
        #[serde(default)]
        chain: u32,
    },
    /// One chain of the curve where a face crosses the sketch plane:
    /// `from = intersection(<face>, chain = 0)`.
    Intersection {
        face: Reference,
        #[serde(default)]
        chain: u32,
    },
    /// A part edge offset `distance` across planar face `face` (into the face for a positive
    /// distance), then projected: `from = offset_on_surface(<edge>, face = <face>, distance = 2 mm)`.
    OffsetOnSurface {
        edge: Reference,
        face: Reference,
        distance: ValueExpr,
    },
}

impl SketchProjection {
    /// Whether this source is one an assembly sketch reads (a published edge set).
    pub fn is_published(&self) -> bool {
        matches!(self, SketchProjection::Published { .. })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "entity")]
pub enum SketchGeometry {
    Point {
        at: PointExpr,
    },
    Line {
        start: PointExpr,
        end: PointExpr,
    },
    /// Counter-clockwise from `start` to `end` about `center`.
    Arc {
        center: PointExpr,
        start: PointExpr,
        end: PointExpr,
    },
    Circle {
        center: PointExpr,
        radius: ValueExpr,
    },
    /// An interpolating cubic through `points` in order. Its points are `start` (the first),
    /// `p1` … `p<n-2>` and `end` (the last); a closed spline runs back to `start` and has no
    /// `end` (its last point is `p<n-1>`).
    Spline {
        points: Vec<PointExpr>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        closed: bool,
    },
    /// An ellipse about `center` with semi-axes `major` and `minor`, its major axis turned
    /// `angle` counter-clockwise from the sketch's x axis (0 when not written). With `start` and
    /// `end` it is the elliptical arc counter-clockwise from `start` to `end`, both on the
    /// ellipse. The written semi-axes and angle are starting values, as a circle's radius is;
    /// the `major_radius` and `minor_radius` dimensions hold them.
    Ellipse {
        center: PointExpr,
        major: ValueExpr,
        minor: ValueExpr,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        angle: Option<ValueExpr>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        start: Option<PointExpr>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        end: Option<PointExpr>,
    },
    /// Lettering: `text` set from `at` (the left end of the first line's baseline), `height`
    /// the capital height, turned `angle` counter-clockwise (0 when not written). In the
    /// single-line `font` each glyph is a pen stroke `stroke` wide (0.15 of the height when not
    /// written); its outline is the profile that extrudes and cuts.
    Text {
        text: String,
        at: PointExpr,
        height: ValueExpr,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        angle: Option<ValueExpr>,
        #[serde(default)]
        font: TextFont,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stroke: Option<ValueExpr>,
    },
}

/// The typeface sketch text is set in. Only the single-line font exists today; the name is
/// written in the file so other faces can join without a format change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextFont {
    /// The original single-line (monoline) glyph set the circuit legend also uses
    /// ([`runmat_types::font`]).
    #[default]
    SingleLine,
}

impl TextFont {
    pub fn word(self) -> &'static str {
        match self {
            TextFont::SingleLine => "single_line",
        }
    }

    pub fn from_word(word: &str) -> Option<Self> {
        (word == "single_line").then_some(TextFont::SingleLine)
    }
}

impl SketchGeometry {
    /// The entity's kind as the display and the Outline name it: `point`, `line`, `arc`,
    /// `circle`, `spline`, `ellipse` or `text`.
    pub fn kind_word(&self) -> &'static str {
        match self {
            SketchGeometry::Point { .. } => "point",
            SketchGeometry::Line { .. } => "line",
            SketchGeometry::Arc { .. } => "arc",
            SketchGeometry::Circle { .. } => "circle",
            SketchGeometry::Spline { .. } => "spline",
            SketchGeometry::Ellipse { .. } => "ellipse",
            SketchGeometry::Text { .. } => "text",
        }
    }

    /// The roles of a spline's points, in order.
    pub fn spline_roles(count: usize, closed: bool) -> Vec<PointRole> {
        (0..count)
            .map(|i| match i {
                0 => PointRole::Start,
                i if i + 1 == count && !closed => PointRole::End,
                i => PointRole::Point(i as u16),
            })
            .collect()
    }

    /// Every defining point of the entity with its role, in order.
    pub fn points(&self) -> Vec<(PointRole, &PointExpr)> {
        match self {
            SketchGeometry::Point { at } => vec![(PointRole::At, at)],
            SketchGeometry::Line { start, end } => {
                vec![(PointRole::Start, start), (PointRole::End, end)]
            }
            SketchGeometry::Arc { center, start, end } => vec![
                (PointRole::Center, center),
                (PointRole::Start, start),
                (PointRole::End, end),
            ],
            SketchGeometry::Circle { center, .. } => vec![(PointRole::Center, center)],
            SketchGeometry::Spline { points, closed } => Self::spline_roles(points.len(), *closed)
                .into_iter()
                .zip(points)
                .collect(),
            SketchGeometry::Ellipse {
                center, start, end, ..
            } => {
                let mut points = vec![(PointRole::Center, center)];
                if let (Some(start), Some(end)) = (start, end) {
                    points.push((PointRole::Start, start));
                    points.push((PointRole::End, end));
                }
                points
            }
            SketchGeometry::Text { at, .. } => vec![(PointRole::At, at)],
        }
    }

    /// The expression of the point with `role`.
    pub fn point(&self, role: PointRole) -> Option<&PointExpr> {
        self.points()
            .into_iter()
            .find(|(r, _)| *r == role)
            .map(|(_, p)| p)
    }
}

/// Which defining point of an entity. Written and serialized as a word: `at`, `start`,
/// `end`, `center`, or `p<i>` for a spline's interior point `i`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PointRole {
    /// A point entity's own position.
    At,
    Start,
    End,
    Center,
    /// A spline's through-point by index (the first is `Start`, the last of an open spline
    /// `End`).
    Point(u16),
}

impl PointRole {
    pub fn word(self) -> String {
        match self {
            PointRole::At => "at".into(),
            PointRole::Start => "start".into(),
            PointRole::End => "end".into(),
            PointRole::Center => "center".into(),
            PointRole::Point(i) => format!("p{i}"),
        }
    }

    pub fn from_word(word: &str) -> Option<PointRole> {
        Some(match word {
            "at" => PointRole::At,
            "start" => PointRole::Start,
            "end" => PointRole::End,
            "center" => PointRole::Center,
            _ => {
                let digits = word.strip_prefix('p')?;
                if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                    return None;
                }
                PointRole::Point(digits.parse().ok()?)
            }
        })
    }
}

impl Serialize for PointRole {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.word())
    }
}

impl<'de> Deserialize<'de> for PointRole {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let word = String::deserialize(d)?;
        PointRole::from_word(&word)
            .ok_or_else(|| serde::de::Error::custom(format!("`{word}` is not a sketch point")))
    }
}

/// A sketch point as written: coordinates, or another entity's point (`l1.end`), which makes the
/// two the same point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "point")]
pub enum PointExpr {
    Coords {
        x: ValueExpr,
        y: ValueExpr,
    },
    Of {
        entity: SketchEntityId,
        role: PointRole,
    },
}

impl PointExpr {
    pub fn mm(x: f64, y: f64) -> Self {
        PointExpr::Coords {
            x: ValueExpr::mm(x),
            y: ValueExpr::mm(y),
        }
    }
    pub fn of(entity: &str, role: PointRole) -> Self {
        PointExpr::Of {
            entity: SketchEntityId::new(entity),
            role,
        }
    }
}

/// A point or curve that a constraint names.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "target")]
pub enum SketchTarget {
    Entity {
        entity: SketchEntityId,
    },
    Point {
        entity: SketchEntityId,
        role: PointRole,
    },
}

impl SketchTarget {
    pub fn entity(id: &str) -> Self {
        SketchTarget::Entity {
            entity: SketchEntityId::new(id),
        }
    }
    pub fn point(id: &str, role: PointRole) -> Self {
        SketchTarget::Point {
            entity: SketchEntityId::new(id),
            role,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SketchConstraint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<ConstraintId>,
    pub kind: SketchConstraintKind,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "constraint")]
pub enum SketchConstraintKind {
    Coincident {
        a: SketchTarget,
        b: SketchTarget,
    },
    Horizontal {
        line: SketchTarget,
    },
    Vertical {
        line: SketchTarget,
    },
    Parallel {
        a: SketchTarget,
        b: SketchTarget,
    },
    Perpendicular {
        a: SketchTarget,
        b: SketchTarget,
    },
    Tangent {
        a: SketchTarget,
        b: SketchTarget,
    },
    Equal {
        a: SketchTarget,
        b: SketchTarget,
    },
    Fix {
        target: SketchTarget,
    },
    /// `point` sits at the middle of the line `line` (a line's midpoint at the origin, a
    /// circle's center halfway along an edge).
    Midpoint {
        point: SketchTarget,
        line: SketchTarget,
    },
    /// Two circles or arcs share a center.
    Concentric {
        a: SketchTarget,
        b: SketchTarget,
    },
    /// Two circles or arcs share a center and a radius.
    Coradial {
        a: SketchTarget,
        b: SketchTarget,
    },
    /// Two lines lie on one infinite line.
    Collinear {
        a: SketchTarget,
        b: SketchTarget,
    },
    /// Points `a` and `b` are mirror images in the line `axis`.
    Symmetric {
        a: SketchTarget,
        b: SketchTarget,
        axis: SketchTarget,
    },
    Distance {
        a: SketchTarget,
        b: SketchTarget,
        value: ValueExpr,
    },
    Length {
        line: SketchTarget,
        value: ValueExpr,
    },
    Radius {
        curve: SketchTarget,
        value: ValueExpr,
    },
    /// A circle's diameter: drafting dimensions full circles by diameter (⌀) and arcs by radius.
    Diameter {
        curve: SketchTarget,
        value: ValueExpr,
    },
    /// An ellipse's semi-major axis.
    MajorRadius {
        curve: SketchTarget,
        value: ValueExpr,
    },
    /// An ellipse's semi-minor axis.
    MinorRadius {
        curve: SketchTarget,
        value: ValueExpr,
    },
    Angle {
        a: SketchTarget,
        b: SketchTarget,
        value: ValueExpr,
    },
    /// A range on a dimension (distance, length, radius or angle; its own value is ignored):
    /// the solver keeps it within `[min, max]` and leaves it free inside the range.
    Limit {
        dimension: Box<SketchConstraintKind>,
        min: ValueExpr,
        max: ValueExpr,
    },
}

/// The `solved { }` cache: SI coordinates per entity point, keyed by name. Entries the solver
/// adds for hidden variables use [`SolvedKey::Hidden`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SolvedCache {
    #[serde(default, with = "crate::serde_rows")]
    pub points: BTreeMap<SolvedKey, [f64; 2]>,
    #[serde(default, with = "crate::serde_rows")]
    pub scalars: BTreeMap<SolvedKey, f64>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "key")]
pub enum SolvedKey {
    Point {
        entity: SketchEntityId,
        role: PointRole,
    },
    Radius {
        entity: SketchEntityId,
    },
    /// An ellipse's semi-axis or angle (`e1.major`, `e1.minor`, `e1.angle`; the angle in
    /// radians).
    Ellipse {
        entity: SketchEntityId,
        measure: EllipseMeasure,
    },
    /// A solver-owned variable with no authored name (e.g. a curve parameter of point-on-spline),
    /// keyed by the constraint that owns it.
    Hidden {
        constraint: ConstraintId,
        index: u32,
    },
}

/// The scalars of an ellipse that the solver owns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EllipseMeasure {
    Major,
    Minor,
    Angle,
}

impl EllipseMeasure {
    pub const ALL: [EllipseMeasure; 3] = [Self::Major, Self::Minor, Self::Angle];

    pub fn word(self) -> &'static str {
        match self {
            Self::Major => "major",
            Self::Minor => "minor",
            Self::Angle => "angle",
        }
    }

    pub fn from_word(word: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.word() == word)
    }
}

/// A solved ellipse: semi-axes in metres and the major axis's angle in radians.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SolvedEllipse {
    pub major: f64,
    pub minor: f64,
    pub angle: f64,
}

impl SolvedEllipse {
    pub fn get(&self, measure: EllipseMeasure) -> f64 {
        match measure {
            EllipseMeasure::Major => self.major,
            EllipseMeasure::Minor => self.minor,
            EllipseMeasure::Angle => self.angle,
        }
    }

    /// The point at eccentric angle `t`.
    pub fn at(&self, center: [f64; 2], t: f64) -> [f64; 2] {
        let (s, c) = self.angle.sin_cos();
        let (x, y) = (self.major * t.cos(), self.minor * t.sin());
        [center[0] + c * x - s * y, center[1] + s * x + c * y]
    }
}

/// Sketch text's sizes, evaluated: capital height, stroke width (metres) and angle (radians).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SolvedText {
    pub height: f64,
    pub stroke: f64,
    pub angle: f64,
}

/// Solved sketch geometry in sketch coordinates (SI).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SolvedSketch {
    #[serde(with = "crate::serde_rows")]
    pub points: BTreeMap<(SketchEntityId, PointRole), [f64; 2]>,
    pub radii: BTreeMap<SketchEntityId, f64>,
    #[serde(default)]
    pub ellipses: BTreeMap<SketchEntityId, SolvedEllipse>,
    #[serde(default)]
    pub texts: BTreeMap<SketchEntityId, SolvedText>,
    pub status: SketchSolveStatus,
    /// Remaining DOF per entity, where the solver reports it.
    #[serde(default)]
    pub entity_dof: BTreeMap<SketchEntityId, u32>,
    #[serde(default)]
    pub dof: Option<u32>,
    /// Constraints reported redundant (consistent but implied by others), by index into
    /// `Sketch::constraints`.
    #[serde(default)]
    pub redundant: Vec<usize>,
    /// Constraints that contradict others, by index into `Sketch::constraints`.
    #[serde(default)]
    pub conflicting: Vec<usize>,
    /// The cache to write back into `solved { }`.
    #[serde(default)]
    pub cache: SolvedCache,
}

impl SolvedSketch {
    pub fn point(&self, entity: &SketchEntityId, role: PointRole) -> Option<[f64; 2]> {
        self.points.get(&(entity.clone(), role)).copied()
    }

    /// Sketch text's outline as closed loops of lines and arcs (outers counter-clockwise, holes
    /// clockwise), each tagged with the text's id; empty when the text has nothing to draw.
    pub fn text_loops(
        &self,
        entity: &SketchEntityId,
        text: &str,
    ) -> Vec<Vec<runmat_geometry_kernel::op::ProfileSegment>> {
        use runmat_types::font::{layout, Anchor, CAP};
        let (Some(at), Some(t)) = (self.point(entity, PointRole::At), self.texts.get(entity))
        else {
            return Vec::new();
        };
        if t.height.is_nan() || t.height <= 0.0 {
            return Vec::new();
        }
        let u = t.height / CAP;
        let (s, c) = t.angle.sin_cos();
        let strokes: Vec<Vec<[f64; 2]>> = layout(text, Anchor::BaselineLeft)
            .into_iter()
            .map(|pl| {
                pl.into_iter()
                    .map(|(gx, gy)| {
                        let (x, y) = (gx * u, gy * u);
                        [at[0] + x * c - y * s, at[1] + x * s + y * c]
                    })
                    .collect()
            })
            .collect();
        runmat_geometry_kernel::stroke::stroke_outline(&strokes, t.stroke / 2.0, entity)
    }

    /// A solved ellipse as a kernel segment: the whole ellipse from the end of its major axis,
    /// or, when `arc`, the elliptical arc counter-clockwise from its start to its end. The
    /// segment's major axis is the longer semi-axis, whichever the sketch names `major`.
    pub fn ellipse_segment(
        &self,
        entity: &SketchEntityId,
        arc: bool,
    ) -> Option<runmat_geometry_kernel::op::ProfileSegment> {
        use runmat_geometry_kernel::op::{ProfileSegment, SegmentKind};
        let center = self.point(entity, PointRole::Center)?;
        let e = self.ellipses.get(entity)?;
        let (a, b) = (e.major.abs(), e.minor.abs());
        if !(a > 0.0 && b > 0.0) {
            return None;
        }
        let (length, turn, ratio) = if a >= b {
            (a, e.angle, b / a)
        } else {
            (b, e.angle + std::f64::consts::FRAC_PI_2, a / b)
        };
        let major = [length * turn.cos(), length * turn.sin()];
        let (start, end) = if arc {
            (
                self.point(entity, PointRole::Start)?,
                self.point(entity, PointRole::End)?,
            )
        } else {
            let tip = [center[0] + major[0], center[1] + major[1]];
            (tip, tip)
        };
        Some(ProfileSegment {
            curve: entity.clone(),
            kind: SegmentKind::Ellipse {
                center,
                major,
                ratio,
                clockwise: false,
            },
            start,
            end,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SketchSolveStatus {
    WellDefined,
    UnderDefined,
    OverDefined,
    /// Explicit coordinates were taken as written; no constraint was solved.
    #[default]
    Explicit,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SketchSolveFailure {
    pub message: String,
    /// Constraint indices involved (conflict set), when known.
    #[serde(default)]
    pub constraints: Vec<usize>,
}
