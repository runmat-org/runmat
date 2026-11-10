//! Datum planes, axes, points and coordinate systems.
//!
//! Origin datums are built in. Every other datum is a feature (`FeatureKind::Plane`, `Axis`,
//! `Point`, `Frame`), so suppression, ordering and references treat it like any other feature.

use crate::ids::FeatureId;
use crate::refs::Reference;
use crate::value::ValueExpr;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OriginPlane {
    Xy,
    Yz,
    Zx,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OriginAxis {
    X,
    Y,
    Z,
}

/// A reference to a datum: a built-in origin datum or a datum feature.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "datum")]
pub enum DatumRef {
    OriginPlane {
        plane: OriginPlane,
    },
    OriginAxis {
        axis: OriginAxis,
    },
    OriginPoint,
    OriginFrame,
    /// A plane, axis, point or frame feature.
    Feature {
        feature: FeatureId,
    },
}

impl fmt::Display for DatumRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DatumRef::OriginPlane { plane } => write!(
                f,
                "origin.{}",
                match plane {
                    OriginPlane::Xy => "xy",
                    OriginPlane::Yz => "yz",
                    OriginPlane::Zx => "zx",
                }
            ),
            DatumRef::OriginAxis { axis } => write!(
                f,
                "origin.{}",
                match axis {
                    OriginAxis::X => "x",
                    OriginAxis::Y => "y",
                    OriginAxis::Z => "z",
                }
            ),
            DatumRef::OriginPoint => f.write_str("origin.point"),
            DatumRef::OriginFrame => f.write_str("origin"),
            DatumRef::Feature { feature } => write!(f, "datum.{}", feature),
        }
    }
}

/// How a plane is written, for messages that ask for one: the text forms `on` and plane fields take.
pub const PLANE_REFERENCE_FORMS: &str = "an origin plane (`origin.xy`, `origin.yz` or `origin.zx`), \
a datum plane (`plane.<name>`) or a planar face (`feature.<name>.face.<face>`, such as `feature.body.face.end`)";

/// Where a sketch or a plane-taking feature sits: a datum plane or a planar face.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "on")]
pub enum PlaneRef {
    Datum { datum: DatumRef },
    Face { face: Reference },
}

impl PlaneRef {
    pub fn origin_xy() -> Self {
        PlaneRef::Datum {
            datum: DatumRef::OriginPlane {
                plane: OriginPlane::Xy,
            },
        }
    }
    pub fn datum(feature: &str) -> Self {
        PlaneRef::Datum {
            datum: DatumRef::Feature {
                feature: FeatureId::new(feature),
            },
        }
    }
}

/// An axis-taking parameter: a datum axis, or a linear edge.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "along")]
pub enum AxisRef {
    Datum { datum: DatumRef },
    Edge { edge: Reference },
}

impl AxisRef {
    pub fn origin(axis: OriginAxis) -> Self {
        AxisRef::Datum {
            datum: DatumRef::OriginAxis { axis },
        }
    }
}

// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "plane")]
pub enum PlaneDef {
    /// Parallel to `base` at a signed distance along its normal.
    Offset { base: PlaneRef, distance: ValueExpr },
    /// Through `axis`, rotated from `base` by `angle`.
    Angle {
        base: PlaneRef,
        axis: AxisRef,
        angle: ValueExpr,
    },
    /// Through three points (normal by the right hand from `a→b` to `a→c`).
    ThreePoint {
        a: PointRef,
        b: PointRef,
        c: PointRef,
    },
    /// Midway between two parallel planes.
    Mid { a: PlaneRef, b: PlaneRef },
    /// Normal to a reference curve or a linear edge at a parameter in `[0, 1]` along it.
    NormalToCurve {
        curve: CurveRef,
        parameter: ValueExpr,
    },
    /// Tangent to a cylindrical face at the point of it nearest `at`.
    TangentToFace { face: Reference, at: PointRef },
}

/// A curve a datum or pattern follows: a reference-curve feature, an edge, or a sketch's open
/// chain (its path).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "curve")]
pub enum CurveRef {
    Feature { feature: FeatureId },
    Edge { edge: Reference },
    Sketch { sketch: FeatureId },
}

// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "axis")]
pub enum AxisDef {
    /// Through two points.
    Through { a: PointRef, b: PointRef },
    /// The axis of a cylindrical or conical face.
    OfFace { face: Reference },
    /// Along a linear edge.
    OfEdge { edge: Reference },
    /// Where two planes meet.
    TwoPlanes { a: PlaneRef, b: PlaneRef },
    /// Through a point, normal to a plane.
    Normal { plane: PlaneRef, through: PointRef },
}

/// A point-taking parameter.
// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "point")]
pub enum PointRef {
    Datum {
        datum: DatumRef,
    },
    Coords {
        x: ValueExpr,
        y: ValueExpr,
        z: ValueExpr,
    },
    Vertex {
        vertex: Reference,
    },
    /// A sketch point (`sketch.profile.a1.center`).
    Sketch {
        sketch: FeatureId,
        entity: crate::ids::SketchEntityId,
        role: crate::model::sketch::PointRole,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "point_def")]
pub enum PointDef {
    At {
        at: PointRef,
    },
    /// Offset from a point by a vector.
    Offset {
        base: PointRef,
        dx: ValueExpr,
        dy: ValueExpr,
        dz: ValueExpr,
    },
    /// The centre of a circular edge or cylindrical face (on its axis, at its centroid).
    CenterOf {
        of: Reference,
    },
    /// At a parameter in `[0, 1]` along a curve or edge.
    OnCurve {
        curve: CurveRef,
        parameter: ValueExpr,
    },
    /// Where an axis meets a plane.
    Intersection {
        axis: AxisRef,
        plane: PlaneRef,
    },
}

// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "frame")]
pub enum FrameDef {
    /// Origin at a point, Z along an axis, X towards a second axis' direction.
    At {
        origin: PointRef,
        z: AxisRef,
        x: Option<AxisRef>,
    },
    /// The frame of a planar face (mate-connector style).
    OnFace { face: Reference },
}

impl CurveRef {
    pub(crate) fn visit<'a>(&'a self, v: &mut crate::model::visit::Visit<'a>, param: &str) {
        match self {
            CurveRef::Feature { feature } => v.feature(feature),
            CurveRef::Edge { edge } => v.reference(param, edge),
            CurveRef::Sketch { sketch } => v.feature(sketch),
        }
    }
}

pub use runmat_geometry_kernel::frame::Frame;

impl OriginPlane {
    /// The frame of an origin plane.
    pub fn frame(self) -> Frame {
        match self {
            OriginPlane::Xy => Frame::WORLD,
            OriginPlane::Yz => Frame {
                origin: [0.0; 3],
                x: [0.0, 1.0, 0.0],
                y: [0.0, 0.0, 1.0],
                z: [1.0, 0.0, 0.0],
            },
            OriginPlane::Zx => Frame {
                origin: [0.0; 3],
                x: [0.0, 0.0, 1.0],
                y: [1.0, 0.0, 0.0],
                z: [0.0, 1.0, 0.0],
            },
        }
    }
}

/// An evaluated datum.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "datum")]
pub enum DatumValue {
    Plane {
        frame: Frame,
    },
    Axis {
        point: [f64; 3],
        direction: [f64; 3],
    },
    Point {
        at: [f64; 3],
    },
    Frame {
        frame: Frame,
    },
}

pub(crate) fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = dot(a, a).sqrt();
    if n == 0.0 {
        a
    } else {
        scale(a, 1.0 / n)
    }
}
