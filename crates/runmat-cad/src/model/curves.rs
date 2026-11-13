//! Reference curves and mate connectors.
//!
//! Curves (helix and spiral, projected, composite, split line, intersection, through points,
//! equation-driven) evaluate to sampled 3D polylines in part coordinates, kept beside the
//! datums; sweeps, patterns and threads can follow them. A split line also splits faces and so
//! is a body feature. Mate connectors are frames assemblies mate to, placed on geometry.

use crate::ids::InputName;
use crate::model::datum::{AxisRef, FrameDef, PlaneRef, PointRef};
use crate::model::feature::ProfileRef;
use crate::model::visit::{Params, Visit};
use crate::refs::Reference;
use crate::value::ValueExpr;
use serde::{Deserialize, Serialize};

// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "curve")]
pub enum CurveDef {
    /// A helix about an axis, starting at `start` (its radius is the start's distance from the
    /// axis), `turns` turns of `pitch`, optionally tapered (radius change per turn).
    Helix {
        axis: AxisRef,
        start: PointRef,
        pitch: ValueExpr,
        turns: ValueExpr,
        #[serde(default)]
        left_hand: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        taper: Option<ValueExpr>,
    },
    /// A flat spiral in a plane: radius grows by `pitch` per turn from `start_radius`.
    Spiral {
        plane: PlaneRef,
        start_radius: ValueExpr,
        pitch: ValueExpr,
        turns: ValueExpr,
    },
    /// Sketch curves projected along the sketch normal onto faces.
    Projected {
        profile: ProfileRef,
        onto: Vec<Reference>,
    },
    /// Edges joined end to end into one curve.
    Composite { edges: Vec<Reference> },
    /// Sketch curves projected onto faces, splitting them.
    SplitLine {
        profile: ProfileRef,
        faces: Vec<Reference>,
    },
    /// Where two faces (or a face and a plane) meet.
    Intersection { a: Reference, b: PlaneRef },
    /// A curve through points, in order.
    ThroughPoints {
        points: Vec<PointRef>,
        #[serde(default)]
        closed: bool,
    },
    /// `x(t), y(t), z(t)` over `[t0, t1]`; the expressions read the parameter as an input.
    Equation {
        parameter: InputName,
        x: ValueExpr,
        y: ValueExpr,
        z: ValueExpr,
        t0: ValueExpr,
        t1: ValueExpr,
        samples: u32,
    },
}

impl Params for CurveDef {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        match self {
            CurveDef::Helix {
                axis,
                start,
                pitch,
                turns,
                taper,
                ..
            } => {
                v.axis("axis", axis);
                v.point("start", start);
                v.value(pitch);
                v.value(turns);
                v.opt_value(taper);
            }
            CurveDef::Spiral {
                plane,
                start_radius,
                pitch,
                turns,
            } => {
                v.plane("plane", plane);
                v.value(start_radius);
                v.value(pitch);
                v.value(turns);
            }
            CurveDef::Projected { profile, onto } => {
                v.profile(profile);
                v.references("onto", onto);
            }
            CurveDef::Composite { edges } => v.references("edges", edges),
            CurveDef::SplitLine { profile, faces } => {
                v.profile(profile);
                v.references("faces", faces);
            }
            CurveDef::Intersection { a, b } => {
                v.reference("a", a);
                v.plane("b", b);
            }
            CurveDef::ThroughPoints { points, .. } => {
                for (i, p) in points.iter().enumerate() {
                    v.point(&format!("points[{}]", i), p);
                }
            }
            CurveDef::Equation { t0, t1, .. } => {
                // x, y, z read the parameter; they are evaluated per sample, not here.
                v.value(t0);
                v.value(t1);
            }
        }
    }
    fn kind_name(&self) -> &'static str {
        match self {
            CurveDef::Helix { .. } => "curve.helix",
            CurveDef::Spiral { .. } => "curve.spiral",
            CurveDef::Projected { .. } => "curve.projected",
            CurveDef::Composite { .. } => "curve.composite",
            CurveDef::SplitLine { .. } => "curve.split_line",
            CurveDef::Intersection { .. } => "curve.intersection",
            CurveDef::ThroughPoints { .. } => "curve.through_points",
            CurveDef::Equation { .. } => "curve.equation",
        }
    }
    fn body_feature(&self) -> bool {
        matches!(self, CurveDef::SplitLine { .. })
    }
}

/// An evaluated reference curve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CurveValue {
    pub points: Vec<[f64; 3]>,
    pub closed: bool,
}

impl CurveValue {
    pub fn length(&self) -> f64 {
        let mut l: f64 = self
            .points
            .windows(2)
            .map(|w| {
                let d = [w[1][0] - w[0][0], w[1][1] - w[0][1], w[1][2] - w[0][2]];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            })
            .sum();
        if self.closed && self.points.len() > 2 {
            let (a, b) = (self.points[self.points.len() - 1], self.points[0]);
            let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            l += (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        }
        l
    }
}

/// Where a mate connector sits.
// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "on")]
pub enum ConnectorPlacement {
    /// Any frame definition (a point and axes, or a planar face's frame).
    Frame { frame: FrameDef },
    /// The centre of a circular edge or cylindrical face, Z along its axis.
    Center { of: Reference },
    /// The midpoint of a linear edge, Z along the edge.
    EdgeMidpoint { edge: Reference },
    /// A vertex, axes from the part's.
    Vertex { vertex: Reference },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MateConnectorDef {
    pub placement: ConnectorPlacement,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<[ValueExpr; 3]>,
    /// Rotation about the connector's own Z.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation: Option<ValueExpr>,
    /// Flip Z (and Y).
    #[serde(default)]
    pub flip: bool,
}

impl Params for MateConnectorDef {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        match &self.placement {
            ConnectorPlacement::Frame { frame } => match frame {
                FrameDef::At { origin, z, x } => {
                    v.point("origin", origin);
                    v.axis("z", z);
                    if let Some(x) = x {
                        v.axis("x", x);
                    }
                }
                FrameDef::OnFace { face } => v.reference("face", face),
            },
            ConnectorPlacement::Center { of } => v.reference("of", of),
            ConnectorPlacement::EdgeMidpoint { edge } => v.reference("edge", edge),
            ConnectorPlacement::Vertex { vertex } => v.reference("vertex", vertex),
        }
        if let Some(o) = &self.offset {
            o.iter().for_each(|x| v.value(x));
        }
        v.opt_value(&self.rotation);
    }
    fn kind_name(&self) -> &'static str {
        "mate_connector"
    }
    fn body_feature(&self) -> bool {
        false
    }
}
