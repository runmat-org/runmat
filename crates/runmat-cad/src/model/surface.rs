//! Surface bodies as first-class bodies: create (extrude, revolve, sweep, loft, boundary,
//! fill, planar, ruled, offset), edit (extend, trim, untrim, knit, thicken, delete hole,
//! replace face, cut with surface) and the mid-surface.

use crate::model::datum::{AxisRef, PlaneRef};
use crate::model::feature::ProfileRef;
use crate::model::visit::{Params, Visit};
use crate::refs::Reference;
use crate::value::ValueExpr;
use serde::{Deserialize, Serialize};

/// Continuity a fill or boundary surface keeps with the faces along an edge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Continuity {
    /// Position only.
    #[default]
    G0,
    /// Tangent.
    G1,
    /// Curvature.
    G2,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoundaryEdge {
    pub edge: Reference,
    #[serde(default)]
    pub continuity: Continuity,
}

// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum SurfaceFeature {
    /// Each profile curve (open chains allowed) swept along the sketch normal.
    Extrude {
        profile: ProfileRef,
        start: ValueExpr,
        end: ValueExpr,
    },
    Revolve {
        profile: ProfileRef,
        axis: AxisRef,
        angle: ValueExpr,
    },
    Sweep {
        profile: ProfileRef,
        path: ProfileRef,
    },
    Loft {
        profiles: Vec<ProfileRef>,
        #[serde(default)]
        guides: Vec<Reference>,
    },
    /// A surface through boundary curves in two directions.
    Boundary {
        first: Vec<BoundaryEdge>,
        #[serde(default)]
        second: Vec<BoundaryEdge>,
    },
    /// A patch filling a closed loop of edges with per-edge continuity.
    Fill { boundary: Vec<BoundaryEdge> },
    /// A planar patch bounded by a closed profile.
    Planar { profile: ProfileRef },
    /// A ruled strip along edges, at a distance and angle from the faces they bound.
    Ruled {
        edges: Vec<Reference>,
        distance: ValueExpr,
        #[serde(default = "zero_angle")]
        angle: ValueExpr,
    },
    /// Offset faces (of a sheet or a solid) along their normals; the source stays.
    Offset {
        faces: Vec<Reference>,
        distance: ValueExpr,
    },
    Extend {
        edges: Vec<Reference>,
        distance: ValueExpr,
    },
    /// Trim sheets by a plane or another surface, keeping one side.
    Trim {
        bodies: Vec<Reference>,
        tool: PlaneRef,
        keep: crate::refs::name::PlaneSide,
    },
    /// Remove the trim of faces (restore the underlying surface's natural boundary).
    Untrim { faces: Vec<Reference> },
    /// Knit sheets sharing edges; `solid` closes a watertight set.
    Knit {
        bodies: Vec<Reference>,
        #[serde(default)]
        solid: bool,
    },
    Thicken {
        bodies: Vec<Reference>,
        thickness: ValueExpr,
        #[serde(default)]
        both_sides: bool,
    },
    /// Close inner loops of a sheet's faces.
    DeleteHole { edges: Vec<Reference> },
    /// Cut a solid with a surface, keeping one side.
    CutWithSurface {
        body: Reference,
        surface: Reference,
        keep: crate::refs::name::PlaneSide,
    },
    /// The mid-surface of a thin solid from pairs of opposite faces.
    MidSurface {
        body: Reference,
        pairs: Vec<(Reference, Reference)>,
    },
}

fn zero_angle() -> ValueExpr {
    ValueExpr::deg(0.0)
}

impl Params for SurfaceFeature {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        use SurfaceFeature as S;
        let edges = |v: &mut Visit<'a>, name: &str, es: &'a [BoundaryEdge]| {
            for (i, e) in es.iter().enumerate() {
                v.reference(format!("{}[{}]", name, i), &e.edge);
            }
        };
        match self {
            S::Extrude {
                profile,
                start,
                end,
            } => {
                v.profile(profile);
                v.value(start);
                v.value(end);
            }
            S::Revolve {
                profile,
                axis,
                angle,
            } => {
                v.profile(profile);
                v.axis("axis", axis);
                v.value(angle);
            }
            S::Sweep { profile, path } => {
                v.profile(profile);
                v.profile(path);
            }
            S::Loft { profiles, guides } => {
                profiles.iter().for_each(|p| v.profile(p));
                v.references("guides", guides);
            }
            S::Boundary { first, second } => {
                edges(v, "first", first);
                edges(v, "second", second);
            }
            S::Fill { boundary } => edges(v, "boundary", boundary),
            S::Planar { profile } => v.profile(profile),
            S::Ruled {
                edges,
                distance,
                angle,
            } => {
                v.references("edges", edges);
                v.value(distance);
                v.value(angle);
            }
            S::Offset { faces, distance } => {
                v.references("faces", faces);
                v.value(distance);
            }
            S::Extend { edges, distance } => {
                v.references("edges", edges);
                v.value(distance);
            }
            S::Trim { bodies, tool, .. } => {
                v.references("bodies", bodies);
                v.plane("tool", tool);
            }
            S::Untrim { faces } => v.references("faces", faces),
            S::Knit { bodies, .. } => v.references("bodies", bodies),
            S::Thicken {
                bodies, thickness, ..
            } => {
                v.references("bodies", bodies);
                v.value(thickness);
            }
            S::DeleteHole { edges } => v.references("edges", edges),
            S::CutWithSurface { body, surface, .. } => {
                v.reference("body", body);
                v.reference("surface", surface);
            }
            S::MidSurface { body, pairs } => {
                v.reference("body", body);
                for (i, (a, b)) in pairs.iter().enumerate() {
                    v.reference(format!("pairs[{}].0", i), a);
                    v.reference(format!("pairs[{}].1", i), b);
                }
            }
        }
    }
    fn kind_name(&self) -> &'static str {
        use SurfaceFeature as S;
        match self {
            S::Extrude { .. } => "surface.extrude",
            S::Revolve { .. } => "surface.revolve",
            S::Sweep { .. } => "surface.sweep",
            S::Loft { .. } => "surface.loft",
            S::Boundary { .. } => "surface.boundary",
            S::Fill { .. } => "surface.fill",
            S::Planar { .. } => "surface.planar",
            S::Ruled { .. } => "surface.ruled",
            S::Offset { .. } => "surface.offset",
            S::Extend { .. } => "surface.extend",
            S::Trim { .. } => "surface.trim",
            S::Untrim { .. } => "surface.untrim",
            S::Knit { .. } => "surface.knit",
            S::Thicken { .. } => "surface.thicken",
            S::DeleteHole { .. } => "surface.delete_hole",
            S::CutWithSurface { .. } => "surface.cut_with_surface",
            S::MidSurface { .. } => "surface.mid_surface",
        }
    }
    fn new_body(&self) -> bool {
        use SurfaceFeature as S;
        matches!(
            self,
            S::Extrude { .. }
                | S::Revolve { .. }
                | S::Sweep { .. }
                | S::Loft { .. }
                | S::Boundary { .. }
                | S::Fill { .. }
                | S::Planar { .. }
                | S::Ruled { .. }
                | S::Offset { .. }
                | S::MidSurface { .. }
        )
    }
}
