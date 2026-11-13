//! A deterministic fake kernel for testing the reference system end to end without OCCT.
//!
//! Solids are orthogonal polyhedra on a rectilinear grid. Every operation re-grids its operands
//! and tool on the union of their coordinates, decides cell occupancy, and derives history
//! from provenance: each boundary facet is labelled with the operand face or tool face it lies
//! on, coplanar connected facets form faces (so flush faces merge and interrupted faces split),
//! and connected cells form bodies. Edges and vertices come from facet adjacency. Fillets and
//! chamfers are synthetic blend faces laid over grid edges and carried through later
//! operations.
//!
//! Rectilinear tools (axis-aligned polygon extrudes on axis-aligned frames) are exact. Every
//! other shape is a sampled tool ([`sampled`]): its occupancy is decided at cell centres and its
//! facets are labelled by the tool surface the boundary crosses, so a curved or off-axis
//! surface becomes one faceted face (a staircase of facets with one label) and the edges where
//! it meets other faces are chained into one edge per run. History stays exact in the K1 sense:
//! roles, tags and ordinals come from the operation's definition; splits, merges and deletions
//! come from occupancy.
//!
//! Supported: extrudes of line, arc and circle profiles on any frame (new, add, cut, intersect,
//! blind, symmetric, through all); revolves at any angle; translational sweeps along polyline
//! paths; ruled lofts through parallel sections; round holes (simple, counterbore, countersink,
//! blind or through); patterns and mirrors of tools and of bodies under any rigid motion or
//! reflection; booleans; splits by axis-aligned planes; transforms; fillets and chamfers on
//! straight edges between planar grid faces; shells (inner walls tagged by the face they
//! offset, rims tagged by the face they open); drafts of planar faces about a neutral plane;
//! and imports of named boxes. Everything else returns `KernelError::NotSupported`, which tests
//! use for continue-past-failure.

// Grid code indexes several per-axis arrays in step; index loops read clearer than zips here.
#![allow(clippy::needless_range_loop)]

mod tools;
mod regrid;
mod run_grid;
mod topology;
mod kernel;
mod ops;
use regrid::*;
use run_grid::*;
use tools::*;
use topology::*;

mod analysis;
mod cdt;
mod display;
mod exact;
mod queries;
mod extra;
mod face_profile;
mod features;
mod mesh;
mod sampled;
mod sheet;

pub use sheet::FakeSheet;

/// Triangles covering a polygon with holes (`loops[0]` counter-clockwise, the rest clockwise,
/// as indices into `pts`); `None` when it could not be bridged.
pub(crate) fn triangulate_polygon(
    pts: &[[f64; 2]],
    loops: &[Vec<usize>],
) -> Option<Vec<[usize; 3]>> {
    cdt::triangulate(pts, loops, &[])
}

use crate::model::datum::Frame;
use crate::refs::history::*;
use crate::refs::name::{InstanceKey, PlaneSide, Role};
use crate::regen::kernel::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

const EPS: f64 = 1e-9;

/// An edge with its grid metadata and its two end points in grid indices.
/// A facet edge segment: the axis it runs along and its start point in grid indices.
type SegmentKey = (usize, [usize; 3]);
/// An edge run's identity: its face pair, axis and line (start index zeroed).
type PieceKey = ((u32, u32), usize, [usize; 3]);
type EdgeRow = (EdgeTopo, EdgeMeta, Vec<[usize; 3]>);

#[derive(Clone, Debug, PartialEq)]
pub struct FakeImportBox {
    pub name: String,
    pub min: [f64; 3],
    pub max: [f64; 3],
    /// Names for the faces in order `-x, +x, -y, +y, -z, +z`.
    pub face_names: [String; 6],
}

#[derive(Clone, Debug, Default)]
pub struct FakeKernel {
    /// Import fixtures by source path.
    pub imports: BTreeMap<String, Vec<FakeImportBox>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PlaneKey {
    axis: usize,
    dir: i8,
    offset: f64,
}

impl PlaneKey {
    fn same(&self, o: &PlaneKey) -> bool {
        self.axis == o.axis && self.dir == o.dir && (self.offset - o.offset).abs() < EPS
    }
}

#[derive(Clone, Copy, Debug)]
struct Rect {
    lo: [f64; 3],
    hi: [f64; 3],
}

impl Rect {
    fn contains(&self, p: [f64; 3]) -> bool {
        (0..3).all(|i| p[i] >= self.lo[i] - EPS && p[i] <= self.hi[i] + EPS)
    }
}

#[derive(Clone, Debug)]
struct FFace {
    /// The plane of a planar grid face; `None` for faceted and blend faces.
    plane: Option<PlaneKey>,
    rects: Vec<Rect>,
    /// For a faceted face (a curved or off-axis surface on the grid): each rect's plane, in
    /// step with `rects`. Empty for planar and blend faces.
    facet_planes: Vec<PlaneKey>,
    /// Informative surface of a faceted face.
    surface: Option<SurfaceKind>,
    /// The true surface of a faceted face, when its tool knows one (for display).
    exact: Option<Arc<exact::Exact>>,
}

/// Where a facet's exact surface comes from.
#[derive(Clone, Debug)]
enum FacetExact {
    None,
    Ready(Arc<exact::Exact>),
    /// Surface `.1` of tool `.0` (in `tools` then `carve` order).
    Tool(usize, usize),
    /// Surface of the tool that labelled the facet, before the tool index is known.
    Surface(usize),
}

/// A plane through `o` with normal `n`, as an exact surface.
fn exact_plane(o: [f64; 3], n: [f64; 3]) -> exact::Exact {
    let a = (0..3)
        .min_by(|i, j| n[*i].abs().total_cmp(&n[*j].abs()))
        .unwrap_or(0);
    let mut t = [0.0; 3];
    t[a] = 1.0;
    let e1 = [
        n[1] * t[2] - n[2] * t[1],
        n[2] * t[0] - n[0] * t[2],
        n[0] * t[1] - n[1] * t[0],
    ];
    let e2 = [
        n[1] * e1[2] - n[2] * e1[1],
        n[2] * e1[0] - n[0] * e1[2],
        n[0] * e1[1] - n[1] * e1[0],
    ];
    exact::Exact::plane(o, e1, e2)
}

/// Whether a facet lies on a planar grid face or on a faceted surface.
#[derive(Clone, Copy, Debug, PartialEq)]
enum FacetKind {
    Planar,
    Faceted(SurfaceKind),
}

impl FFace {
    fn blend() -> Self {
        FFace {
            plane: None,
            rects: Vec::new(),
            facet_planes: Vec::new(),
            surface: None,
            exact: None,
        }
    }

    fn is_grid(&self) -> bool {
        !self.rects.is_empty()
    }

    fn rect_planes(&self) -> Vec<(PlaneKey, Rect)> {
        match self.plane {
            Some(p) => self.rects.iter().map(|r| (p, *r)).collect(),
            None => self
                .facet_planes
                .iter()
                .copied()
                .zip(self.rects.iter().copied())
                .collect(),
        }
    }

    fn lies_on(&self, plane: &PlaneKey, p: [f64; 3]) -> bool {
        match &self.plane {
            Some(fp) => fp.same(plane) && self.rects.iter().any(|r| r.contains(p)),
            None => self
                .facet_planes
                .iter()
                .zip(&self.rects)
                .any(|(fp, r)| fp.same(plane) && r.contains(p)),
        }
    }

    fn hint(&self) -> SurfaceKind {
        match self.plane {
            Some(p) => {
                let mut n = [0.0; 3];
                n[p.axis] = p.dir as f64;
                SurfaceKind::Plane { normal: n }
            }
            None => self.surface.unwrap_or(SurfaceKind::Other),
        }
    }

    fn kind(&self) -> FacetKind {
        if self.plane.is_some() {
            FacetKind::Planar
        } else {
            FacetKind::Faceted(self.hint())
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct EdgeMeta {
    along: usize,
    /// Coordinates of the line on the other two axes, as world values in axis order.
    at: [f64; 3],
    lo: f64,
    hi: f64,
    /// Plane of each face, when both are planar grid faces.
    planes: Option<[PlaneKey; 2]>,
}

#[derive(Clone, Debug)]
struct Decoration {
    role: Role,
    size: f64,
    along: usize,
    at: [f64; 3],
    lo: f64,
    hi: f64,
    planes: [PlaneKey; 2],
    /// The blend faces this decoration currently owns in its body: (lo, hi, face index).
    pieces: Vec<(f64, f64, u32)>,
    /// For a decoration made by this operation: the label its faces get.
    fresh: Option<HistorySource>,
    /// A variable-radius, conic or curvature-continuous blend: not a cylinder.
    curved: bool,
    /// Curvature-continuous (G2) to its faces.
    g2: bool,
}

/// The kernel's body payload.
#[derive(Clone, Debug)]
pub struct FakeSolid {
    grid: [Vec<f64>; 3],
    occ: Vec<bool>,
    faces: Vec<FFace>,
    edge_meta: Vec<EdgeMeta>,
    decorations: Vec<Decoration>,
    topology: Topology,
}

impl FakeSolid {
    fn dims(&self) -> [usize; 3] {
        [
            self.grid[0].len().saturating_sub(1),
            self.grid[1].len().saturating_sub(1),
            self.grid[2].len().saturating_sub(1),
        ]
    }

    fn contains(&self, p: [f64; 3]) -> bool {
        let d = self.dims();
        let mut c = [0usize; 3];
        for a in 0..3 {
            let g = &self.grid[a];
            if g.len() < 2 || p[a] <= g[0] || p[a] >= g[g.len() - 1] {
                return false;
            }
            c[a] = g.partition_point(|x| *x <= p[a]) - 1;
            if c[a] >= d[a] {
                return false;
            }
        }
        self.occ[c[0] + d[0] * (c[1] + d[1] * c[2])]
    }

    fn bounds(&self) -> Option<([f64; 3], [f64; 3])> {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        let mut any = false;
        for f in &self.faces {
            for r in &f.rects {
                any = true;
                for a in 0..3 {
                    lo[a] = lo[a].min(r.lo[a]);
                    hi[a] = hi[a].max(r.hi[a]);
                }
            }
        }
        any.then_some((lo, hi))
    }
}
