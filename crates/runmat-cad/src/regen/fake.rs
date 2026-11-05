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

mod analysis;
mod cdt;
mod display;
mod exact;
mod extra;
mod features;
mod mesh;
mod sampled;
mod sheet;

pub use sheet::FakeSheet;

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

// ------------------------------------------------------------------------------- tools

#[derive(Clone, Debug)]
struct Prism {
    k: usize,
    u: usize,
    v: usize,
    loops: Vec<Vec<[f64; 2]>>,
    edge_labels: Vec<Vec<HistorySource>>,
    lo: f64,
    hi: f64,
    lo_label: HistorySource,
    hi_label: HistorySource,
}

impl Prism {
    fn inside_uv(&self, q: [f64; 2]) -> bool {
        let mut inside = false;
        for l in &self.loops {
            let n = l.len();
            for i in 0..n {
                let (a, b) = (l[i], l[(i + 1) % n]);
                if (a[1] > q[1]) != (b[1] > q[1]) {
                    let x = a[0] + (q[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
                    if q[0] < x {
                        inside = !inside;
                    }
                }
            }
        }
        inside
    }

    fn inside(&self, p: [f64; 3]) -> bool {
        p[self.k] > self.lo && p[self.k] < self.hi && self.inside_uv([p[self.u], p[self.v]])
    }

    fn label(&self, axis: usize, offset: f64, center: [f64; 3]) -> Option<HistorySource> {
        let uv = [center[self.u], center[self.v]];
        if axis == self.k {
            if (offset - self.lo).abs() < EPS && self.inside_uv(uv) {
                return Some(self.lo_label.clone());
            }
            if (offset - self.hi).abs() < EPS && self.inside_uv(uv) {
                return Some(self.hi_label.clone());
            }
            return None;
        }
        if center[self.k] < self.lo - EPS || center[self.k] > self.hi + EPS {
            return None;
        }
        let (on, other) = if axis == self.u { (0, 1) } else { (1, 0) };
        for (li, l) in self.loops.iter().enumerate() {
            let n = l.len();
            for i in 0..n {
                let (a, b) = (l[i], l[(i + 1) % n]);
                if (a[on] - offset).abs() < EPS && (b[on] - offset).abs() < EPS {
                    let (m0, m1) = (a[other].min(b[other]), a[other].max(b[other]));
                    if uv[other] >= m0 - EPS && uv[other] <= m1 + EPS {
                        return Some(self.edge_labels[li][i].clone());
                    }
                }
            }
        }
        None
    }

    fn breaks(&self, out: &mut [Vec<f64>; 3]) {
        out[self.k].push(self.lo);
        out[self.k].push(self.hi);
        for l in &self.loops {
            for p in l {
                out[self.u].push(p[0]);
                out[self.v].push(p[1]);
            }
        }
    }

    /// Apply a signed axis permutation (right-angle rotations, axis reflections, translation)
    /// exactly; `None` for any other motion.
    fn permuted(&self, t: &Transform) -> Option<Prism> {
        let perm = signed_permutation(t)?;
        let map = |a: usize, x: f64| -> (usize, f64) {
            let (b, sg) = perm[a];
            (b, sg * x + t.translation[b])
        };
        let mut p = self.clone();
        p.k = perm[self.k].0;
        p.u = perm[self.u].0;
        p.v = perm[self.v].0;
        for l in &mut p.loops {
            for q in l.iter_mut() {
                *q = [map(self.u, q[0]).1, map(self.v, q[1]).1];
            }
        }
        let (a, b) = (map(self.k, self.lo).1, map(self.k, self.hi).1);
        if a <= b {
            p.lo = a;
            p.hi = b;
        } else {
            p.lo = b;
            p.hi = a;
            std::mem::swap(&mut p.lo_label, &mut p.hi_label);
        }
        Some(p)
    }

    /// The same prism as a sampled tool, for motions that leave the grid axes.
    fn to_sampled(&self) -> sampled::Sampled {
        let mut rot = [[0.0; 3]; 3];
        rot[self.u][0] = 1.0;
        rot[self.v][1] = 1.0;
        rot[self.k][2] = 1.0;
        let to_world = Transform {
            rotation: rot,
            translation: [0.0; 3],
        };
        let loops: Vec<Vec<(sampled::Curve2, HistorySource)>> = self
            .loops
            .iter()
            .zip(&self.edge_labels)
            .map(|(l, labs)| {
                (0..l.len())
                    .map(|i| {
                        (
                            sampled::Curve2::Line {
                                a: l[i],
                                b: l[(i + 1) % l.len()],
                            },
                            labs[i].clone(),
                        )
                    })
                    .collect()
            })
            .collect();
        sampled::prism(
            to_world,
            &loops,
            self.lo,
            self.hi,
            [self.lo_label.clone(), self.hi_label.clone()],
        )
    }

    fn relabel(&mut self, seed: u32, instance: &InstanceKey) {
        let fix = |s: &mut HistorySource| {
            if let HistorySource::Generated {
                seed: sd,
                instance: inst,
                ..
            } = s
            {
                *sd = Some(seed);
                *inst = Some(instance.clone());
            }
        };
        fix(&mut self.lo_label);
        fix(&mut self.hi_label);
        for l in &mut self.edge_labels {
            for s in l {
                fix(s);
            }
        }
    }
}

/// For each source axis, the axis it maps to and the sign, when `t`'s rotation is a signed
/// permutation.
fn signed_permutation(t: &Transform) -> Option<[(usize, f64); 3]> {
    let r = &t.rotation;
    let mut out = [(0usize, 0.0f64); 3];
    for a in 0..3 {
        let mut found = None;
        for b in 0..3 {
            let x = r[b][a];
            if (x.abs() - 1.0).abs() < EPS {
                if found.is_some() {
                    return None;
                }
                found = Some((b, x.signum()));
            } else if x.abs() > EPS {
                return None;
            }
        }
        out[a] = found?;
    }
    Some(out)
}

/// A tool shape on the grid: exact rectilinear prisms, or sampled shapes.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
enum Tool {
    Exact(Prism),
    Sampled(Box<sampled::Sampled>),
    /// A closed set of labelled polygons (thickened or knitted sheets).
    Polys(Box<sheet::PolyTool>),
    /// A tool clipped to half-spaces `(p − origin)·normal ≤ 0`, the cuts labelled.
    Clipped(Box<Tool>, Vec<([f64; 3], [f64; 3], HistorySource)>),
}

fn half(o: [f64; 3], n: [f64; 3], p: [f64; 3]) -> f64 {
    (p[0] - o[0]) * n[0] + (p[1] - o[1]) * n[1] + (p[2] - o[2]) * n[2]
}

impl Tool {
    fn inside(&self, p: [f64; 3]) -> bool {
        match self {
            Tool::Exact(t) => t.inside(p),
            Tool::Sampled(t) => t.inside(p),
            Tool::Polys(t) => t.inside(p),
            Tool::Clipped(t, planes) => {
                t.inside(p) && planes.iter().all(|(o, n, _)| half(*o, *n, p) < 0.0)
            }
        }
    }

    fn breaks(&self, out: &mut [Vec<f64>; 3]) {
        match self {
            Tool::Exact(t) => t.breaks(out),
            Tool::Sampled(t) => t.breaks(out),
            Tool::Polys(t) => t.breaks(out),
            Tool::Clipped(t, planes) => {
                t.breaks(out);
                for (o, n, _) in planes {
                    for a in 0..3 {
                        out[a].push(o[a]);
                    }
                    if axis_of(*n).is_none() {
                        let mut tmp: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
                        t.breaks(&mut tmp);
                        for a in 0..3 {
                            let lo = tmp[a].iter().cloned().fold(f64::INFINITY, f64::min);
                            let hi = tmp[a].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            if lo.is_finite() && hi.is_finite() {
                                for k in 0..=sampled::RES {
                                    out[a].push(lo + (hi - lo) * k as f64 / sampled::RES as f64);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The label of a facet on `plane` at `cen` between cell centres `a` (the facet's cell)
    /// and `b` (across it), when this tool bounds it.
    fn label(
        &self,
        plane: &PlaneKey,
        cen: [f64; 3],
        a: [f64; 3],
        b: [f64; 3],
    ) -> Option<(HistorySource, FacetKind, FacetExact)> {
        match self {
            Tool::Exact(t) => t
                .label(plane.axis, plane.offset, cen)
                .map(|l| (l, FacetKind::Planar, FacetExact::None)),
            Tool::Sampled(t) => t.crossing(a, b).map(|i| {
                let s = &t.surfaces[i];
                let kind = match s.plane {
                    Some((ax, off)) if ax == plane.axis && (off - plane.offset).abs() < 1e-7 => {
                        FacetKind::Planar
                    }
                    _ => FacetKind::Faceted(s.hint),
                };
                (s.label.clone(), kind, FacetExact::Surface(i))
            }),
            Tool::Polys(t) => t.label(plane, cen, a, b),
            Tool::Clipped(t, planes) => {
                for (o, n, l) in planes {
                    let (da, db) = (half(*o, *n, a), half(*o, *n, b));
                    if (da <= 0.0) != (db <= 0.0) {
                        let s = da / (da - db);
                        let x = [
                            a[0] + s * (b[0] - a[0]),
                            a[1] + s * (b[1] - a[1]),
                            a[2] + s * (b[2] - a[2]),
                        ];
                        let inside_other = t.inside(x) || t.inside(if da <= 0.0 { a } else { b });
                        if inside_other {
                            let kind = match axis_of(*n) {
                                Some((ax, _)) if ax == plane.axis => FacetKind::Planar,
                                _ => FacetKind::Faceted(SurfaceKind::Plane { normal: *n }),
                            };
                            let ex = FacetExact::Ready(Arc::new(exact_plane(*o, *n)));
                            return Some((l.clone(), kind, ex));
                        }
                    }
                }
                t.label(plane, cen, a, b)
            }
        }
    }

    /// The exact surface of the sampled surface `i` this tool labelled a facet with.
    fn exact_surface(&self, i: usize) -> Option<exact::Exact> {
        match self {
            Tool::Sampled(t) => t.exact_of(i),
            Tool::Clipped(t, _) => t.exact_surface(i),
            _ => None,
        }
    }

    /// Replace labels by `f` (where it returns one).
    fn map_labels(&mut self, f: &dyn Fn(&HistorySource) -> Option<HistorySource>) {
        let mut fix = |l: &mut HistorySource| {
            if let Some(n) = f(l) {
                *l = n;
            }
        };
        match self {
            Tool::Exact(t) => {
                fix(&mut t.lo_label);
                fix(&mut t.hi_label);
                for l in &mut t.edge_labels {
                    for x in l {
                        fix(x);
                    }
                }
            }
            Tool::Sampled(t) => t.map_labels(&mut fix),
            Tool::Polys(t) => {
                for (_, l) in &mut t.facets {
                    fix(l);
                }
            }
            Tool::Clipped(t, planes) => {
                t.map_labels(f);
                for (_, _, l) in planes {
                    fix(l);
                }
            }
        }
    }

    fn relabel(&mut self, seed: u32, instance: &InstanceKey) {
        match self {
            Tool::Exact(t) => t.relabel(seed, instance),
            Tool::Sampled(t) => t.relabel(seed, instance),
            Tool::Clipped(t, _) => t.relabel(seed, instance),
            Tool::Polys(t) => {
                for (_, l) in &mut t.facets {
                    if let HistorySource::Generated {
                        seed: sd,
                        instance: inst,
                        ..
                    } = l
                    {
                        *sd = Some(seed);
                        *inst = Some(instance.clone());
                    }
                }
            }
        }
    }

    fn transformed(&self, t: &Transform) -> Tool {
        match self {
            Tool::Exact(p) => match p.permuted(t) {
                Some(q) => Tool::Exact(q),
                None => Tool::Sampled(Box::new(p.to_sampled().transformed(t))),
            },
            Tool::Sampled(s) => Tool::Sampled(Box::new(s.transformed(t))),
            Tool::Polys(p) => {
                let facets = p
                    .facets
                    .iter()
                    .map(|(f, l)| {
                        let mut g = f.clone();
                        for lp in &mut g.loops {
                            for q in lp.iter_mut() {
                                *q = t.apply(*q);
                            }
                        }
                        g.normal = t.apply_vector(f.normal);
                        (g, l.clone())
                    })
                    .collect();
                Tool::Polys(Box::new(sheet::PolyTool::new(facets)))
            }
            Tool::Clipped(inner, planes) => Tool::Clipped(
                Box::new(inner.transformed(t)),
                planes
                    .iter()
                    .map(|(o, n, l)| (t.apply(*o), t.apply_vector(*n), l.clone()))
                    .collect(),
            ),
        }
    }
}

fn gen(role: Role, tag: KernelTag, ordinal: Option<u32>) -> HistorySource {
    HistorySource::Generated {
        role,
        tag,
        ordinal,
        seed: None,
        instance: None,
    }
}

fn axis_of(v: [f64; 3]) -> Option<(usize, f64)> {
    let mut found = None;
    for (a, x) in v.iter().enumerate() {
        if (x.abs() - 1.0).abs() < 1e-9 {
            if found.is_some() {
                return None;
            }
            found = Some((a, x.signum()));
        } else if x.abs() > 1e-9 {
            return None;
        }
    }
    found
}

fn frame_axes(frame: &Frame) -> Result<(usize, f64, usize, usize), KernelError> {
    let unsupported = || KernelError::NotSupported {
        what: "fake kernel: frames not aligned to the part axes".into(),
    };
    let (k, ks) = axis_of(frame.z).ok_or_else(unsupported)?;
    let (u, _) = axis_of(frame.x).ok_or_else(unsupported)?;
    let (v, _) = axis_of(frame.y).ok_or_else(unsupported)?;
    Ok((k, ks, u, v))
}

fn prism_from_profile(
    profile: &CanonicalProfile,
    start: f64,
    end: f64,
    through_all: Option<(f64, f64)>,
    profile_index: u32,
) -> Result<Prism, KernelError> {
    let (k, ks, u, v) = frame_axes(&profile.frame)?;
    let mut loops = Vec::new();
    let mut labels = Vec::new();
    for l in &profile.loops {
        let mut pts = Vec::new();
        let mut ls = Vec::new();
        for s in l {
            if !matches!(s.kind, SegmentKind::Line) {
                return Err(KernelError::NotSupported {
                    what: "fake kernel: arcs and circles in profiles".into(),
                });
            }
            let a = profile.frame.to_world(s.start);
            let b = profile.frame.to_world(s.end);
            if (a[u] - b[u]).abs() > EPS && (a[v] - b[v]).abs() > EPS {
                return Err(KernelError::NotSupported {
                    what: "fake kernel: profile segments not parallel to an axis".into(),
                });
            }
            pts.push([a[u], a[v]]);
            ls.push(gen(
                Role::Side,
                KernelTag::Curve {
                    profile: profile_index,
                    curve: s.curve.clone(),
                },
                None,
            ));
        }
        loops.push(pts);
        labels.push(ls);
    }
    let o = profile.frame.origin[k];
    let (mut s0, mut s1) = (o + ks * start, o + ks * end);
    let (mut l0, mut l1) = (
        gen(Role::StartCap, KernelTag::None, None),
        gen(Role::EndCap, KernelTag::None, None),
    );
    if let Some((blo, bhi)) = through_all {
        // Extend the end cap past every operand.
        s1 = if (end - start) * ks > 0.0 { bhi } else { blo };
    }
    if s0 > s1 {
        std::mem::swap(&mut s0, &mut s1);
        std::mem::swap(&mut l0, &mut l1);
    }
    Ok(Prism {
        k,
        u,
        v,
        loops,
        edge_labels: labels,
        lo: s0,
        hi: s1,
        lo_label: l0,
        hi_label: l1,
    })
}

// ------------------------------------------------------------------------------- re-gridding

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    New,
    Add,
    Cut,
    Intersect,
    BoolUnion,
    BoolCut,
    BoolIntersect,
    Split,
    Keep,
}

struct GridJob<'a> {
    operands: Vec<&'a FakeSolid>,
    /// Operand indices into the operation's operand list (for labels).
    operand_ids: Vec<u32>,
    tools: Vec<Tool>,
    mode: Mode,
    /// Split plane: axis, offset, normal sign.
    split: Option<(usize, f64, f64)>,
    /// For imports: one tool part per body.
    separate_tools: bool,
    body_label: Option<Vec<HistorySource>>,
    /// A shell or draft applied to the (single) operand in `Mode::Keep`.
    modifier: Option<Modifier>,
    /// Side faces of collinear profile curves merge into one face (the definition asked).
    merge_collinear: bool,
    /// Tools removed from the operands before `tools` combine (a flange trimming its sheet).
    carve: Vec<Tool>,
}

#[derive(Clone, Debug)]
struct ShellSpec {
    open: BTreeSet<u32>,
    thickness: f64,
    /// Walls with their own thickness.
    per_face: BTreeMap<u32, f64>,
    /// Walls grow outside the body, which becomes the cavity.
    outward: bool,
}

impl ShellSpec {
    fn of(&self, face: u32) -> f64 {
        self.per_face.get(&face).copied().unwrap_or(self.thickness)
    }

    /// For an outward shell, whether `p` (outside the body) is wall: within a face's thickness
    /// of it (offset faces meeting in sharp corners, so the distance is the largest per axis),
    /// and not in front of an open face.
    fn outward_wall(&self, o: &FakeSolid, p: [f64; 3]) -> bool {
        let mut near = false;
        for (fi, f) in o.faces.iter().enumerate() {
            let t = self.of(fi as u32);
            if self.open.contains(&(fi as u32)) {
                if let Some(pk) = f.plane {
                    let ahead = (pk.dir as f64) * (p[pk.axis] - pk.offset) > EPS;
                    let across = f.rects.iter().any(|r| {
                        (0..3).all(|a| {
                            a == pk.axis || (p[a] > r.lo[a] - t - EPS && p[a] < r.hi[a] + t + EPS)
                        })
                    });
                    if ahead && across {
                        return false;
                    }
                }
                continue;
            }
            near |= face_cheb(p, f) <= t + EPS;
        }
        near
    }

    /// The face an outward wall facet at `p` on `plane` offsets: a face facing the same way
    /// at its thickness behind the facet, the nearest across it (corners go to the first).
    fn outer_of(&self, o: &FakeSolid, p: [f64; 3], plane: &PlaneKey) -> Option<u32> {
        let mut best: Option<(f64, u32)> = None;
        for (fi, f) in o.faces.iter().enumerate() {
            if self.open.contains(&(fi as u32)) {
                continue;
            }
            let Some(fp) = f
                .plane
                .filter(|fp| fp.axis == plane.axis && fp.dir == plane.dir)
            else {
                continue;
            };
            let behind = (plane.dir as f64) * (plane.offset - fp.offset);
            if (behind - self.of(fi as u32)).abs() > EPS {
                continue;
            }
            let d = face_cheb(p, f);
            if best.map(|b| d < b.0 - EPS).unwrap_or(true) {
                best = Some((d, fi as u32));
            }
        }
        best.map(|b| b.1)
    }
}

#[derive(Clone, Debug)]
struct DraftSpec {
    /// The drafted face's index in the operand.
    face: u32,
    plane: PlaneKey,
    lo: [f64; 3],
    hi: [f64; 3],
    origin: [f64; 3],
    pull: [f64; 3],
    tan: f64,
    reach: f64,
    /// Outward offset of the face plane (offset face, push-pull); zero for a draft.
    shift: f64,
}

impl DraftSpec {
    fn region(&self, p: [f64; 3]) -> bool {
        (0..3).all(|a| {
            if a == self.plane.axis {
                (p[a] - self.plane.offset).abs() <= self.reach + EPS
            } else {
                p[a] >= self.lo[a] - EPS && p[a] <= self.hi[a] + EPS
            }
        })
    }

    /// Within the face's extent along the pull, and across it out to the reach past its side
    /// edges, at any depth along the face's plane: where the draft's half-space decides the
    /// material either side of a facet (the far edge of an outward lean, mitred corners).
    fn flank(&self, p: [f64; 3]) -> bool {
        (0..3).all(|a| {
            if a == self.plane.axis {
                true
            } else if self.pull[a].abs() > 1e-12 {
                p[a] >= self.lo[a] - EPS && p[a] <= self.hi[a] + EPS
            } else {
                p[a] >= self.lo[a] - self.reach - EPS && p[a] <= self.hi[a] + self.reach + EPS
            }
        })
    }

    /// Past the face's side edges (across the pull, not along it), within reach of its
    /// plane: where drafts of neighbouring faces that lean out meet in a mitred corner.
    fn corner(&self, p: [f64; 3]) -> bool {
        !self.region(p)
            && (0..3).all(|a| {
                if a == self.plane.axis {
                    (p[a] - self.plane.offset).abs() <= self.reach + EPS
                } else if self.pull[a].abs() > 1e-12 {
                    p[a] >= self.lo[a] - EPS && p[a] <= self.hi[a] + EPS
                } else {
                    p[a] >= self.lo[a] - self.reach - EPS && p[a] <= self.hi[a] + self.reach + EPS
                }
            })
    }

    /// Whether `p` is material after the draft (near this face).
    fn material(&self, p: [f64; 3]) -> bool {
        let h = (0..3)
            .map(|a| (p[a] - self.origin[a]) * self.pull[a])
            .sum::<f64>();
        (self.plane.dir as f64) * (p[self.plane.axis] - self.plane.offset) - self.shift
            < -self.tan * h
    }

    fn normal(&self) -> [f64; 3] {
        let mut n = [0.0; 3];
        n[self.plane.axis] = self.plane.dir as f64;
        for a in 0..3 {
            n[a] += self.tan * self.pull[a];
        }
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        [n[0] / l, n[1] / l, n[2] / l]
    }
}

#[derive(Clone, Debug)]
enum Modifier {
    Shell(ShellSpec),
    Draft(Vec<DraftSpec>),
    /// Delete faces and heal: the region the deleted faces bound is filled (a cavity) or
    /// cleared (a protrusion); neighbours extend over it on their own planes.
    Fill(FillSpec),
}

#[derive(Clone, Debug)]
struct FillSpec {
    faces: BTreeSet<u32>,
    lo: [f64; 3],
    hi: [f64; 3],
    cavity: bool,
}

fn rect_distance(p: [f64; 3], r: &Rect) -> f64 {
    let mut s = 0.0;
    for a in 0..3 {
        let d = (r.lo[a] - p[a]).max(p[a] - r.hi[a]).max(0.0);
        s += d * d;
    }
    s.sqrt()
}

/// The largest per-axis distance from `p` to a face's rects.
fn face_cheb(p: [f64; 3], f: &FFace) -> f64 {
    f.rects
        .iter()
        .map(|r| {
            (0..3)
                .map(|a| (r.lo[a] - p[a]).max(p[a] - r.hi[a]).max(0.0))
                .fold(0.0, f64::max)
        })
        .fold(f64::INFINITY, f64::min)
}

fn face_distance(p: [f64; 3], f: &FFace) -> f64 {
    f.rects
        .iter()
        .map(|r| rect_distance(p, r))
        .fold(f64::INFINITY, f64::min)
}

struct GridResult {
    solids: Vec<FakeSolid>,
    face_sources: Vec<Vec<Vec<HistorySource>>>,
    body_sources: Vec<Vec<HistorySource>>,
}

fn dedup_sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.retain(|x| x.is_finite());
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let mut out: Vec<f64> = Vec::new();
    for x in v {
        if out.last().map(|l| (x - l).abs() > EPS).unwrap_or(true) {
            out.push(x);
        }
    }
    out
}

/// Whether two coplanar, adjacent planar facets belong to one face. Faces from different
/// sources merge where they are flush (a tool cap flush with an operand face, two operands
/// meeting in a union). Two faces that were already separate stay separate: distinct faces of
/// one operand, and side faces of distinct profile curves (collinear sketch lines), unless the
/// definition asks for collinear sides to merge.
fn coplanar_may_merge(a: &[HistorySource], b: &[HistorySource], merge_collinear: bool) -> bool {
    if a.iter().any(|l| b.contains(l)) {
        return true;
    }
    let side_curve = |l: &HistorySource| {
        matches!(
            l,
            HistorySource::Generated {
                role: Role::Side,
                tag: KernelTag::Curve { .. },
                ..
            }
        )
    };
    if !a.is_empty() && !b.is_empty() && a.iter().all(side_curve) && b.iter().all(side_curve) {
        return merge_collinear;
    }
    let operand_of = |ls: &[HistorySource]| -> Option<u32> {
        let mut op = None;
        for l in ls {
            match l {
                HistorySource::Input { from } => match op {
                    None => op = Some(from.operand),
                    Some(o) if o == from.operand => {}
                    Some(_) => return None,
                },
                _ => return None,
            }
        }
        op
    };
    match (operand_of(a), operand_of(b)) {
        (Some(x), Some(y)) => x != y,
        _ => true,
    }
}

fn run_grid(job: GridJob<'_>) -> Result<GridResult, KernelError> {
    let mut breaks: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for o in &job.operands {
        for a in 0..3 {
            breaks[a].extend(o.grid[a].iter().copied());
        }
    }
    for t in job.tools.iter().chain(&job.carve) {
        t.breaks(&mut breaks);
    }
    if let Some((a, c, _)) = job.split {
        breaks[a].push(c);
    }
    match &job.modifier {
        Some(Modifier::Shell(sh)) => {
            if let Some(o) = job.operands.first() {
                for (fi, f) in o.faces.iter().enumerate() {
                    if sh.open.contains(&(fi as u32)) {
                        continue;
                    }
                    let sign = if sh.outward { -1.0 } else { 1.0 };
                    for (pk, _) in f.rect_planes() {
                        breaks[pk.axis].push(pk.offset - sign * (pk.dir as f64) * sh.of(fi as u32));
                    }
                }
            }
        }
        Some(Modifier::Fill(fs)) => {
            for a in 0..3 {
                breaks[a].push(fs.lo[a]);
                breaks[a].push(fs.hi[a]);
            }
        }
        Some(Modifier::Draft(ds)) => {
            for d in ds {
                // The moved plane of an offset face.
                breaks[d.plane.axis].push(d.plane.offset + (d.plane.dir as f64) * d.shift);
                for a in 0..3 {
                    let (lo, hi) = if a == d.plane.axis {
                        (d.plane.offset - d.reach, d.plane.offset + d.reach)
                    } else {
                        (d.lo[a], d.hi[a])
                    };
                    for k in 0..=sampled::RES {
                        breaks[a].push(lo + (hi - lo) * k as f64 / sampled::RES as f64);
                    }
                }
            }
        }
        None => {}
    }
    let grid: [Vec<f64>; 3] = [
        dedup_sorted(breaks[0].clone()),
        dedup_sorted(breaks[1].clone()),
        dedup_sorted(breaks[2].clone()),
    ];
    let d = [
        grid[0].len().saturating_sub(1),
        grid[1].len().saturating_sub(1),
        grid[2].len().saturating_sub(1),
    ];
    let n = d[0] * d[1] * d[2];
    if n == 0 {
        return Ok(GridResult {
            solids: Vec::new(),
            face_sources: Vec::new(),
            body_sources: Vec::new(),
        });
    }
    if n > 4_000_000 {
        return Err(KernelError::NotSupported {
            what: "fake kernel: grid too large".into(),
        });
    }
    let idx = |c: [usize; 3]| c[0] + d[0] * (c[1] + d[1] * c[2]);
    let center = |c: [usize; 3]| -> [f64; 3] {
        [
            (grid[0][c[0]] + grid[0][c[0] + 1]) / 2.0,
            (grid[1][c[1]] + grid[1][c[1] + 1]) / 2.0,
            (grid[2][c[2]] + grid[2][c[2] + 1]) / 2.0,
        ]
    };
    let mut owners = vec![0u64; n];
    let mut tools = vec![0u64; n];
    let mut occ = vec![false; n];
    for z in 0..d[2] {
        for y in 0..d[1] {
            for x in 0..d[0] {
                let c = [x, y, z];
                let p = center(c);
                let i = idx(c);
                let carved = job.carve.iter().any(|t| t.inside(p));
                for (k, o) in job.operands.iter().enumerate() {
                    if o.contains(p) && !carved {
                        owners[i] |= 1 << k;
                    }
                }
                for (k, t) in job.tools.iter().enumerate() {
                    if t.inside(p) {
                        tools[i] |= 1 << k.min(63);
                    }
                }
                let o = owners[i] != 0;
                let t = tools[i] != 0;
                occ[i] = match job.mode {
                    Mode::New => t,
                    Mode::Add => o || t,
                    Mode::Cut => o && !t,
                    Mode::Intersect => o && t,
                    Mode::BoolUnion | Mode::Split | Mode::Keep => o,
                    Mode::BoolCut => owners[i] & 1 != 0 && owners[i] & !1 == 0,
                    Mode::BoolIntersect => owners[i] & 1 != 0 && owners[i] & !1 != 0,
                };
                if matches!(job.mode, Mode::BoolCut | Mode::BoolIntersect) && occ[i] {
                    owners[i] = 1;
                }
                match &job.modifier {
                    Some(Modifier::Shell(sh)) if sh.outward => {
                        // The body becomes the cavity; walls grow around it.
                        let o = job.operands[0];
                        occ[i] = !occ[i] && sh.outward_wall(o, p);
                        owners[i] = if occ[i] { 1 } else { 0 };
                    }
                    Some(Modifier::Shell(sh)) if occ[i] => {
                        let o = job.operands[0];
                        let cavity = o.faces.iter().enumerate().all(|(fi, f)| {
                            sh.open.contains(&(fi as u32))
                                || !f.is_grid()
                                || face_distance(p, f) > sh.of(fi as u32) + EPS
                        });
                        if cavity {
                            occ[i] = false;
                        }
                    }
                    Some(Modifier::Fill(fs)) => {
                        if (0..3).all(|a| p[a] > fs.lo[a] - EPS && p[a] < fs.hi[a] + EPS) {
                            occ[i] = fs.cavity;
                            if fs.cavity {
                                owners[i] |= 1;
                            } else {
                                owners[i] = 0;
                            }
                        }
                    }
                    Some(Modifier::Draft(ds)) => {
                        // Near drafted faces the material is what every draft whose region
                        // holds the point leaves (corners where two drafts meet take both).
                        let mut decided = None;
                        for d in ds.iter().filter(|d| d.region(p)) {
                            decided = Some(decided.unwrap_or(true) && d.material(p));
                        }
                        if let Some(m) = decided {
                            occ[i] = m;
                            if m {
                                owners[i] |= 1;
                            }
                        } else if !occ[i] {
                            // Past the side edges of faces leaning out, the corner between
                            // two of them is material where both leave it (a mitre); one
                            // draft alone adds nothing past its face's edges.
                            let mut n = 0;
                            let mut all = true;
                            for d in ds.iter().filter(|d| d.corner(p)) {
                                n += 1;
                                all &= d.material(p);
                            }
                            if n >= 2 && all {
                                occ[i] = true;
                                owners[i] |= 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Components.
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while p[r] != r {
            r = p[r];
        }
        let mut y = x;
        while p[y] != r {
            let nx = p[y];
            p[y] = r;
            y = nx;
        }
        r
    }
    let connect = |a: usize, b: usize, axis: usize, boundary: usize| -> bool {
        if let Some((sa, c, _)) = job.split {
            if sa == axis && (grid[axis][boundary] - c).abs() < EPS {
                return false;
            }
        }
        match job.mode {
            Mode::New => !job.separate_tools || tools[a] & tools[b] != 0,
            Mode::Add => owners[a] & owners[b] != 0 || tools[a] != 0 || tools[b] != 0,
            Mode::BoolUnion => true,
            _ => owners[a] & owners[b] != 0,
        }
    };
    for z in 0..d[2] {
        for y in 0..d[1] {
            for x in 0..d[0] {
                let c = [x, y, z];
                let i = idx(c);
                if !occ[i] {
                    continue;
                }
                for a in 0..3 {
                    if c[a] + 1 < d[a] {
                        let mut nc = c;
                        nc[a] += 1;
                        let j = idx(nc);
                        if occ[j] && connect(i, j, a, c[a] + 1) {
                            let (ra, rb) = (find(&mut parent, i), find(&mut parent, j));
                            if ra != rb {
                                parent[ra.max(rb)] = ra.min(rb);
                            }
                        }
                    }
                }
            }
        }
    }
    let mut comp_of = vec![usize::MAX; n];
    let mut roots: Vec<usize> = Vec::new();
    for i in 0..n {
        if occ[i] {
            let r = find(&mut parent, i);
            let k = match roots.iter().position(|x| *x == r) {
                Some(k) => k,
                None => {
                    roots.push(r);
                    roots.len() - 1
                }
            };
            comp_of[i] = k;
        }
    }
    let ncomp = roots.len();

    // Facets.
    struct Facet {
        cell: usize,
        c: [usize; 3],
        axis: usize,
        dir: i8,
        comp: usize,
    }
    let mut facets: Vec<Facet> = Vec::new();
    let mut facet_at: BTreeMap<(usize, usize, i8), usize> = BTreeMap::new();
    for z in 0..d[2] {
        for y in 0..d[1] {
            for x in 0..d[0] {
                let c = [x, y, z];
                let i = idx(c);
                if !occ[i] {
                    continue;
                }
                for a in 0..3 {
                    for dir in [-1i8, 1] {
                        let other = if dir < 0 {
                            if c[a] == 0 {
                                None
                            } else {
                                let mut nc = c;
                                nc[a] -= 1;
                                Some(idx(nc))
                            }
                        } else if c[a] + 1 >= d[a] {
                            None
                        } else {
                            let mut nc = c;
                            nc[a] += 1;
                            Some(idx(nc))
                        };
                        let exposed = match other {
                            None => true,
                            Some(j) => !occ[j] || comp_of[j] != comp_of[i],
                        };
                        if exposed {
                            facet_at.insert((i, a, dir), facets.len());
                            facets.push(Facet {
                                cell: i,
                                c,
                                axis: a,
                                dir,
                                comp: comp_of[i],
                            });
                        }
                    }
                }
            }
        }
    }
    let facet_plane = |f: &Facet| -> PlaneKey {
        let b = if f.dir > 0 {
            f.c[f.axis] + 1
        } else {
            f.c[f.axis]
        };
        PlaneKey {
            axis: f.axis,
            dir: f.dir,
            offset: grid[f.axis][b],
        }
    };
    let facet_rect = |f: &Facet| -> Rect {
        let mut lo = [0.0; 3];
        let mut hi = [0.0; 3];
        for a in 0..3 {
            lo[a] = grid[a][f.c[a]];
            hi[a] = grid[a][f.c[a] + 1];
        }
        let p = facet_plane(f);
        lo[f.axis] = p.offset;
        hi[f.axis] = p.offset;
        Rect { lo, hi }
    };

    // Labels.
    let drafts: &[DraftSpec] = match &job.modifier {
        Some(Modifier::Draft(ds)) => ds,
        _ => &[],
    };
    let draft_reach = drafts.iter().map(|d| d.reach).fold(0.0, f64::max);
    let mut labels: Vec<Vec<HistorySource>> = Vec::with_capacity(facets.len());
    let mut kinds: Vec<FacetKind> = Vec::with_capacity(facets.len());
    let mut exacts: Vec<FacetExact> = Vec::with_capacity(facets.len());
    for f in &facets {
        let plane = facet_plane(f);
        let r = facet_rect(f);
        let mut cen = [0.0; 3];
        for a in 0..3 {
            cen[a] = (r.lo[a] + r.hi[a]) / 2.0;
        }
        let cin = center(f.c);
        // The cell across the facet (or the mirror point past the grid's edge).
        let cout = {
            let mut nc = f.c;
            let inside_grid = if f.dir > 0 {
                nc[f.axis] += 1;
                nc[f.axis] < d[f.axis]
            } else if nc[f.axis] > 0 {
                nc[f.axis] -= 1;
                true
            } else {
                false
            };
            if inside_grid {
                center(nc)
            } else {
                let mut m = cin;
                m[plane.axis] = 2.0 * plane.offset - cin[plane.axis];
                m
            }
        };
        let mut ls: Vec<HistorySource> = Vec::new();
        let mut kind = FacetKind::Planar;
        let mut ex = FacetExact::None;
        // A drafted face's tilted surface: the draft deciding either side of the facet whose
        // material changes across it. On the face's flank (see [`DraftSpec::flank`]) the
        // draft's half-space decides; elsewhere it leaves the facet to the operand's faces.
        for d in drafts
            .iter()
            .filter(|d| d.region(cin) || d.region(cout) || d.corner(cin) || d.corner(cout))
        {
            let side = |p: [f64; 3]| !d.flank(p) || d.material(p);
            if side(cin) != side(cout) {
                ls.push(HistorySource::Input {
                    from: OperandEntity {
                        operand: job.operand_ids[0],
                        entity: d.face,
                    },
                });
                kind = FacetKind::Faceted(SurfaceKind::Plane { normal: d.normal() });
                break;
            }
        }
        if ls.is_empty() {
            for (k, o) in job.operands.iter().enumerate() {
                let bool_mode = matches!(
                    job.mode,
                    Mode::BoolUnion | Mode::BoolCut | Mode::BoolIntersect
                );
                if owners[f.cell] & (1 << k) == 0 && !matches!(job.mode, Mode::Add) && !bool_mode {
                    continue;
                }
                for (fi, face) in o.faces.iter().enumerate() {
                    if let Some(Modifier::Fill(fs)) = &job.modifier {
                        if fs.faces.contains(&(fi as u32)) {
                            continue;
                        }
                    }
                    if face.lies_on(&plane, cen) {
                        ls.push(HistorySource::Input {
                            from: OperandEntity {
                                operand: job.operand_ids[k],
                                entity: fi as u32,
                            },
                        });
                        if let FacetKind::Faceted(h) = face.kind() {
                            kind = FacetKind::Faceted(h);
                            if let Some(e) = &face.exact {
                                ex = FacetExact::Ready(e.clone());
                            }
                        }
                    }
                }
            }
        }
        if ls.is_empty() && job.mode == Mode::BoolCut {
            // Cavity walls lie on the tool bodies' faces, facing the other way.
            let flipped = PlaneKey {
                dir: -plane.dir,
                ..plane
            };
            for (k, o) in job.operands.iter().enumerate().skip(1) {
                for (fi, face) in o.faces.iter().enumerate() {
                    if face.lies_on(&flipped, cen) {
                        ls.push(HistorySource::Input {
                            from: OperandEntity {
                                operand: job.operand_ids[k],
                                entity: fi as u32,
                            },
                        });
                        if let FacetKind::Faceted(h) = face.kind() {
                            kind = FacetKind::Faceted(h);
                            if let Some(e) = &face.exact {
                                ex = FacetExact::Ready(e.clone());
                            }
                        }
                    }
                }
            }
        }
        if ls.is_empty() && !drafts.is_empty() {
            // A neighbour of a drafted face, extended or trimmed on its own plane (out to a
            // mitred corner: within the reach along each axis).
            let o = job.operands[0];
            let mut best: Option<(f64, usize)> = None;
            for (fi, face) in o.faces.iter().enumerate() {
                if face.plane.map(|p| p.same(&plane)).unwrap_or(false) {
                    let d = face_distance(cen, face);
                    if d <= draft_reach * 3f64.sqrt() + EPS && best.map(|b| d < b.0).unwrap_or(true)
                    {
                        best = Some((d, fi));
                    }
                }
            }
            if let Some((_, fi)) = best {
                ls.push(HistorySource::Input {
                    from: OperandEntity {
                        operand: job.operand_ids[0],
                        entity: fi as u32,
                    },
                });
            }
        }
        if let (true, Some(Modifier::Fill(fs))) = (ls.is_empty(), &job.modifier) {
            let o = job.operands[0];
            let mut best: Option<(f64, usize)> = None;
            for (fi, face) in o.faces.iter().enumerate() {
                if fs.faces.contains(&(fi as u32)) {
                    continue;
                }
                if face.plane.map(|p| p.same(&plane)).unwrap_or(false) {
                    let d = face_distance(cen, face);
                    if best.map(|b| d < b.0).unwrap_or(true) {
                        best = Some((d, fi));
                    }
                }
            }
            if let Some((_, fi)) = best {
                ls.push(HistorySource::Input {
                    from: OperandEntity {
                        operand: job.operand_ids[0],
                        entity: fi as u32,
                    },
                });
            }
        }
        if let (true, Some(Modifier::Shell(sh))) = (ls.is_empty(), &job.modifier) {
            if sh.outward {
                let o = job.operands[0];
                let rim = o.faces.iter().enumerate().find(|(fi, f)| {
                    sh.open.contains(&(*fi as u32))
                        && f.plane.map(|p| p.same(&plane)).unwrap_or(false)
                });
                if o.contains(cout) {
                    // The cavity: the body's own faces, now facing into it.
                    let flipped = PlaneKey {
                        dir: -plane.dir,
                        ..plane
                    };
                    for (fi, face) in o.faces.iter().enumerate() {
                        if face.lies_on(&flipped, cen) {
                            ls.push(HistorySource::Input {
                                from: OperandEntity {
                                    operand: job.operand_ids[0],
                                    entity: fi as u32,
                                },
                            });
                        }
                    }
                } else if let Some((fi, _)) = rim {
                    let from = OperandEntity {
                        operand: job.operand_ids[0],
                        entity: fi as u32,
                    };
                    ls.push(gen(
                        Role::ShellRim,
                        KernelTag::InputFace { face: from },
                        None,
                    ));
                } else if let Some(fi) = sh.outer_of(o, cen, &plane) {
                    let from = OperandEntity {
                        operand: job.operand_ids[0],
                        entity: fi,
                    };
                    ls.push(gen(
                        Role::ShellOuter,
                        KernelTag::InputFace { face: from },
                        None,
                    ));
                }
            }
        }
        let inward_shell = job
            .modifier
            .as_ref()
            .filter(|m| !matches!(m, Modifier::Shell(sh) if sh.outward));
        if let Some(Modifier::Shell(sh)) = inward_shell {
            let o = job.operands[0];
            for l in ls.iter_mut() {
                if let HistorySource::Input { from } = l {
                    if sh.open.contains(&from.entity) {
                        *l = gen(Role::ShellRim, KernelTag::InputFace { face: *from }, None);
                    }
                }
            }
            if ls.is_empty() {
                let mut best: Option<(f64, usize)> = None;
                for (fi, face) in o.faces.iter().enumerate() {
                    if sh.open.contains(&(fi as u32)) || !face.is_grid() {
                        continue;
                    }
                    let d = (face_distance(cen, face) - sh.of(fi as u32)).abs();
                    if best.map(|b| d < b.0 - EPS).unwrap_or(true) {
                        best = Some((d, fi));
                    }
                }
                if let Some((_, fi)) = best {
                    let face = &o.faces[fi];
                    kind = match face.plane {
                        Some(p) if p.axis == plane.axis => FacetKind::Planar,
                        _ => FacetKind::Faceted(SurfaceKind::Other),
                    };
                    ls.push(gen(
                        Role::ShellInner,
                        KernelTag::InputFace {
                            face: OperandEntity {
                                operand: job.operand_ids[0],
                                entity: fi as u32,
                            },
                        },
                        None,
                    ));
                }
            }
        }
        if ls.is_empty() {
            for (ti, t) in job.tools.iter().chain(&job.carve).enumerate() {
                if let Some((l, k, e)) = t.label(&plane, cen, cin, cout) {
                    ls.push(l);
                    kind = k;
                    ex = match e {
                        FacetExact::Surface(i) => FacetExact::Tool(ti, i),
                        other => other,
                    };
                    break;
                }
            }
        }
        if ls.is_empty() {
            if let Some((sa, c, s)) = job.split {
                if sa == plane.axis && (plane.offset - c).abs() < EPS {
                    // The facet's solid is on the side opposite its outward direction.
                    let solid_positive = (plane.dir as f64) * s < 0.0;
                    let side = if solid_positive {
                        PlaneSide::Positive
                    } else {
                        PlaneSide::Negative
                    };
                    ls.push(gen(Role::SplitFace, KernelTag::Side { side }, None));
                }
            }
        }
        if ls.is_empty() {
            return Err(KernelError::HistoryIncomplete {
                message: format!("fake kernel: facet at {:?} has no source", cen),
            });
        }
        ls.sort();
        ls.dedup();
        labels.push(ls);
        kinds.push(kind);
        exacts.push(if kind == FacetKind::Planar {
            FacetExact::None
        } else {
            ex
        });
    }

    // Faces: planar facets join coplanar neighbours (so flush faces merge, see
    // [`coplanar_may_merge`]); a faceted facet
    // joins any facet sharing a facet edge and a label (one curved surface, one face).
    let mut fparent: Vec<usize> = (0..facets.len()).collect();
    for (fi, f) in facets.iter().enumerate() {
        if kinds[fi] != FacetKind::Planar {
            continue;
        }
        for b in 0..3 {
            if b == f.axis || f.c[b] + 1 >= d[b] {
                continue;
            }
            let mut nc = f.c;
            nc[b] += 1;
            if let Some(&gj) = facet_at.get(&(idx(nc), f.axis, f.dir)) {
                if facets[gj].comp == f.comp
                    && kinds[gj] == FacetKind::Planar
                    && facet_plane(&facets[gj]).same(&facet_plane(f))
                    && coplanar_may_merge(&labels[fi], &labels[gj], job.merge_collinear)
                {
                    let (ra, rb) = (find(&mut fparent, fi), find(&mut fparent, gj));
                    if ra != rb {
                        fparent[ra.max(rb)] = ra.min(rb);
                    }
                }
            }
        }
    }
    if kinds.iter().any(|k| *k != FacetKind::Planar) {
        let mut seg_facets: BTreeMap<SegmentKey, Vec<usize>> = BTreeMap::new();
        for (fi, f) in facets.iter().enumerate() {
            let ia = if f.dir > 0 {
                f.c[f.axis] + 1
            } else {
                f.c[f.axis]
            };
            let others: Vec<usize> = (0..3).filter(|a| *a != f.axis).collect();
            for (along, fixed) in [(others[0], others[1]), (others[1], others[0])] {
                for off in 0..2 {
                    let mut pt = f.c;
                    pt[f.axis] = ia;
                    pt[fixed] = f.c[fixed] + off;
                    seg_facets.entry((along, pt)).or_default().push(fi);
                }
            }
        }
        for list in seg_facets.values() {
            for (x, &i) in list.iter().enumerate() {
                for &j in &list[x + 1..] {
                    if facets[i].comp != facets[j].comp
                        || (kinds[i] == FacetKind::Planar && kinds[j] == FacetKind::Planar)
                    {
                        continue;
                    }
                    if labels[i].iter().any(|l| labels[j].contains(l)) {
                        let (ra, rb) = (find(&mut fparent, i), find(&mut fparent, j));
                        if ra != rb {
                            fparent[ra.max(rb)] = ra.min(rb);
                        }
                    }
                }
            }
        }
    }
    // Face order: by component, then first facet.
    let mut face_of_root: BTreeMap<usize, usize> = BTreeMap::new();
    let mut comp_faces: Vec<Vec<usize>> = vec![Vec::new(); ncomp]; // roots per comp in order
    let mut facet_face = vec![0usize; facets.len()];
    for fi in 0..facets.len() {
        let r = find(&mut fparent, fi);
        let comp = facets[fi].comp;
        let local = match face_of_root.get(&r) {
            Some(l) => *l,
            None => {
                let l = comp_faces[comp].len();
                comp_faces[comp].push(r);
                face_of_root.insert(r, l);
                l
            }
        };
        facet_face[fi] = local;
    }

    let mut solids = Vec::with_capacity(ncomp);
    let mut face_sources = Vec::with_capacity(ncomp);
    let mut body_sources = Vec::with_capacity(ncomp);
    for comp in 0..ncomp {
        let nf = comp_faces[comp].len();
        let mut faces: Vec<FFace> = vec![FFace::blend(); nf];
        let mut fsrc: Vec<BTreeSet<HistorySource>> = vec![BTreeSet::new(); nf];
        let mut planar = vec![true; nf];
        for (fi, f) in facets.iter().enumerate() {
            if f.comp != comp {
                continue;
            }
            let l = facet_face[fi];
            let pk = facet_plane(f);
            if let FacetKind::Faceted(h) = kinds[fi] {
                planar[l] = false;
                if faces[l].surface.is_none() {
                    faces[l].surface = Some(h);
                }
                if faces[l].exact.is_none() {
                    faces[l].exact = match &exacts[fi] {
                        FacetExact::Ready(e) => Some(e.clone()),
                        FacetExact::Tool(ti, i) => job
                            .tools
                            .iter()
                            .chain(&job.carve)
                            .nth(*ti)
                            .and_then(|t| t.exact_surface(*i))
                            .map(Arc::new),
                        _ => None,
                    };
                }
            }
            if let Some(p0) = faces[l]
                .plane
                .or_else(|| faces[l].facet_planes.first().copied())
            {
                if !p0.same(&pk) {
                    planar[l] = false;
                }
            }
            if faces[l].plane.is_none() && faces[l].facet_planes.is_empty() {
                faces[l].plane = Some(pk);
            }
            faces[l].facet_planes.push(pk);
            faces[l].rects.push(facet_rect(f));
            fsrc[l].extend(labels[fi].iter().cloned());
        }
        for (l, face) in faces.iter_mut().enumerate() {
            if planar[l] {
                face.facet_planes.clear();
                face.surface = None;
                face.exact = None;
            } else {
                face.plane = None;
            }
        }
        let mut cocc = vec![false; n];
        let mut bsrc: BTreeSet<HistorySource> = BTreeSet::new();
        let mut has_tool_only = false;
        for i in 0..n {
            if comp_of[i] == comp {
                cocc[i] = true;
                for k in 0..job.operands.len() {
                    if owners[i] & (1 << k) != 0 {
                        bsrc.insert(HistorySource::Input {
                            from: OperandEntity {
                                operand: job.operand_ids[k],
                                entity: 0,
                            },
                        });
                    }
                }
                if owners[i] == 0 {
                    has_tool_only = true;
                }
            }
        }
        let mut bvec: Vec<HistorySource> = bsrc.into_iter().collect();
        if bvec.is_empty() && has_tool_only {
            match &job.body_label {
                Some(per_tool) => {
                    let first = (0..n)
                        .find(|i| comp_of[*i] == comp)
                        .expect("component has cells");
                    let t = tools[first].trailing_zeros() as usize;
                    bvec.push(
                        per_tool
                            .get(t)
                            .cloned()
                            .unwrap_or_else(|| gen(Role::Body, KernelTag::None, None)),
                    );
                }
                None => bvec.push(gen(Role::Body, KernelTag::None, None)),
            }
        }
        // Split side tag for bodies of an operand cut in two.
        if let Some((a, c, s)) = job.split {
            let first = (0..n)
                .find(|i| comp_of[*i] == comp)
                .expect("component has cells");
            let ci = [first % d[0], (first / d[0]) % d[1], first / (d[0] * d[1])];
            let p = center(ci);
            let side = if (p[a] - c) * s > 0.0 {
                PlaneSide::Positive
            } else {
                PlaneSide::Negative
            };
            bvec.push(gen(Role::Body, KernelTag::Side { side }, None));
        }
        let solid = FakeSolid {
            grid: grid.clone(),
            occ: cocc,
            faces,
            edge_meta: Vec::new(),
            decorations: Vec::new(),
            topology: Topology::default(),
        };
        solids.push(solid);
        face_sources.push(fsrc.into_iter().map(|s| s.into_iter().collect()).collect());
        body_sources.push(bvec);
    }
    Ok(GridResult {
        solids,
        face_sources,
        body_sources,
    })
}

/// Edges and vertices from a solid's grid faces, then decorations applied.
fn build_topology(s: &mut FakeSolid) {
    let grid = &s.grid;
    let gi = |a: usize, x: f64| -> usize { grid[a].partition_point(|g| *g < x - EPS) };
    // Segment map: (along, start grid point) -> [(face, cell key)].
    let mut segs: BTreeMap<SegmentKey, Vec<(u32, [usize; 3])>> = BTreeMap::new();
    for (fi, f) in s.faces.iter().enumerate() {
        for (p, r) in f.rect_planes() {
            let r = &r;
            let mut c = [0usize; 3];
            for a in 0..3 {
                c[a] = gi(a, r.lo[a]);
            }
            let ia = gi(p.axis, p.offset);
            // The cell this facet bounds.
            let mut cell = c;
            cell[p.axis] = if p.dir > 0 { ia.saturating_sub(1) } else { ia };
            let others: Vec<usize> = (0..3).filter(|a| *a != p.axis).collect();
            for (along, fixed) in [(others[0], others[1]), (others[1], others[0])] {
                for off in 0..2 {
                    let mut pt = c;
                    pt[p.axis] = ia;
                    pt[fixed] = c[fixed] + off;
                    segs.entry((along, pt)).or_default().push((fi as u32, cell));
                }
            }
        }
    }
    // Pieces between two different faces.
    let mut pieces: BTreeMap<PieceKey, Vec<usize>> = BTreeMap::new();
    for ((along, pt), list) in &segs {
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        if list.len() == 2 {
            pairs.push((list[0].0, list[1].0));
        } else if list.len() == 4 {
            let mut by_cell: BTreeMap<[usize; 3], Vec<u32>> = BTreeMap::new();
            for (f, c) in list {
                by_cell.entry(*c).or_default().push(*f);
            }
            for fs in by_cell.values() {
                if fs.len() == 2 {
                    pairs.push((fs[0], fs[1]));
                }
            }
        }
        for (a, b) in pairs {
            if a == b {
                continue;
            }
            let key = (a.min(b), a.max(b));
            let mut line = *pt;
            line[*along] = 0;
            pieces
                .entry((key, *along, line))
                .or_default()
                .push(pt[*along]);
        }
    }
    // Runs.
    struct Run {
        faces: (u32, u32),
        along: usize,
        line: [usize; 3],
        lo: usize,
        hi: usize,
    }
    let mut runs: Vec<Run> = Vec::new();
    for ((faces, along, line), mut starts) in pieces {
        starts.sort();
        starts.dedup();
        let mut i = 0;
        while i < starts.len() {
            let lo = starts[i];
            let mut hi = lo + 1;
            while i + 1 < starts.len() && starts[i + 1] == hi {
                i += 1;
                hi += 1;
            }
            runs.push(Run {
                faces,
                along,
                line,
                lo,
                hi,
            });
            i += 1;
        }
    }
    // Split runs at endpoints of other runs lying inside them.
    let point = |r: &Run, t: usize| -> [usize; 3] {
        let mut p = r.line;
        p[r.along] = t;
        p
    };
    let ends: BTreeSet<[usize; 3]> = runs
        .iter()
        .flat_map(|r| [point(r, r.lo), point(r, r.hi)])
        .collect();
    let mut split_runs: Vec<Run> = Vec::new();
    for r in runs {
        let mut cuts: Vec<usize> = (r.lo + 1..r.hi)
            .filter(|t| ends.contains(&point(&r, *t)))
            .collect();
        cuts.push(r.hi);
        let mut lo = r.lo;
        for c in cuts {
            split_runs.push(Run {
                faces: r.faces,
                along: r.along,
                line: r.line,
                lo,
                hi: c,
            });
            lo = c;
        }
    }
    let world = |p: [usize; 3]| -> [f64; 3] { [grid[0][p[0]], grid[1][p[1]], grid[2][p[2]]] };

    // Faces geometry.
    let mut topo = Topology::default();
    for f in &s.faces {
        topo.faces.push(FaceTopo { geom: face_geom(f) });
    }
    // Runs touching a faceted face are chained into one edge per connected run between the
    // same two faces (a staircase approximating one curve is one edge). A chain continues
    // through a point only where exactly two runs meet.
    let faceted = |f: u32| {
        let face = &s.faces[f as usize];
        face.plane.is_none() && face.is_grid()
    };
    let mut degree: BTreeMap<[usize; 3], usize> = BTreeMap::new();
    for r in &split_runs {
        *degree.entry(point(r, r.lo)).or_default() += 1;
        *degree.entry(point(r, r.hi)).or_default() += 1;
    }
    let mut cparent: Vec<usize> = (0..split_runs.len()).collect();
    fn cfind(p: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while p[r] != r {
            r = p[r];
        }
        p[x] = r;
        r
    }
    {
        type ChainKey = ([usize; 3], (u32, u32));
        let mut at: BTreeMap<ChainKey, Vec<usize>> = BTreeMap::new();
        for (i, r) in split_runs.iter().enumerate() {
            if !(faceted(r.faces.0) || faceted(r.faces.1)) {
                continue;
            }
            for t in [r.lo, r.hi] {
                let pt = point(r, t);
                if degree.get(&pt).copied() == Some(2) {
                    at.entry((pt, r.faces)).or_default().push(i);
                }
            }
        }
        for list in at.values() {
            if list.len() == 2 {
                let (a, b) = (cfind(&mut cparent, list[0]), cfind(&mut cparent, list[1]));
                if a != b {
                    cparent[a.max(b)] = a.min(b);
                }
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..split_runs.len() {
        let r = cfind(&mut cparent, i);
        groups.entry(r).or_default().push(i);
    }
    let mut edges: Vec<EdgeRow> = Vec::new();
    for members in groups.values() {
        let r = &split_runs[members[0]];
        if members.len() == 1 {
            let a = point(r, r.lo);
            let b = point(r, r.hi);
            let (wa, wb) = (world(a), world(b));
            let pa = s.faces[r.faces.0 as usize].plane;
            let pb = s.faces[r.faces.1 as usize].plane;
            edges.push((
                EdgeTopo {
                    faces: [r.faces.0, r.faces.1],
                    geom: EdgeGeom {
                        start: wa,
                        end: wb,
                        length: (wb[r.along] - wa[r.along]).abs(),
                    },
                    laminar: false,
                },
                EdgeMeta {
                    along: r.along,
                    at: wa,
                    lo: wa[r.along],
                    hi: wb[r.along],
                    planes: match (pa, pb) {
                        (Some(x), Some(y)) => Some([x, y]),
                        _ => None,
                    },
                },
                vec![a, b],
            ));
            continue;
        }
        // A chain: its ends are the points used once within it; a closed chain has none.
        let mut uses: BTreeMap<[usize; 3], usize> = BTreeMap::new();
        let mut length = 0.0;
        for &m in members {
            let q = &split_runs[m];
            *uses.entry(point(q, q.lo)).or_default() += 1;
            *uses.entry(point(q, q.hi)).or_default() += 1;
            length += (grid[q.along][q.hi] - grid[q.along][q.lo]).abs();
        }
        let ends: Vec<[usize; 3]> = uses
            .iter()
            .filter(|(_, n)| **n == 1)
            .map(|(p, _)| *p)
            .collect();
        let (wa, wb) = match ends.as_slice() {
            [a, b] => (world(*a), world(*b)),
            _ => {
                let p = *uses.keys().next().expect("chain has points");
                (world(p), world(p))
            }
        };
        edges.push((
            EdgeTopo {
                faces: [r.faces.0, r.faces.1],
                geom: EdgeGeom {
                    start: wa,
                    end: wb,
                    length,
                },
                laminar: false,
            },
            EdgeMeta {
                along: r.along,
                at: wa,
                lo: 0.0,
                hi: length,
                planes: None,
            },
            ends,
        ));
    }

    // Decorations: replace matching edges with blend faces.
    let mut decorations = std::mem::take(&mut s.decorations);
    let mut final_edges: Vec<EdgeRow> = Vec::new();
    let mut new_faces: Vec<(FFace, FaceGeom)> = Vec::new();
    for dec in &mut decorations {
        dec.pieces.clear();
    }
    for (e, meta, ends) in edges {
        let mut hit = None;
        if let Some(planes) = meta.planes {
            for (di, dec) in decorations.iter().enumerate() {
                let same_planes = (planes[0].same(&dec.planes[0])
                    && planes[1].same(&dec.planes[1]))
                    || (planes[0].same(&dec.planes[1]) && planes[1].same(&dec.planes[0]));
                let same_line = meta.along == dec.along
                    && (0..3)
                        .filter(|a| *a != dec.along)
                        .all(|a| (meta.at[a] - dec.at[a]).abs() < EPS);
                let overlap = meta.lo < dec.hi - EPS && meta.hi > dec.lo + EPS;
                if same_planes && same_line && overlap {
                    hit = Some(di);
                    break;
                }
            }
        }
        match hit {
            None => final_edges.push((e, meta, ends)),
            Some(di) => {
                let dec = &mut decorations[di];
                let fidx = (s.faces.len() + new_faces.len()) as u32;
                dec.pieces.push((meta.lo, meta.hi, fidx));
                let mid = [
                    (e.geom.start[0] + e.geom.end[0]) / 2.0,
                    (e.geom.start[1] + e.geom.end[1]) / 2.0,
                    (e.geom.start[2] + e.geom.end[2]) / 2.0,
                ];
                let mut dir = [0.0; 3];
                dir[dec.along] = 1.0;
                let mut lo = e.geom.start;
                let mut hi = e.geom.end;
                for a in 0..3 {
                    let (x, y) = (lo[a].min(hi[a]), lo[a].max(hi[a]));
                    lo[a] = x;
                    hi[a] = y;
                }
                let area = e.geom.length
                    * dec.size
                    * if dec.role == Role::Blend {
                        std::f64::consts::FRAC_PI_2
                    } else {
                        std::f64::consts::SQRT_2
                    };
                let surface = if dec.role == Role::Blend && !dec.curved {
                    SurfaceKind::Cylinder {
                        axis_point: e.geom.start,
                        axis_direction: dir,
                        radius: dec.size,
                    }
                } else {
                    SurfaceKind::Other
                };
                new_faces.push((
                    FFace::blend(),
                    FaceGeom {
                        surface,
                        centroid: mid,
                        area,
                        bbox_min: lo,
                        bbox_max: hi,
                    },
                ));
                let meta2 = EdgeMeta {
                    planes: None,
                    ..meta
                };
                final_edges.push((
                    EdgeTopo {
                        faces: [e.faces[0], fidx],
                        geom: e.geom,
                        laminar: false,
                    },
                    meta2,
                    ends.clone(),
                ));
                final_edges.push((
                    EdgeTopo {
                        faces: [fidx, e.faces[1]],
                        geom: e.geom,
                        laminar: false,
                    },
                    meta2,
                    ends,
                ));
            }
        }
    }
    // Each blend piece meets the face across each of its ends along an arc (a chamfer: a
    // line) between its two tangent points, the edge a B-rep kernel builds there.
    let mut arcs: Vec<EdgeRow> = Vec::new();
    for dec in &decorations {
        for &(_, _, fidx) in &dec.pieces {
            let Some((sharp, ends)) = final_edges
                .iter()
                .find(|(e, _, _)| e.faces[1] == fidx)
                .map(|(e, _, ends)| (e, ends.clone()))
            else {
                continue;
            };
            let (f0, f1) = (sharp.faces[0], {
                let other = final_edges
                    .iter()
                    .find(|(e, _, _)| e.faces[0] == fidx)
                    .map(|(e, _, _)| e.faces[1]);
                match other {
                    Some(f) => f,
                    None => continue,
                }
            });
            for end in ends {
                // The face across the end: it shares an edge at this corner with both faces.
                let at_end = |f: u32| -> BTreeSet<u32> {
                    final_edges
                        .iter()
                        .filter(|(e, _, ends)| ends.contains(&end) && e.faces.contains(&f))
                        .flat_map(|(e, _, _)| e.faces)
                        .filter(|x| *x != f && *x != fidx)
                        .collect()
                };
                let across: Vec<u32> = at_end(f0)
                    .intersection(&at_end(f1))
                    .copied()
                    .filter(|x| *x != f0 && *x != f1)
                    .collect();
                let [x] = across.as_slice() else {
                    continue;
                };
                let corner = world(end);
                let [(a0, s0), (a1, s1)] = dec.planes.map(|p| (p.axis, p.dir as f64));
                let mut t0 = corner;
                t0[a1] -= s1 * dec.size;
                let mut t1 = corner;
                t1[a0] -= s0 * dec.size;
                let (start, finish) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
                let length = dec.size
                    * if dec.role == Role::Chamfer {
                        std::f64::consts::SQRT_2
                    } else {
                        std::f64::consts::FRAC_PI_2
                    };
                arcs.push((
                    EdgeTopo {
                        faces: [*x, fidx],
                        geom: EdgeGeom {
                            start,
                            end: finish,
                            length,
                        },
                        laminar: false,
                    },
                    EdgeMeta {
                        along: dec.along,
                        at: corner,
                        lo: corner[dec.along],
                        hi: corner[dec.along],
                        planes: None,
                    },
                    vec![end],
                ));
            }
        }
    }
    final_edges.extend(arcs);
    for (f, g) in new_faces {
        s.faces.push(f);
        topo.faces.push(FaceTopo { geom: g });
    }
    s.decorations = decorations
        .into_iter()
        .filter(|d| !d.pieces.is_empty())
        .collect();

    let mut vmap: BTreeMap<[usize; 3], Vec<u32>> = BTreeMap::new();
    for (ei, (_, _, ends)) in final_edges.iter().enumerate() {
        for p in ends {
            vmap.entry(*p).or_default().push(ei as u32);
        }
    }
    s.edge_meta = final_edges.iter().map(|(_, m, _)| *m).collect();
    topo.edges = final_edges.into_iter().map(|(e, _, _)| e).collect();
    topo.vertices = vmap
        .into_iter()
        .map(|(p, edges)| VertexTopo {
            edges,
            point: world(p),
        })
        .collect();
    s.topology = topo;
}

fn face_geom(f: &FFace) -> FaceGeom {
    let mut area = 0.0;
    let mut c = [0.0; 3];
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for r in &f.rects {
        let mut a = 1.0;
        for ax in 0..3 {
            let w = r.hi[ax] - r.lo[ax];
            if w > EPS {
                a *= w;
            }
            lo[ax] = lo[ax].min(r.lo[ax]);
            hi[ax] = hi[ax].max(r.hi[ax]);
        }
        area += a;
        for ax in 0..3 {
            c[ax] += a * (r.lo[ax] + r.hi[ax]) / 2.0;
        }
    }
    if area > 0.0 {
        for x in &mut c {
            *x /= area;
        }
    }
    FaceGeom {
        surface: f.hint(),
        centroid: c,
        area,
        bbox_min: lo,
        bbox_max: hi,
    }
}

// ------------------------------------------------------------------------------- the kernel

impl FakeKernel {
    pub fn new() -> Self {
        Self::default()
    }

    fn id_static() -> KernelId {
        KernelId {
            name: "fake-grid".into(),
            version: "1".into(),
        }
    }

    fn solid<'a>(operands: &[&'a BodyHandle], i: usize) -> Result<&'a FakeSolid, KernelError> {
        operands[i]
            .downcast::<FakeSolid>()
            .ok_or(KernelError::ForeignOperand { operand: i as u32 })
    }

    fn bounds(solids: &[&FakeSolid]) -> (f64, f64, [f64; 3], [f64; 3]) {
        let mut lo = [-1.0f64; 3];
        let mut hi = [1.0f64; 3];
        let mut first = true;
        for s in solids {
            if let Some((l, h)) = s.bounds() {
                for a in 0..3 {
                    if first {
                        lo[a] = l[a];
                        hi[a] = h[a];
                    } else {
                        lo[a] = lo[a].min(l[a]);
                        hi[a] = hi[a].max(h[a]);
                    }
                }
                first = false;
            }
        }
        let m = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max).max(1e-3);
        (
            lo.iter().cloned().fold(f64::INFINITY, f64::min) - m,
            hi.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + m,
            lo,
            hi,
        )
    }

    /// Finish a grid result: carry operand decorations, add fresh ones, build topology, and
    /// write history.
    fn finish(
        &self,
        operands: &[&FakeSolid],
        operand_ids: &[u32],
        mut res: GridResult,
        fresh: Vec<Decoration>,
        extra: Vec<(FakeSolid, Vec<Vec<HistorySource>>, Vec<HistorySource>)>,
    ) -> Result<KernelOutput, KernelError> {
        // Carry decorations: every operand decoration goes to each output (only bodies with a
        // matching edge keep it).
        let mut carried: Vec<(Decoration, u32)> = Vec::new();
        for (k, o) in operands.iter().enumerate() {
            for dec in &o.decorations {
                carried.push((dec.clone(), operand_ids[k]));
            }
        }
        for (s, srcs, b) in extra {
            res.solids.push(s);
            res.face_sources.push(srcs);
            res.body_sources.push(b);
        }
        let mut bodies = Vec::new();
        let mut history = OpHistory::default();
        for (bi, solid) in res.solids.iter_mut().enumerate() {
            let grid_faces = solid.faces.len();
            {
                let mut decs: Vec<Decoration> = carried.iter().map(|(d, _)| d.clone()).collect();
                decs.extend(fresh.iter().cloned());
                let keep_old = solid.decorations.clone();
                solid.faces.truncate(grid_faces);
                solid.decorations = if decs.is_empty() { keep_old } else { decs };
                build_topology(solid);
            }
            for f in 0..solid.faces.len() {
                let sources = if f < grid_faces {
                    res.face_sources[bi][f].clone()
                } else {
                    // A blend face: find its decoration and source.
                    let dec = solid
                        .decorations
                        .iter()
                        .find(|d| d.pieces.iter().any(|p| p.2 == f as u32))
                        .expect("blend face belongs to a decoration");
                    let piece = dec.pieces.iter().find(|p| p.2 == f as u32).expect("piece");
                    match &dec.fresh {
                        Some(label) => vec![label.clone()],
                        None => {
                            // Previous faces of this decoration in its operand, by overlap.
                            let mut v = Vec::new();
                            for (k, o) in operands.iter().enumerate() {
                                for od in &o.decorations {
                                    if od.along == dec.along
                                        && od.planes[0].same(&dec.planes[0])
                                        && od.planes[1].same(&dec.planes[1])
                                        && (od.lo - dec.lo).abs() < EPS
                                    {
                                        for (lo, hi, fid) in &od.pieces {
                                            if piece.0 < hi - EPS && piece.1 > lo + EPS {
                                                v.push(HistorySource::Input {
                                                    from: OperandEntity {
                                                        operand: operand_ids[k],
                                                        entity: *fid,
                                                    },
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                            if v.is_empty() {
                                return Err(KernelError::HistoryIncomplete {
                                    message: "fake kernel: blend face lost its source".into(),
                                });
                            }
                            v.sort();
                            v.dedup();
                            v
                        }
                    }
                };
                history.faces.push(FaceRecord {
                    body: bi as u32,
                    face: f as u32,
                    evolution: Evolution::Modified,
                    sources,
                });
            }
            for d in &mut solid.decorations {
                d.fresh = None;
            }
            history.bodies.push(BodyRecord {
                body: bi as u32,
                sources: res.body_sources[bi].clone(),
            });
        }
        // Evolution and deletions.
        let mut uses: BTreeMap<OperandEntity, u32> = BTreeMap::new();
        for r in &history.faces {
            for s in &r.sources {
                if let HistorySource::Input { from } | HistorySource::Copy { from, .. } = s {
                    *uses.entry(*from).or_default() += 1;
                }
            }
        }
        for r in &mut history.faces {
            let inputs: Vec<&OperandEntity> = r
                .sources
                .iter()
                .filter_map(|s| match s {
                    HistorySource::Input { from } => Some(from),
                    _ => None,
                })
                .collect();
            r.evolution = if inputs.is_empty() {
                Evolution::Generated
            } else if inputs.len() > 1 || r.sources.len() > inputs.len() {
                Evolution::Merged
            } else if uses.get(inputs[0]).copied().unwrap_or(0) > 1 {
                Evolution::Split
            } else {
                Evolution::Modified
            };
        }
        for (k, o) in operands.iter().enumerate() {
            for f in 0..o.faces.len() {
                let oe = OperandEntity {
                    operand: operand_ids[k],
                    entity: f as u32,
                };
                if !uses.contains_key(&oe) {
                    history.deleted_faces.push(oe);
                }
            }
            let used = history.bodies.iter().any(|b| {
                b.sources.iter().any(|s| matches!(s, HistorySource::Input { from } | HistorySource::Copy { from, .. } if from.operand == operand_ids[k]))
            });
            if !used {
                history.deleted_bodies.push(operand_ids[k]);
            }
        }
        history.deleted_faces.sort();
        history.deleted_bodies.sort();
        for s in res.solids {
            let topology = s.topology.clone();
            bodies.push(KernelBody {
                handle: BodyHandle::new(Self::id_static(), s),
                topology,
            });
        }
        Ok(KernelOutput { bodies, history })
    }

    fn tools(
        &self,
        op: &ToolOp,
        solids: &[&FakeSolid],
        profile_index: u32,
    ) -> Result<Vec<Tool>, KernelError> {
        let (blo, bhi, lo, hi) = Self::bounds(solids);
        let corners: Vec<[f64; 3]> = (0..8)
            .map(|i| {
                [
                    if i & 1 == 0 { lo[0] } else { hi[0] },
                    if i & 2 == 0 { lo[1] } else { hi[1] },
                    if i & 4 == 0 { lo[2] } else { hi[2] },
                ]
            })
            .collect();
        // How far along `dir` from `origin` the operands reach, either way, with margin.
        let reach = |origin: [f64; 3], dir: [f64; 3]| -> (f64, f64) {
            let ds: Vec<f64> = corners
                .iter()
                .map(|c| (0..3).map(|a| (c[a] - origin[a]) * dir[a]).sum::<f64>())
                .collect();
            let m = (bhi - blo).abs().max(1.0) * 0.05;
            (
                ds.iter().cloned().fold(f64::INFINITY, f64::min) - m,
                ds.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + m,
            )
        };
        let side = |curve: &crate::ids::SketchEntityId| {
            gen(
                Role::Side,
                KernelTag::Curve {
                    profile: profile_index,
                    curve: curve.clone(),
                },
                None,
            )
        };
        let curves = |p: &CanonicalProfile| -> Vec<Vec<(sampled::Curve2, HistorySource)>> {
            p.loops
                .iter()
                .map(|l| {
                    l.iter()
                        .map(|seg| (sampled::Curve2::from_segment(seg), side(&seg.curve)))
                        .collect()
                })
                .collect()
        };
        let caps = || {
            [
                gen(Role::StartCap, KernelTag::None, None),
                gen(Role::EndCap, KernelTag::None, None),
            ]
        };
        match &op.shape {
            ToolShape::Prism {
                profile,
                start,
                end,
                start_bound,
                end_bound,
                draft,
                thin,
            } => {
                let through_all = matches!(end_bound, PrismBound::ThroughAll);
                let plain = start_bound.is_blind()
                    && (end_bound.is_blind() || through_all)
                    && draft.is_none()
                    && thin.is_none();
                if !plain {
                    return self.extrusion(op, solids, profile_index, &reach);
                }
                let exact = prism_from_profile(
                    profile,
                    *start,
                    *end,
                    through_all.then_some((blo, bhi)),
                    profile_index,
                );
                match exact {
                    Ok(p) => Ok(vec![Tool::Exact(p)]),
                    Err(KernelError::NotSupported { .. }) => {
                        let mut e = *end;
                        if through_all {
                            let (a, b) = reach(profile.frame.origin, profile.frame.z);
                            e = if end - start >= 0.0 { b } else { a };
                        }
                        Ok(vec![Tool::Sampled(Box::new(sampled::prism(
                            Transform::from_frame(&profile.frame),
                            &curves(profile),
                            *start,
                            e,
                            caps(),
                        )))])
                    }
                    Err(other) => Err(other),
                }
            }
            ToolShape::Revolve {
                profile,
                axis_point,
                axis_direction,
                angle,
            } => {
                let to_world = Transform::from_frame(&profile.frame);
                let inv = to_world.inverse();
                let ap = inv.apply(*axis_point);
                let ad = inv.apply_vector(*axis_direction);
                if ap[2].abs() > 1e-7 || ad[2].abs() > 1e-7 {
                    return Err(KernelError::Failed {
                        message: "the revolve axis must lie in the profile plane".into(),
                    });
                }
                Ok(vec![Tool::Sampled(Box::new(sampled::revolve(
                    to_world,
                    &curves(profile),
                    [ap[0], ap[1]],
                    [ad[0], ad[1]],
                    *angle,
                    Some(caps()),
                )?))])
            }
            ToolShape::Sweep {
                profile,
                path,
                path_curves,
            } => {
                let to_world = Transform::from_frame(&profile.frame);
                let inv = to_world.inverse();
                let local: Vec<[f64; 3]> = path.iter().map(|p| inv.apply(*p)).collect();
                Ok(vec![Tool::Sampled(Box::new(sampled::sweep(
                    to_world,
                    &curves(profile),
                    &local,
                    path_curves,
                    caps(),
                )?))])
            }
            ToolShape::Loft { profiles } => {
                let first = profiles.first().ok_or_else(|| KernelError::Failed {
                    message: "a loft needs profiles".into(),
                })?;
                let to_world = Transform::from_frame(&first.frame);
                let inv = to_world.inverse();
                let tilted = |p: &CanonicalProfile| {
                    let zl = inv.apply_vector(p.frame.z);
                    zl[0].abs() > 1e-7 || zl[1].abs() > 1e-7
                };
                if profiles.iter().any(tilted) {
                    // Sections in any orientation: a closed polyhedron through them.
                    return Ok(vec![Tool::Polys(Box::new(sheet::loft_tool(
                        profiles,
                        &side,
                        caps(),
                    )?))]);
                }
                let mut sections = Vec::new();
                for p in profiles {
                    let z = inv.apply(p.frame.origin)[2];
                    let map = |q: [f64; 2]| {
                        let l = inv.apply(p.frame.to_world(q));
                        [l[0], l[1]]
                    };
                    let loops: Vec<Vec<sampled::Curve2>> = p
                        .loops
                        .iter()
                        .map(|l| {
                            l.iter()
                                .map(|seg| {
                                    let mut m = seg.clone();
                                    m.start = map(seg.start);
                                    m.end = map(seg.end);
                                    m.kind = seg.kind.mapped(&map);
                                    sampled::Curve2::from_segment(&m)
                                })
                                .collect()
                        })
                        .collect();
                    sections.push((z, loops));
                }
                // Sides are tagged by the first section's curves.
                let labels: Vec<Vec<HistorySource>> = first
                    .loops
                    .iter()
                    .map(|l| l.iter().map(|seg| side(&seg.curve)).collect())
                    .collect();
                Ok(vec![Tool::Sampled(Box::new(sampled::loft(
                    to_world,
                    sections,
                    &labels,
                    caps(),
                )?))])
            }
            ToolShape::Holes { .. } => self.holes(&op.shape, solids, &reach),
        }
    }

    fn mode(c: Combine) -> Mode {
        match c {
            Combine::New => Mode::New,
            Combine::Add => Mode::Add,
            Combine::Cut => Mode::Cut,
            Combine::Intersect => Mode::Intersect,
        }
    }
}

impl CadKernel for FakeKernel {
    fn id(&self) -> KernelId {
        Self::id_static()
    }

    fn execute(
        &self,
        op: &KernelOp,
        operands: &[&BodyHandle],
    ) -> Result<KernelOutput, KernelError> {
        if let Some(out) = self.execute_extra(op, operands) {
            return out;
        }
        let solids: Vec<&FakeSolid> = (0..operands.len())
            .map(|i| Self::solid(operands, i))
            .collect::<Result<_, _>>()?;
        let ids: Vec<u32> = (0..operands.len() as u32).collect();
        match op {
            KernelOp::Tool(t) => self.tool_op(t, &solids),
            KernelOp::PatternTools {
                seeds,
                instances,
                varied,
            } => {
                let combine =
                    seeds
                        .first()
                        .map(|s| s.combine)
                        .ok_or_else(|| KernelError::Failed {
                            message: "pattern with no seeds".into(),
                        })?;
                if seeds.iter().any(|s| s.combine != combine) || combine == Combine::New {
                    return Err(KernelError::NotSupported {
                        what: "fake kernel: patterns of new-body or mixed tools".into(),
                    });
                }
                if !varied.is_empty() && varied.len() != instances.len() {
                    return Err(KernelError::Failed {
                        message: "a variable pattern needs tools for every instance".into(),
                    });
                }
                let mut tools = Vec::new();
                for (ii, inst) in instances.iter().enumerate() {
                    let these = if varied.is_empty() {
                        seeds
                    } else {
                        &varied[ii]
                    };
                    for (si, seed) in these.iter().enumerate() {
                        for p in &self.tools(seed, &solids, si as u32)? {
                            let mut q = p.transformed(&inst.transform);
                            q.relabel(si as u32, &inst.key);
                            tools.push(q);
                        }
                    }
                }
                let res = run_grid(GridJob {
                    operands: solids.clone(),
                    operand_ids: ids.clone(),
                    tools,
                    mode: Self::mode(combine),
                    split: None,
                    separate_tools: false,
                    body_label: None,
                    modifier: None,
                    merge_collinear: false,
                    carve: Vec::new(),
                })?;
                self.finish(&solids, &ids, res, Vec::new(), Vec::new())
            }
            KernelOp::FullRound {
                first,
                center,
                second,
            } => self.full_round(first, center, second, &solids),
            KernelOp::VertexChamfer {
                vertices,
                distances,
            } => self.vertex_chamfer(vertices, *distances, &solids),
            KernelOp::Fillet { .. } | KernelOp::Chamfer { .. } | KernelOp::FaceBlend { .. } => {
                let (role, radius, curved, g2, edges) = match op {
                    KernelOp::Fillet {
                        edges,
                        radius,
                        variable,
                        profile,
                        setbacks,
                    } => {
                        if !setbacks.is_empty() {
                            return Err(KernelError::NotSupported {
                                what: "fake kernel: setback corner blends".into(),
                            });
                        }
                        let r = variable.iter().map(|v| v.1).fold(*radius, f64::max);
                        (
                            Role::Blend,
                            r,
                            !variable.is_empty() || !profile.is_circular(),
                            matches!(profile, BlendProfile::Curvature),
                            edges.clone(),
                        )
                    }
                    KernelOp::FaceBlend {
                        first,
                        second,
                        radius,
                        profile,
                    } => (
                        Role::Blend,
                        *radius,
                        !profile.is_circular(),
                        matches!(profile, BlendProfile::Curvature),
                        self.face_blend(first, second, *radius, &solids)?,
                    ),
                    KernelOp::Chamfer { edges, size } => {
                        (Role::Chamfer, size.reach(), false, false, edges.clone())
                    }
                    _ => unreachable!("blends only"),
                };
                let edges = &edges;
                let radius = &radius;
                let mut fresh_all: Vec<Vec<Decoration>> = vec![Vec::new(); solids.len()];
                for e in edges {
                    let s = solids[e.operand as usize];
                    let meta = s
                        .edge_meta
                        .get(e.entity as usize)
                        .ok_or(KernelError::Failed {
                            message: "edge out of range".into(),
                        })?;
                    let planes = meta.planes.ok_or_else(|| KernelError::NotSupported {
                        what: "fake kernel: blending an edge of a blend or a curved edge".into(),
                    })?;
                    fresh_all[e.operand as usize].push(Decoration {
                        role,
                        size: *radius,
                        along: meta.along,
                        at: meta.at,
                        lo: meta.lo,
                        hi: meta.hi,
                        planes,
                        pieces: Vec::new(),
                        fresh: Some(gen(role, KernelTag::InputEdge { edge: *e }, None)),
                        curved,
                        g2,
                    });
                }
                // Each operand is rebuilt alone with its old and new decorations.
                let mut bodies = Vec::new();
                let mut history = OpHistory::default();
                for (k, s) in solids.iter().enumerate() {
                    let job = GridJob {
                        operands: vec![s],
                        operand_ids: vec![k as u32],
                        tools: Vec::new(),
                        mode: Mode::Keep,
                        split: None,
                        separate_tools: false,
                        body_label: None,
                        modifier: None,
                        merge_collinear: false,
                        carve: Vec::new(),
                    };
                    let res = run_grid(job)?;
                    let mut out = self.finish(
                        &[s],
                        &[k as u32],
                        res,
                        std::mem::take(&mut fresh_all[k]),
                        Vec::new(),
                    )?;
                    let base = bodies.len() as u32;
                    for r in &mut out.history.faces {
                        r.body += base;
                    }
                    for r in &mut out.history.bodies {
                        r.body += base;
                    }
                    history.faces.extend(out.history.faces);
                    history.bodies.extend(out.history.bodies);
                    history.deleted_faces.extend(out.history.deleted_faces);
                    history.deleted_bodies.extend(out.history.deleted_bodies);
                    bodies.extend(out.bodies);
                }
                Ok(KernelOutput { bodies, history })
            }
            KernelOp::Shell {
                open_faces,
                thickness,
                faces,
                outward,
            } => {
                let per_face: BTreeMap<u32, f64> =
                    faces.iter().map(|(e, t)| (e.entity, *t)).collect();
                if solids.len() != 1 {
                    return Err(KernelError::Failed {
                        message: "a shell works on one body".into(),
                    });
                }
                if *thickness <= 0.0 {
                    return Err(KernelError::Failed {
                        message: "shell thickness must be positive".into(),
                    });
                }
                if !solids[0].decorations.is_empty() {
                    return Err(KernelError::NotSupported {
                        what: "fake kernel: shelling a blended body".into(),
                    });
                }
                let open: BTreeSet<u32> = open_faces.iter().map(|e| e.entity).collect();
                let res = run_grid(GridJob {
                    operands: solids.clone(),
                    operand_ids: ids.clone(),
                    tools: Vec::new(),
                    mode: Mode::Keep,
                    split: None,
                    separate_tools: false,
                    body_label: None,
                    modifier: Some(Modifier::Shell(ShellSpec {
                        open,
                        thickness: *thickness,
                        per_face,
                        outward: *outward,
                    })),
                    merge_collinear: false,
                    carve: Vec::new(),
                })?;
                self.finish(&solids, &ids, res, Vec::new(), Vec::new())
            }
            KernelOp::Draft {
                faces,
                neutral,
                angle,
                parting,
                step: _,
            } => {
                let pull = neutral.z;
                let tan = angle.tan();
                let mut bodies = Vec::new();
                let mut history = OpHistory::default();
                for (k, so) in solids.iter().enumerate() {
                    let mut specs = Vec::new();
                    for e in faces.iter().filter(|e| e.operand as usize == k) {
                        let face = so.faces.get(e.entity as usize).ok_or(KernelError::Failed {
                            message: "draft face out of range".into(),
                        })?;
                        let plane = face.plane.ok_or_else(|| KernelError::NotSupported {
                            what: "fake kernel: drafting a curved face".into(),
                        })?;
                        if pull[plane.axis].abs() > 1e-9 {
                            return Err(KernelError::Failed {
                                message: "a drafted face must be parallel to the pull direction"
                                    .into(),
                            });
                        }
                        let mut lo = [f64::INFINITY; 3];
                        let mut hi = [f64::NEG_INFINITY; 3];
                        for r in &face.rects {
                            for a in 0..3 {
                                lo[a] = lo[a].min(r.lo[a]);
                                hi[a] = hi[a].max(r.hi[a]);
                            }
                        }
                        // A parting-line draft hinges each face about the parting edges it
                        // touches: its hinge height is theirs.
                        let mut hinge_at = neutral.origin;
                        let hinge: Vec<f64> = parting
                            .iter()
                            .filter(|p| p.operand as usize == k)
                            .filter_map(|p| so.topology.edges.get(p.entity as usize))
                            .filter(|ed| ed.faces.contains(&e.entity))
                            .flat_map(|ed| [ed.geom.start, ed.geom.end])
                            .map(|q| (0..3).map(|a| (q[a] - neutral.origin[a]) * pull[a]).sum())
                            .collect();
                        if !parting.is_empty() {
                            if hinge.is_empty() {
                                return Err(KernelError::Failed {
                                    message: "a drafted face touches no parting-line edge".into(),
                                });
                            }
                            let h = hinge.iter().sum::<f64>() / hinge.len() as f64;
                            for a in 0..3 {
                                hinge_at[a] += h * pull[a];
                            }
                        }
                        let mut max_h: f64 = 0.0;
                        for i in 0..8 {
                            let c = [
                                if i & 1 == 0 { lo[0] } else { hi[0] },
                                if i & 2 == 0 { lo[1] } else { hi[1] },
                                if i & 4 == 0 { lo[2] } else { hi[2] },
                            ];
                            let h: f64 = (0..3).map(|a| (c[a] - hinge_at[a]) * pull[a]).sum();
                            max_h = max_h.max(h.abs());
                        }
                        specs.push(DraftSpec {
                            face: e.entity,
                            plane,
                            lo,
                            hi,
                            origin: hinge_at,
                            pull,
                            tan,
                            reach: tan.abs() * max_h + EPS,
                            shift: 0.0,
                        });
                    }
                    let job = GridJob {
                        operands: vec![so],
                        operand_ids: vec![k as u32],
                        tools: Vec::new(),
                        mode: Mode::Keep,
                        split: None,
                        separate_tools: false,
                        body_label: None,
                        modifier: (!specs.is_empty()).then_some(Modifier::Draft(specs)),
                        merge_collinear: false,
                        carve: Vec::new(),
                    };
                    let res = run_grid(job)?;
                    let mut out = self.finish(&[so], &[k as u32], res, Vec::new(), Vec::new())?;
                    let base = bodies.len() as u32;
                    for r in &mut out.history.faces {
                        r.body += base;
                    }
                    for r in &mut out.history.bodies {
                        r.body += base;
                    }
                    history.faces.extend(out.history.faces);
                    history.bodies.extend(out.history.bodies);
                    history.deleted_faces.extend(out.history.deleted_faces);
                    history.deleted_bodies.extend(out.history.deleted_bodies);
                    bodies.extend(out.bodies);
                }
                Ok(KernelOutput { bodies, history })
            }
            KernelOp::PatternBodies { instances } => {
                let keep = run_grid(GridJob {
                    operands: solids.clone(),
                    operand_ids: ids.clone(),
                    tools: Vec::new(),
                    mode: Mode::Keep,
                    split: None,
                    separate_tools: false,
                    body_label: None,
                    modifier: None,
                    merge_collinear: false,
                    carve: Vec::new(),
                })?;
                let mut extra = Vec::new();
                for inst in instances {
                    let t = inst.transform;
                    if t.rotation != Transform::translation([0.0; 3]).rotation {
                        // Any other motion: re-sample each body's image on the grid.
                        for (k, s) in solids.iter().enumerate() {
                            let img = sampled::image(s, k as u32, &t, Some(&inst.key))?;
                            let res = run_grid(GridJob {
                                operands: Vec::new(),
                                operand_ids: Vec::new(),
                                tools: vec![Tool::Sampled(Box::new(img))],
                                mode: Mode::New,
                                split: None,
                                separate_tools: false,
                                body_label: Some(vec![HistorySource::Copy {
                                    from: OperandEntity {
                                        operand: k as u32,
                                        entity: 0,
                                    },
                                    instance: inst.key.clone(),
                                }]),
                                modifier: None,
                                merge_collinear: false,
                                carve: Vec::new(),
                            })?;
                            for ((c, f), b) in res
                                .solids
                                .into_iter()
                                .zip(res.face_sources)
                                .zip(res.body_sources)
                            {
                                extra.push((c, f, b));
                            }
                        }
                        continue;
                    }
                    for (k, s) in solids.iter().enumerate() {
                        let mut c = (*s).clone();
                        shift(&mut c, t.translation);
                        let srcs = (0..s.faces.len())
                            .map(|f| {
                                vec![HistorySource::Copy {
                                    from: OperandEntity {
                                        operand: k as u32,
                                        entity: f as u32,
                                    },
                                    instance: inst.key.clone(),
                                }]
                            })
                            .collect::<Vec<_>>();
                        // Drop blend faces from the grid face list; they are rebuilt.
                        let grid_faces = c.faces.iter().take_while(|f| f.is_grid()).count();
                        c.faces.truncate(grid_faces);
                        let srcs = srcs.into_iter().take(grid_faces).collect();
                        extra.push((
                            c,
                            srcs,
                            vec![HistorySource::Copy {
                                from: OperandEntity {
                                    operand: k as u32,
                                    entity: 0,
                                },
                                instance: inst.key.clone(),
                            }],
                        ));
                    }
                }
                self.finish(&solids, &ids, keep, Vec::new(), extra)
            }
            KernelOp::Boolean {
                combine,
                keep_tools,
            } => {
                if solids.is_empty() {
                    return Err(KernelError::Failed {
                        message: "boolean without a target".into(),
                    });
                }
                let mode = match combine {
                    Combine::Add => Mode::BoolUnion,
                    Combine::Cut => Mode::BoolCut,
                    Combine::Intersect => Mode::BoolIntersect,
                    Combine::New => {
                        return Err(KernelError::Failed {
                            message: "boolean needs a combine".into(),
                        })
                    }
                };
                let res = run_grid(GridJob {
                    operands: solids.clone(),
                    operand_ids: ids.clone(),
                    tools: Vec::new(),
                    mode,
                    split: None,
                    separate_tools: false,
                    body_label: None,
                    modifier: None,
                    merge_collinear: false,
                    carve: Vec::new(),
                })?;
                let mut extra = Vec::new();
                if *keep_tools && mode != Mode::BoolUnion {
                    for k in 1..solids.len() {
                        let alone = run_grid(GridJob {
                            operands: vec![solids[k]],
                            operand_ids: vec![k as u32],
                            tools: Vec::new(),
                            mode: Mode::Keep,
                            split: None,
                            separate_tools: false,
                            body_label: None,
                            modifier: None,
                            merge_collinear: false,
                            carve: Vec::new(),
                        })?;
                        for ((s, f), b) in alone
                            .solids
                            .into_iter()
                            .zip(alone.face_sources)
                            .zip(alone.body_sources)
                        {
                            extra.push((s, f, b));
                        }
                    }
                }
                self.finish(&solids, &ids, res, Vec::new(), extra)
            }
            KernelOp::Split { frame } => {
                let (a, s) = axis_of(frame.z).ok_or_else(|| KernelError::NotSupported {
                    what: "fake kernel: split planes not normal to an axis".into(),
                })?;
                let c = frame.origin[a];
                // Each operand alone, so only bodies the plane crosses get side tags.
                let mut bodies = Vec::new();
                let mut history = OpHistory::default();
                for (k, so) in solids.iter().enumerate() {
                    let crosses = so
                        .bounds()
                        .map(|(lo, hi)| lo[a] < c - EPS && hi[a] > c + EPS)
                        .unwrap_or(false);
                    let job = GridJob {
                        operands: vec![so],
                        operand_ids: vec![k as u32],
                        tools: Vec::new(),
                        mode: if crosses { Mode::Split } else { Mode::Keep },
                        split: crosses.then_some((a, c, s)),
                        separate_tools: false,
                        body_label: None,
                        modifier: None,
                        merge_collinear: false,
                        carve: Vec::new(),
                    };
                    let res = run_grid(job)?;
                    let mut out = self.finish(&[so], &[k as u32], res, Vec::new(), Vec::new())?;
                    let base = bodies.len() as u32;
                    for r in &mut out.history.faces {
                        r.body += base;
                    }
                    for r in &mut out.history.bodies {
                        r.body += base;
                    }
                    history.faces.extend(out.history.faces);
                    history.bodies.extend(out.history.bodies);
                    history.deleted_faces.extend(out.history.deleted_faces);
                    history.deleted_bodies.extend(out.history.deleted_bodies);
                    bodies.extend(out.bodies);
                }
                Ok(KernelOutput { bodies, history })
            }
            KernelOp::Transform { transform } => {
                if transform.rotation != Transform::translation([0.0; 3]).rotation {
                    // Rotations and reflections: each body re-sampled at its new place.
                    let mut bodies = Vec::new();
                    let mut history = OpHistory::default();
                    for (k, so) in solids.iter().enumerate() {
                        let img = sampled::image(so, k as u32, transform, None)?;
                        let res = run_grid(GridJob {
                            operands: Vec::new(),
                            operand_ids: Vec::new(),
                            tools: vec![Tool::Sampled(Box::new(img))],
                            mode: Mode::New,
                            split: None,
                            separate_tools: false,
                            body_label: Some(vec![HistorySource::Input {
                                from: OperandEntity {
                                    operand: k as u32,
                                    entity: 0,
                                },
                            }]),
                            modifier: None,
                            merge_collinear: false,
                            carve: Vec::new(),
                        })?;
                        let mut out = self.finish(&[], &[], res, Vec::new(), Vec::new())?;
                        // Faces and the body continue the operand's.
                        out.history.deleted_faces.clear();
                        out.history.deleted_bodies.clear();
                        let used: BTreeSet<u32> = out
                            .history
                            .faces
                            .iter()
                            .flat_map(|r| r.sources.iter())
                            .filter_map(|s| match s {
                                HistorySource::Input { from } => Some(from.entity),
                                _ => None,
                            })
                            .collect();
                        for f in 0..so.faces.len() as u32 {
                            if !used.contains(&f) {
                                history.deleted_faces.push(OperandEntity {
                                    operand: k as u32,
                                    entity: f,
                                });
                            }
                        }
                        for r in &mut out.history.faces {
                            r.body += bodies.len() as u32;
                            r.evolution = Evolution::Modified;
                        }
                        for r in &mut out.history.bodies {
                            r.body += bodies.len() as u32;
                        }
                        history.faces.extend(out.history.faces);
                        history.bodies.extend(out.history.bodies);
                        bodies.extend(out.bodies);
                    }
                    return Ok(KernelOutput { bodies, history });
                }
                let moved: Vec<FakeSolid> = solids
                    .iter()
                    .map(|s| {
                        let mut c = (*s).clone();
                        shift(&mut c, transform.translation);
                        c
                    })
                    .collect();
                let refs: Vec<&FakeSolid> = moved.iter().collect();
                // Re-grid the moved solids so labels follow the moved faces.
                let res = run_grid(GridJob {
                    operands: refs.clone(),
                    operand_ids: ids.clone(),
                    tools: Vec::new(),
                    mode: Mode::Keep,
                    split: None,
                    separate_tools: false,
                    body_label: None,
                    modifier: None,
                    merge_collinear: false,
                    carve: Vec::new(),
                })?;
                self.finish(&refs, &ids, res, Vec::new(), Vec::new())
            }
            KernelOp::Import { source, .. } => {
                let boxes = self
                    .imports
                    .get(source)
                    .ok_or_else(|| KernelError::Failed {
                        message: format!("cannot read `{}`", source),
                    })?;
                let mut tools = Vec::new();
                let mut body_labels = Vec::new();
                for b in boxes {
                    let lab = |i: usize| {
                        gen(
                            Role::Side,
                            KernelTag::CadName {
                                name: b.face_names[i].clone(),
                            },
                            None,
                        )
                    };
                    // A box is a prism along z of a rectangle in x/y.
                    let pts = vec![
                        [b.min[0], b.min[1]],
                        [b.max[0], b.min[1]],
                        [b.max[0], b.max[1]],
                        [b.min[0], b.max[1]],
                    ];
                    let labels = vec![lab(2), lab(1), lab(3), lab(0)];
                    tools.push(Tool::Exact(Prism {
                        k: 2,
                        u: 0,
                        v: 1,
                        loops: vec![pts],
                        edge_labels: vec![labels],
                        lo: b.min[2],
                        hi: b.max[2],
                        lo_label: lab(4),
                        hi_label: lab(5),
                    }));
                    body_labels.push(gen(
                        Role::Body,
                        KernelTag::CadName {
                            name: b.name.clone(),
                        },
                        None,
                    ));
                }
                let res = run_grid(GridJob {
                    operands: Vec::new(),
                    operand_ids: Vec::new(),
                    tools,
                    mode: Mode::New,
                    split: None,
                    separate_tools: true,
                    body_label: Some(body_labels),
                    modifier: None,
                    merge_collinear: false,
                    carve: Vec::new(),
                })?;
                self.finish(&[], &[], res, Vec::new(), Vec::new())
            }
            // Handled by `execute_extra`.
            KernelOp::Labeled { .. }
            | KernelOp::MoveFaces { .. }
            | KernelOp::DeleteFaces { .. }
            | KernelOp::ReplaceFaces { .. }
            | KernelOp::Scale { .. }
            | KernelOp::Surface(_) => unreachable!("dispatched first"),
        }
    }
}

fn shift(s: &mut FakeSolid, t: [f64; 3]) {
    for a in 0..3 {
        for g in &mut s.grid[a] {
            *g += t[a];
        }
    }
    for f in &mut s.faces {
        if let Some(p) = &mut f.plane {
            p.offset += t[p.axis];
        }
        for p in &mut f.facet_planes {
            p.offset += t[p.axis];
        }
        if let Some(h) = &mut f.surface {
            *h = sampled::move_hint(h, &Transform::translation(t));
        }
        if let Some(e) = &mut f.exact {
            *e = Arc::new(e.transformed(&Transform::translation(t)));
        }
        for r in &mut f.rects {
            for a in 0..3 {
                r.lo[a] += t[a];
                r.hi[a] += t[a];
            }
        }
    }
    for d in &mut s.decorations {
        for a in 0..3 {
            d.at[a] += t[a];
        }
        d.lo += t[d.along];
        d.hi += t[d.along];
        for p in &mut d.planes {
            p.offset += t[p.axis];
        }
        for piece in &mut d.pieces {
            piece.0 += t[d.along];
            piece.1 += t[d.along];
        }
    }
}
