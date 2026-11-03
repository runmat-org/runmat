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
//! Supported: extrudes of axis-aligned rectilinear polygons on axis-aligned frames (new, add,
//! cut, intersect, blind, symmetric, through all), square "holes", patterns and mirrors of those
//! tools by translation and axis reflection, body patterns, booleans, splits by axis-aligned
//! planes, translation, fillets and chamfers on straight grid edges, and imports of named boxes.
//! Everything else returns `KernelError::NotSupported`, which tests use for continue-past-failure.

// Grid code indexes several per-axis arrays in step; index loops read clearer than zips here.
#![allow(clippy::needless_range_loop)]

use crate::model::datum::Frame;
use crate::refs::history::*;
use crate::refs::name::{InstanceKey, PlaneSide, Role};
use crate::regen::kernel::*;
use std::collections::{BTreeMap, BTreeSet};

const EPS: f64 = 1e-9;

/// An edge with its grid metadata and its two end points in grid indices.
/// A facet edge segment: the axis it runs along and its start point in grid indices.
type SegmentKey = (usize, [usize; 3]);
/// An edge run's identity: its face pair, axis and line (start index zeroed).
type PieceKey = ((u32, u32), usize, [usize; 3]);
type EdgeRow = (EdgeTopo, EdgeMeta, [[usize; 3]; 2]);

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
    /// `None` for blend faces.
    plane: Option<PlaneKey>,
    rects: Vec<Rect>,
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

    /// Apply a diagonal transform (translation and axis reflections only).
    fn transformed(&self, t: &Transform) -> Result<Prism, KernelError> {
        let r = &t.rotation;
        for i in 0..3 {
            for j in 0..3 {
                let want_zero = i != j;
                if (want_zero && r[i][j].abs() > EPS)
                    || (!want_zero && (r[i][j].abs() - 1.0).abs() > EPS)
                {
                    return Err(KernelError::NotSupported {
                        what: "fake kernel: rotations in patterns".into(),
                    });
                }
            }
        }
        let map = |a: usize, x: f64| r[a][a] * x + t.translation[a];
        let mut p = self.clone();
        for l in &mut p.loops {
            for q in l.iter_mut() {
                *q = [map(self.u, q[0]), map(self.v, q[1])];
            }
        }
        let (a, b) = (map(self.k, self.lo), map(self.k, self.hi));
        if a <= b {
            p.lo = a;
            p.hi = b;
        } else {
            p.lo = b;
            p.hi = a;
            std::mem::swap(&mut p.lo_label, &mut p.hi_label);
        }
        Ok(p)
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

fn hole_prisms(
    frame: &Frame,
    positions: &[(String, [f64; 2])],
    diameter: f64,
    depth: Option<f64>,
    bounds: (f64, f64),
) -> Result<Vec<Prism>, KernelError> {
    let (k, ks, u, v) = frame_axes(frame)?;
    let r = diameter / 2.0;
    let mut out = Vec::new();
    for (key, at) in positions {
        let corners = [[r, -r], [r, r], [-r, r], [-r, -r]];
        let pts: Vec<[f64; 2]> = corners
            .iter()
            .map(|c| {
                let w = frame.to_world([at[0] + c[0], at[1] + c[1]]);
                [w[u], w[v]]
            })
            .collect();
        // Wall ordinals follow the definition: +x side, +y side, −x side, −y side.
        let labels: Vec<HistorySource> = (0..4)
            .map(|i| gen(Role::HoleWall, KernelTag::Key { key: key.clone() }, Some(i)))
            .collect();
        let top = frame.origin[k];
        let bottom = match depth {
            Some(d) => top - ks * d,
            None => {
                if ks > 0.0 {
                    bounds.0
                } else {
                    bounds.1
                }
            }
        };
        let bottom_label = gen(Role::HoleBottom, KernelTag::Key { key: key.clone() }, None);
        let top_label = gen(
            Role::HoleBottom,
            KernelTag::Key { key: key.clone() },
            Some(1),
        );
        let (lo, hi, lo_label, hi_label) = if bottom < top {
            (bottom, top, bottom_label, top_label)
        } else {
            (top, bottom, top_label, bottom_label)
        };
        out.push(Prism {
            k,
            u,
            v,
            loops: vec![pts],
            edge_labels: vec![labels],
            lo,
            hi,
            lo_label,
            hi_label,
        });
    }
    Ok(out)
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
    tools: Vec<Prism>,
    mode: Mode,
    /// Split plane: axis, offset, normal sign.
    split: Option<(usize, f64, f64)>,
    /// For imports: one tool part per body.
    separate_tools: bool,
    body_label: Option<Vec<HistorySource>>,
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

fn run_grid(job: GridJob<'_>) -> Result<GridResult, KernelError> {
    let mut breaks: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for o in &job.operands {
        for a in 0..3 {
            breaks[a].extend(o.grid[a].iter().copied());
        }
    }
    for t in &job.tools {
        t.breaks(&mut breaks);
    }
    if let Some((a, c, _)) = job.split {
        breaks[a].push(c);
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
                for (k, o) in job.operands.iter().enumerate() {
                    if o.contains(p) {
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
    let mut labels: Vec<Vec<HistorySource>> = Vec::with_capacity(facets.len());
    for f in &facets {
        let plane = facet_plane(f);
        let r = facet_rect(f);
        let mut cen = [0.0; 3];
        for a in 0..3 {
            cen[a] = (r.lo[a] + r.hi[a]) / 2.0;
        }
        let mut ls: Vec<HistorySource> = Vec::new();
        for (k, o) in job.operands.iter().enumerate() {
            let bool_mode = matches!(
                job.mode,
                Mode::BoolUnion | Mode::BoolCut | Mode::BoolIntersect
            );
            if owners[f.cell] & (1 << k) == 0 && !matches!(job.mode, Mode::Add) && !bool_mode {
                continue;
            }
            for (fi, face) in o.faces.iter().enumerate() {
                if let Some(p) = &face.plane {
                    if p.same(&plane) && face.rects.iter().any(|r| r.contains(cen)) {
                        ls.push(HistorySource::Input {
                            from: OperandEntity {
                                operand: job.operand_ids[k],
                                entity: fi as u32,
                            },
                        });
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
                    if let Some(p) = &face.plane {
                        if p.same(&flipped) && face.rects.iter().any(|r| r.contains(cen)) {
                            ls.push(HistorySource::Input {
                                from: OperandEntity {
                                    operand: job.operand_ids[k],
                                    entity: fi as u32,
                                },
                            });
                        }
                    }
                }
            }
        }
        if ls.is_empty() {
            for t in &job.tools {
                if let Some(l) = t.label(plane.axis, plane.offset, cen) {
                    ls.push(l);
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
    }

    // Faces: coplanar facets that share a facet edge, per component.
    let mut fparent: Vec<usize> = (0..facets.len()).collect();
    for (fi, f) in facets.iter().enumerate() {
        for b in 0..3 {
            if b == f.axis || f.c[b] + 1 >= d[b] {
                continue;
            }
            let mut nc = f.c;
            nc[b] += 1;
            if let Some(&gj) = facet_at.get(&(idx(nc), f.axis, f.dir)) {
                if facets[gj].comp == f.comp && facet_plane(&facets[gj]).same(&facet_plane(f)) {
                    let (ra, rb) = (find(&mut fparent, fi), find(&mut fparent, gj));
                    if ra != rb {
                        fparent[ra.max(rb)] = ra.min(rb);
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
        let mut faces: Vec<FFace> = vec![
            FFace {
                plane: None,
                rects: Vec::new()
            };
            nf
        ];
        let mut fsrc: Vec<BTreeSet<HistorySource>> = vec![BTreeSet::new(); nf];
        for (fi, f) in facets.iter().enumerate() {
            if f.comp != comp {
                continue;
            }
            let l = facet_face[fi];
            faces[l].plane = Some(facet_plane(f));
            faces[l].rects.push(facet_rect(f));
            fsrc[l].extend(labels[fi].iter().cloned());
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
        let Some(p) = f.plane else { continue };
        for r in &f.rects {
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
    let mut edges: Vec<EdgeRow> = Vec::new();
    for r in &split_runs {
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
            [a, b],
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
                let surface = if dec.role == Role::Blend {
                    SurfaceKind::Cylinder {
                        axis_point: e.geom.start,
                        axis_direction: dir,
                        radius: dec.size,
                    }
                } else {
                    SurfaceKind::Other
                };
                new_faces.push((
                    FFace {
                        plane: None,
                        rects: Vec::new(),
                    },
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
                    },
                    meta2,
                    ends,
                ));
                final_edges.push((
                    EdgeTopo {
                        faces: [fidx, e.faces[1]],
                        geom: e.geom,
                    },
                    meta2,
                    ends,
                ));
            }
        }
    }
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
    let normal = match f.plane {
        Some(p) => {
            let mut n = [0.0; 3];
            n[p.axis] = p.dir as f64;
            n
        }
        None => [0.0; 3],
    };
    FaceGeom {
        surface: SurfaceKind::Plane { normal },
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

    fn tool_prisms(
        &self,
        op: &ToolOp,
        solids: &[&FakeSolid],
        profile_index: u32,
    ) -> Result<Vec<Prism>, KernelError> {
        let (blo, bhi, _, _) = Self::bounds(solids);
        match &op.shape {
            ToolShape::Prism {
                profile,
                start,
                end,
                through_all,
            } => Ok(vec![prism_from_profile(
                profile,
                *start,
                *end,
                through_all.then_some((blo, bhi)),
                profile_index,
            )?]),
            ToolShape::Holes {
                frame,
                positions,
                diameter,
                depth,
                counterbore,
                countersink,
            } => {
                if counterbore.is_some() || countersink.is_some() {
                    return Err(KernelError::NotSupported {
                        what: "fake kernel: counterbores and countersinks".into(),
                    });
                }
                hole_prisms(frame, positions, *diameter, *depth, (blo, bhi))
            }
            ToolShape::Revolve { .. } => Err(KernelError::NotSupported {
                what: "fake kernel: revolve".into(),
            }),
            ToolShape::Sweep { .. } => Err(KernelError::NotSupported {
                what: "fake kernel: sweep".into(),
            }),
            ToolShape::Loft { .. } => Err(KernelError::NotSupported {
                what: "fake kernel: loft".into(),
            }),
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
        let solids: Vec<&FakeSolid> = (0..operands.len())
            .map(|i| Self::solid(operands, i))
            .collect::<Result<_, _>>()?;
        let ids: Vec<u32> = (0..operands.len() as u32).collect();
        match op {
            KernelOp::Tool(t) => {
                let tools = self.tool_prisms(t, &solids, 0)?;
                let mode = Self::mode(t.combine);
                let ops = if mode == Mode::New {
                    Vec::new()
                } else {
                    solids.clone()
                };
                let res = run_grid(GridJob {
                    operands: ops.clone(),
                    operand_ids: if mode == Mode::New {
                        Vec::new()
                    } else {
                        ids.clone()
                    },
                    tools,
                    mode,
                    split: None,
                    separate_tools: false,
                    body_label: None,
                })?;
                if mode == Mode::New {
                    self.finish(&[], &[], res, Vec::new(), Vec::new())
                } else {
                    self.finish(&ops, &ids, res, Vec::new(), Vec::new())
                }
            }
            KernelOp::PatternTools { seeds, instances } => {
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
                let mut tools = Vec::new();
                for (si, seed) in seeds.iter().enumerate() {
                    let base = self.tool_prisms(seed, &solids, 0)?;
                    for inst in instances {
                        for p in &base {
                            let mut q = p.transformed(&inst.transform)?;
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
                })?;
                self.finish(&solids, &ids, res, Vec::new(), Vec::new())
            }
            KernelOp::Fillet { edges, radius }
            | KernelOp::Chamfer {
                edges,
                distance: radius,
                ..
            } => {
                let role = if matches!(op, KernelOp::Fillet { .. }) {
                    Role::Blend
                } else {
                    Role::Chamfer
                };
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
                        what: "fake kernel: blending an edge of a blend".into(),
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
            KernelOp::Shell { .. } => Err(KernelError::NotSupported {
                what: "fake kernel: shell".into(),
            }),
            KernelOp::Draft { .. } => Err(KernelError::NotSupported {
                what: "fake kernel: draft".into(),
            }),
            KernelOp::PatternBodies { instances } => {
                let keep = run_grid(GridJob {
                    operands: solids.clone(),
                    operand_ids: ids.clone(),
                    tools: Vec::new(),
                    mode: Mode::Keep,
                    split: None,
                    separate_tools: false,
                    body_label: None,
                })?;
                let mut extra = Vec::new();
                for inst in instances {
                    let t = inst.transform;
                    if t.rotation != Transform::translation([0.0; 3]).rotation {
                        return Err(KernelError::NotSupported {
                            what: "fake kernel: body patterns other than translation".into(),
                        });
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
                        let grid_faces = c.faces.iter().take_while(|f| f.plane.is_some()).count();
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
                    return Err(KernelError::NotSupported {
                        what: "fake kernel: rotations".into(),
                    });
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
                    tools.push(Prism {
                        k: 2,
                        u: 0,
                        v: 1,
                        loops: vec![pts],
                        edge_labels: vec![labels],
                        lo: b.min[2],
                        hi: b.max[2],
                        lo_label: lab(4),
                        hi_label: lab(5),
                    });
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
                })?;
                self.finish(&[], &[], res, Vec::new(), Vec::new())
            }
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
