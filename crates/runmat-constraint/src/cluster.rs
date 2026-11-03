//! Constructive rigid-cluster pre-pass for 2D sketches.
//!
//! Before a component is solved numerically, this module recognises rigid clusters of points
//! and places them by construction, bottom-up in the manner of Fudos and Hoffmann (1997), on
//! the triangle decomposition of Owen (1991):
//!
//! - a **cluster** is a set of points whose relative positions are known, stored in the
//!   cluster's own frame. Every point–point distance (a distance dimension, a line length, an
//!   arc or known circle radius) starts as a two-point **rigid** cluster. Fixed geometry is the
//!   **ground** cluster, in world coordinates. A distance plus a horizontal or vertical
//!   relation on the same pair is an **oriented** cluster, known up to a translation;
//! - two clusters sharing two points merge by the rigid motion that maps one pair onto the
//!   other; an oriented (or ground) cluster sharing one point with another oriented cluster
//!   merges by translation;
//! - a rigid cluster holding both points of a horizontal or vertical relation turns until the
//!   relation holds and becomes oriented;
//! - three clusters that pairwise share one point (Owen's virtual bonds) know the three sides
//!   of the triangle on the shared points: the triangle is constructed in one cluster's frame
//!   (circle ∩ circle) and the other two are moved onto it;
//! - an angle (or parallel, perpendicular) relation between a line of a rigid cluster and a
//!   line of an oriented or ground cluster turns the rigid cluster until it holds; it becomes
//!   oriented;
//! - **hinge**: two rigid clusters sharing exactly one point, with an angle relation between a
//!   line of each, merge by the rotation about the shared point that satisfies it;
//! - a point with two loci on ground geometry (a circle about a ground point, a horizontal or
//!   vertical through one, a ground line) is placed at their intersection.
//!
//! Branches follow Owen's rule: every constructed triangle keeps the orientation its three
//! points have in the starting geometry, and no cluster is ever reflected. A loci intersection
//! takes the solution nearest the point's starting position (circle ∩ circle keeps the
//! orientation). Inconsistent relations are never merged, so a contradiction is left for the
//! numerical solve to diagnose.
//!
//! Ground points end at their constructed world positions. A cluster that never reaches the
//! ground has no pose the constraints decide: it is placed where it best fits its points'
//! starting positions (rotation and translation, or translation only when oriented). The
//! output only seeds the numerical solve, which verifies every constraint and owns the rank
//! diagnosis. Work is bounded so a hostile sketch cannot stall the pre-pass.

use crate::ids::EntityId;
use crate::residual::Compiled;
use crate::residual::Op;
use crate::system::{Constraint, ConstraintKindRecord, EntityKind, System};
use std::collections::{BTreeMap, BTreeSet};

const NONE: usize = usize::MAX;
/// Membership scans allowed per construction (about 0.1 s): a hostile sketch stops early and
/// the numerical solve starts from whatever was placed.
const MAX_WORK: usize = 20_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    Rigid,
    Oriented,
    Ground,
}

#[derive(Clone, Debug)]
struct Cluster {
    kind: Kind,
    pts: BTreeMap<usize, [f64; 2]>,
}

/// An angle between two lines, each given by its end point groups.
#[derive(Clone, Copy, Debug)]
struct AngleRel {
    a: (usize, usize),
    b: (usize, usize),
    theta: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Axis {
    H,
    V,
}

/// What the pre-pass placed.
#[derive(Clone, Debug, Default)]
pub(crate) struct Construction {
    /// `(global var, value)` for every coordinate placed.
    pub values: Vec<(usize, f64)>,
    /// Clusters with two or more points at the end.
    pub clusters: usize,
    /// Point entities placed.
    pub points: usize,
}

struct Group {
    members: Vec<EntityId>,
    ground: Option<[f64; 2]>,
    cur: [f64; 2],
    bad: bool,
}

struct Builder<'a> {
    sys: &'a System,
    x: &'a [f64],
    groups: Vec<Group>,
    group_of: Vec<usize>,
    dists: Vec<(usize, usize, f64)>,
    hv: Vec<(usize, usize, Axis)>,
    /// Point group on the line through two point groups.
    online: Vec<(usize, usize, usize)>,
    /// Signed angle relations: from line `(a0, a1)`'s direction to line `(b0, b1)`'s, radians.
    angles: Vec<AngleRel>,
    clusters: Vec<Option<Cluster>>,
    /// Group → clusters holding it.
    index: Vec<BTreeSet<usize>>,
    /// Cluster pairs found inconsistent (never retried).
    refused: BTreeSet<(usize, usize)>,
    tol: f64,
    work: usize,
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}
fn add(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}
fn scale(a: [f64; 2], s: f64) -> [f64; 2] {
    [a[0] * s, a[1] * s]
}
fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn len(a: [f64; 2]) -> f64 {
    a[0].hypot(a[1])
}
fn wrap_angle(a: f64) -> f64 {
    let t = std::f64::consts::TAU;
    let r = a.rem_euclid(t);
    if r > std::f64::consts::PI {
        r - t
    } else {
        r
    }
}
fn rot(a: [f64; 2], c: f64, s: f64) -> [f64; 2] {
    [c * a[0] - s * a[1], s * a[0] + c * a[1]]
}

/// A proper rigid motion `p ↦ R p + t`.
#[derive(Clone, Copy, Debug)]
struct Motion {
    c: f64,
    s: f64,
    t: [f64; 2],
}

impl Motion {
    fn apply(&self, p: [f64; 2]) -> [f64; 2] {
        add(rot(p, self.c, self.s), self.t)
    }
    fn translation(t: [f64; 2]) -> Motion {
        Motion { c: 1.0, s: 0.0, t }
    }
    /// The motion mapping `p1 → q1` and the direction `p1p2` onto `q1q2` (lengths must agree).
    fn from_pairs(p1: [f64; 2], p2: [f64; 2], q1: [f64; 2], q2: [f64; 2]) -> Motion {
        let (s, d) = (sub(p2, p1), sub(q2, q1));
        let ang = cross(s, d).atan2(dot(s, d));
        let (sn, cs) = ang.sin_cos();
        let t = sub(q1, rot(p1, cs, sn));
        Motion { c: cs, s: sn, t }
    }
}

/// Runs the pre-pass on one component: `rows` are its problem rows' constraint indices
/// (ascending, deduplicated), `free` marks free variables, `in_comp` the component's free
/// variables.
pub(crate) fn construct(
    sys: &System,
    compiled: &[Compiled],
    ops: &[Option<Op>],
    constraints: &[usize],
    is_free: &[bool],
    in_comp: &[bool],
    x: &[f64],
) -> Construction {
    let mut b = Builder {
        sys,
        x,
        groups: Vec::new(),
        group_of: vec![NONE; sys.entities.len()],
        dists: Vec::new(),
        hv: Vec::new(),
        online: Vec::new(),
        angles: Vec::new(),
        clusters: Vec::new(),
        index: Vec::new(),
        refused: BTreeSet::new(),
        tol: 0.0,
        work: 0,
    };
    if !b.read(compiled, ops, constraints, is_free, in_comp) {
        return Construction::default();
    }
    b.seed();
    b.saturate();
    b.finish(is_free)
}

impl<'a> Builder<'a> {
    fn pt_vars(&self, e: EntityId) -> Option<(usize, usize)> {
        match self.sys.entity(e).kind {
            EntityKind::Point2 { x, y } => Some((x.index(), y.index())),
            _ => None,
        }
    }

    /// Reads points and relations. Returns false when there is nothing to construct.
    fn read(
        &mut self,
        compiled: &[Compiled],
        ops: &[Option<Op>],
        constraints: &[usize],
        is_free: &[bool],
        in_comp: &[bool],
    ) -> bool {
        let sys = self.sys;
        // Usable points: both coordinates free in this component, or both constant.
        let mut usable = vec![false; sys.entities.len()];
        let mut constant = vec![false; sys.entities.len()];
        let mut touched = BTreeSet::new();
        for &c in constraints {
            for v in &compiled[c].vars {
                if let Some(owner) = sys.var(*v).owner {
                    touched.insert(owner);
                }
            }
        }
        // Points owned through curves: the layout lists point coordinates, whose owner is the
        // point itself, so `touched` already holds every point a constraint reads.
        for &e in &touched {
            if let Some((vx, vy)) = self.pt_vars(e) {
                let (fx, fy) = (is_free[vx], is_free[vy]);
                if fx && fy && in_comp[vx] && in_comp[vy] {
                    usable[e.index()] = true;
                } else if !fx && !fy {
                    usable[e.index()] = true;
                    constant[e.index()] = true;
                }
            }
        }
        // Union-find over coincidences.
        let n = sys.entities.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        let union = |a: EntityId, b: EntityId, parent: &mut Vec<usize>| {
            if !usable[a.index()] || !usable[b.index()] {
                return;
            }
            let (ra, rb) = (find(parent, a.index()), find(parent, b.index()));
            if ra != rb {
                let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
                parent[hi] = lo;
            }
        };
        let user = |c: usize| match &sys.constraints[c].constraint {
            ConstraintKindRecord::User(u) if ops[c].is_some() => Some(u),
            _ => None,
        };
        let center_of = |e: EntityId| match &sys.entity(e).kind {
            EntityKind::Circle { center, .. }
            | EntityKind::Arc { center, .. }
            | EntityKind::Ellipse { center, .. }
            | EntityKind::EllipticalArc { center, .. } => Some(*center),
            _ => None,
        };
        for &c in constraints {
            match user(c) {
                Some(Constraint::Coincident(a, b)) => union(*a, *b, &mut parent),
                Some(Constraint::Concentric(a, b)) => {
                    if let (Some(ca), Some(cb)) = (center_of(*a), center_of(*b)) {
                        union(ca, cb, &mut parent)
                    }
                }
                _ => {}
            }
        }
        // Ground positions: constant points, and points held by Fix constraints.
        let mut fixed_at: BTreeMap<EntityId, [f64; 2]> = BTreeMap::new();
        for (e, _) in constant.iter().enumerate().filter(|(_, c)| **c) {
            let (vx, vy) = self.pt_vars(EntityId(e as u32)).unwrap();
            fixed_at.insert(EntityId(e as u32), [self.x[vx], self.x[vy]]);
        }
        let mut known_radius: BTreeMap<EntityId, f64> = BTreeMap::new();
        for e in 0..n {
            if let EntityKind::Circle { radius, .. } = sys.entities[e].kind {
                if !is_free[radius.index()] {
                    known_radius.insert(EntityId(e as u32), self.x[radius.index()]);
                }
            }
        }
        for &c in constraints {
            let Some(u) = user(c) else { continue };
            match u {
                Constraint::Fix(e) => {
                    let t = &sys.constraints[c].fix_targets;
                    let mut pin = |p: EntityId, k: usize| {
                        if usable[p.index()] && t.len() >= k + 2 {
                            fixed_at.entry(p).or_insert([t[k], t[k + 1]]);
                        }
                    };
                    match &sys.entity(*e).kind {
                        EntityKind::Point2 { .. } => pin(*e, 0),
                        EntityKind::Line { start, end } => {
                            pin(*start, 0);
                            pin(*end, 2);
                        }
                        EntityKind::Circle { center, .. } => {
                            pin(*center, 0);
                            known_radius.insert(*e, t[2]);
                        }
                        EntityKind::Arc { center, start, end } => {
                            pin(*center, 0);
                            pin(*start, 2);
                            pin(*end, 4);
                        }
                        EntityKind::Ellipse { center, .. } => pin(*center, 0),
                        EntityKind::EllipticalArc {
                            center, start, end, ..
                        } => {
                            pin(*center, 0);
                            pin(*start, 5);
                            pin(*end, 7);
                        }
                        EntityKind::BSpline { control_points, .. } => {
                            for (i, p) in control_points.iter().enumerate() {
                                pin(*p, 2 * i);
                            }
                        }
                        _ => {}
                    }
                }
                Constraint::Radius { curve, value } => {
                    if let EntityKind::Circle { .. } = sys.entity(*curve).kind {
                        known_radius.insert(*curve, *value);
                    }
                }
                Constraint::Diameter { curve, value } => {
                    if let EntityKind::Circle { .. } = sys.entity(*curve).kind {
                        known_radius.insert(*curve, 0.5 * value);
                    }
                }
                _ => {}
            }
        }
        // Groups.
        for (e, _) in usable.iter().enumerate().filter(|(_, u)| **u) {
            let root = find(&mut parent, e);
            if self.group_of[root] == NONE {
                self.group_of[root] = self.groups.len();
                let (vx, vy) = self.pt_vars(EntityId(root as u32)).unwrap();
                self.groups.push(Group {
                    members: Vec::new(),
                    ground: None,
                    cur: [self.x[vx], self.x[vy]],
                    bad: false,
                });
            }
            let g = self.group_of[root];
            self.group_of[e] = g;
            self.groups[g].members.push(EntityId(e as u32));
        }
        let mut extent = 1.0f64;
        for g in &mut self.groups {
            for m in &g.members {
                if let Some(p) = fixed_at.get(m) {
                    match g.ground {
                        None => g.ground = Some(*p),
                        Some(q) => {
                            if len(sub(*p, q)) > 1e-9 * (1.0 + len(q)) {
                                g.bad = true;
                            }
                        }
                    }
                }
            }
            extent = extent.max(g.cur[0].abs()).max(g.cur[1].abs());
        }
        // Relations.
        let gid = |s: &Self, e: EntityId| {
            let g = s.group_of[e.index()];
            (g != NONE && !s.groups[g].bad).then_some(g)
        };
        let line_ends = |l: EntityId| match sys.entity(l).kind {
            EntityKind::Line { start, end } => Some((start, end)),
            _ => None,
        };
        let mut dists = Vec::new();
        let mut hv = Vec::new();
        let mut online = Vec::new();
        let mut angles = Vec::new();
        for &c in constraints {
            let Some(u) = user(c) else { continue };
            let mut dist = |a: EntityId, b: EntityId, d: f64| {
                if let (Some(ga), Some(gb)) = (gid(self, a), gid(self, b)) {
                    if ga != gb && d > 0.0 {
                        dists.push((ga, gb, d));
                    }
                }
            };
            match u {
                Constraint::Distance { a, b, value } => dist(*a, *b, *value),
                Constraint::Length { line, value } => {
                    let (a, b) = line_ends(*line).unwrap();
                    dist(a, b, *value);
                }
                Constraint::Radius { curve, value } | Constraint::Diameter { curve, value } => {
                    let r = if matches!(u, Constraint::Diameter { .. }) {
                        0.5 * value
                    } else {
                        *value
                    };
                    if let EntityKind::Arc { center, start, end } = sys.entity(*curve).kind {
                        dist(center, start, r);
                        dist(center, end, r);
                    }
                }
                Constraint::PointOnCurve { point, curve } => match sys.entity(*curve).kind {
                    EntityKind::Circle { center, .. } => {
                        if let Some(r) = known_radius.get(curve) {
                            dist(*point, center, *r);
                        }
                    }
                    EntityKind::Line { start, end } => {
                        if let (Some(p), Some(a), Some(b)) =
                            (gid(self, *point), gid(self, start), gid(self, end))
                        {
                            if p != a && p != b && a != b {
                                online.push((p, a, b));
                            }
                        }
                    }
                    _ => {}
                },
                Constraint::Horizontal(l) | Constraint::Vertical(l) => {
                    let (a, b) = line_ends(*l).unwrap();
                    if let (Some(ga), Some(gb)) = (gid(self, a), gid(self, b)) {
                        if ga != gb {
                            let axis = if matches!(u, Constraint::Horizontal(_)) {
                                Axis::H
                            } else {
                                Axis::V
                            };
                            hv.push((ga, gb, axis));
                        }
                    }
                }
                Constraint::Angle { a, b, .. }
                | Constraint::Parallel(a, b)
                | Constraint::Perpendicular(a, b) => {
                    let (Some((a0, a1)), Some((b0, b1))) = (line_ends(*a), line_ends(*b)) else {
                        continue;
                    };
                    let (Some(ga0), Some(ga1), Some(gb0), Some(gb1)) =
                        (gid(self, a0), gid(self, a1), gid(self, b0), gid(self, b1))
                    else {
                        continue;
                    };
                    if ga0 == ga1 || gb0 == gb1 {
                        continue;
                    }
                    let cur = |g0: usize, g1: usize| {
                        let v = sub(self.groups[g1].cur, self.groups[g0].cur);
                        v[1].atan2(v[0])
                    };
                    let start = wrap_angle(cur(gb0, gb1) - cur(ga0, ga1));
                    // Parallel and perpendicular keep the branch the starting geometry has.
                    let theta = match u {
                        Constraint::Angle { value, .. } => *value,
                        Constraint::Parallel(..) => {
                            if start.abs() <= std::f64::consts::FRAC_PI_2 {
                                0.0
                            } else {
                                std::f64::consts::PI
                            }
                        }
                        _ => std::f64::consts::FRAC_PI_2.copysign(start),
                    };
                    angles.push(AngleRel {
                        a: (ga0, ga1),
                        b: (gb0, gb1),
                        theta,
                    });
                }
                Constraint::HorizontalPoints(a, b) | Constraint::VerticalPoints(a, b) => {
                    if let (Some(ga), Some(gb)) = (gid(self, *a), gid(self, *b)) {
                        if ga != gb {
                            let axis = if matches!(u, Constraint::HorizontalPoints(..)) {
                                Axis::H
                            } else {
                                Axis::V
                            };
                            hv.push((ga, gb, axis));
                        }
                    }
                }
                _ => {}
            }
        }
        for &(_, _, d) in &dists {
            extent = extent.max(d);
        }
        self.dists = dists;
        self.hv = hv;
        self.online = online;
        self.angles = angles;
        self.tol = 1e-9 * extent;
        !self.groups.is_empty()
            && (!self.dists.is_empty() || !self.hv.is_empty() || !self.online.is_empty())
    }

    fn push_cluster(&mut self, c: Cluster) -> usize {
        let id = self.clusters.len();
        for g in c.pts.keys() {
            self.index[*g].insert(id);
        }
        self.clusters.push(Some(c));
        id
    }

    /// Initial clusters: ground, oriented pairs, rigid pairs.
    fn seed(&mut self) {
        self.index = vec![BTreeSet::new(); self.groups.len()];
        let mut ground = BTreeMap::new();
        for (g, grp) in self.groups.iter().enumerate() {
            if let (Some(p), false) = (grp.ground, grp.bad) {
                ground.insert(g, p);
            }
        }
        self.push_cluster(Cluster {
            kind: Kind::Ground,
            pts: ground,
        });
        let mut pair_axes: BTreeMap<(usize, usize), (bool, bool)> = BTreeMap::new();
        for &(a, b, axis) in &self.hv {
            let k = (a.min(b), a.max(b));
            let e = pair_axes.entry(k).or_insert((false, false));
            match axis {
                Axis::H => e.0 = true,
                Axis::V => e.1 = true,
            }
        }
        let dists = self.dists.clone();
        for (a, b, d) in dists {
            let k = (a.min(b), a.max(b));
            let (ca, cb) = (self.groups[a].cur, self.groups[b].cur);
            let off = match pair_axes.get(&k) {
                Some((true, true)) => continue, // coincident, contradicted by d > 0
                Some((true, false)) => [if cb[0] >= ca[0] { d } else { -d }, 0.0],
                Some((false, true)) => [0.0, if cb[1] >= ca[1] { d } else { -d }],
                _ => {
                    let v = sub(cb, ca);
                    let l = len(v);
                    if l > 1e-300 {
                        scale(v, d / l)
                    } else {
                        [d, 0.0]
                    }
                }
            };
            let kind = if pair_axes.contains_key(&k) {
                Kind::Oriented
            } else {
                Kind::Rigid
            };
            let mut pts = BTreeMap::new();
            pts.insert(a, ca);
            pts.insert(b, add(ca, off));
            self.push_cluster(Cluster { kind, pts });
        }
    }

    fn live(&self, i: usize) -> &Cluster {
        self.clusters[i].as_ref().unwrap()
    }

    /// Clusters sharing a point with `i`, ascending.
    fn neighbours(&mut self, i: usize) -> Vec<usize> {
        let mut out = BTreeSet::new();
        let mut work = 0;
        for g in self.live(i).pts.keys() {
            work += self.index[*g].len();
            out.extend(self.index[*g].iter().copied().filter(|&j| j != i));
        }
        self.work += work;
        out.into_iter().collect()
    }

    fn shared(&mut self, i: usize, j: usize) -> Vec<usize> {
        let (a, b) = (self.live(i), self.live(j));
        let (small, big) = if a.pts.len() <= b.pts.len() {
            (a, b)
        } else {
            (b, a)
        };
        let out: Vec<usize> = small
            .pts
            .keys()
            .copied()
            .filter(|g| big.pts.contains_key(g))
            .collect();
        self.work += small.pts.len();
        out
    }

    /// Moves cluster `from` into `into` by `m`, after checking every shared point agrees.
    fn absorb(&mut self, into: usize, from: usize, m: Motion, kind: Kind) -> bool {
        let src = self.clusters[from].clone().unwrap();
        let dst = self.live(into);
        for (g, p) in &src.pts {
            if let Some(q) = dst.pts.get(g) {
                if len(sub(m.apply(*p), *q)) > self.tol * 10.0 {
                    let k = (into.min(from), into.max(from));
                    self.refused.insert(k);
                    return false;
                }
            }
        }
        self.clusters[from] = None;
        let dst = self.clusters[into].as_mut().unwrap();
        dst.kind = kind;
        for (g, p) in src.pts {
            self.index[g].remove(&from);
            self.index[g].insert(into);
            dst.pts.entry(g).or_insert_with(|| m.apply(p));
        }
        self.work += self.live(into).pts.len();
        true
    }

    /// Merges two clusters sharing points when their kinds allow it. Returns the survivor.
    fn try_merge(&mut self, i: usize, j: usize) -> Option<usize> {
        if self.refused.contains(&(i.min(j), i.max(j))) {
            return None;
        }
        let shared = self.shared(i, j);
        if shared.is_empty() {
            return None;
        }
        let (ki, kj) = (self.live(i).kind, self.live(j).kind);
        // The stronger cluster keeps its frame (ties: the lower id).
        let (into, from) = if ki > kj || (ki == kj && i < j) {
            (i, j)
        } else {
            (j, i)
        };
        let (kin, kfrom) = (self.live(into).kind, self.live(from).kind);
        if kin == Kind::Ground && kfrom == Kind::Ground {
            return None;
        }
        let m = if kfrom == Kind::Rigid {
            if shared.len() < 2 {
                return None;
            }
            // The two shared points farthest apart.
            let src = self.live(from);
            let p0 = shared[0];
            let far = *shared[1..]
                .iter()
                .max_by(|&&a, &&b| {
                    let da = len(sub(src.pts[&a], src.pts[&p0]));
                    let db = len(sub(src.pts[&b], src.pts[&p0]));
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            let dst = self.live(into);
            let (p1, p2) = (src.pts[&p0], src.pts[&far]);
            let (q1, q2) = (dst.pts[&p0], dst.pts[&far]);
            let (ls, ld) = (len(sub(p2, p1)), len(sub(q2, q1)));
            if (ls - ld).abs() > self.tol * 10.0 || ls <= self.tol {
                self.refused.insert((i.min(j), i.max(j)));
                return None;
            }
            Motion::from_pairs(p1, p2, q1, q2)
        } else {
            // Oriented into oriented or ground: translation by the first shared point.
            let g = shared[0];
            Motion::translation(sub(self.live(into).pts[&g], self.live(from).pts[&g]))
        };
        let kind = kin.max(kfrom);
        self.absorb(into, from, m, kind).then_some(into)
    }

    /// The angle of the rotation fitting cluster `i`'s frame to its points' starting
    /// positions.
    fn frame_angle(&self, i: usize) -> f64 {
        let c = self.live(i);
        let k = c.pts.len() as f64;
        let (mut lc, mut wc) = ([0.0, 0.0], [0.0, 0.0]);
        for (g, p) in &c.pts {
            lc = add(lc, *p);
            wc = add(wc, self.groups[*g].cur);
        }
        lc = scale(lc, 1.0 / k);
        wc = scale(wc, 1.0 / k);
        let (mut sc, mut sd) = (0.0, 0.0);
        for (g, p) in &c.pts {
            let (l, w) = (sub(*p, lc), sub(self.groups[*g].cur, wc));
            sc += cross(l, w);
            sd += dot(l, w);
        }
        sc.atan2(sd)
    }

    /// A rigid cluster holding both points of a horizontal or vertical relation turns to
    /// satisfy it and becomes oriented.
    fn try_orient(&mut self, i: usize) -> bool {
        if self.live(i).kind != Kind::Rigid {
            return false;
        }
        let rel = self.hv.iter().copied().find(|&(a, b, _)| {
            let c = self.live(i);
            c.pts.contains_key(&a) && c.pts.contains_key(&b)
        });
        let Some((a, b, axis)) = rel else {
            return false;
        };
        let c = self.live(i);
        let v = sub(c.pts[&b], c.pts[&a]);
        if len(v) <= self.tol * 10.0 {
            return false;
        }
        let phi = v[1].atan2(v[0]);
        let world = phi + self.frame_angle(i);
        let half = std::f64::consts::FRAC_PI_2;
        let targets: [f64; 2] = match axis {
            Axis::H => [0.0, std::f64::consts::PI],
            Axis::V => [half, -half],
        };
        let wrap = |a: f64| {
            let t = std::f64::consts::TAU;
            let r = a.rem_euclid(t);
            if r > std::f64::consts::PI {
                r - t
            } else {
                r
            }
        };
        let tau = if wrap(targets[0] - world).abs() <= wrap(targets[1] - world).abs() {
            targets[0]
        } else {
            targets[1]
        };
        let (sn, cs) = (tau - phi).sin_cos();
        let c = self.clusters[i].as_mut().unwrap();
        for p in c.pts.values_mut() {
            *p = rot(*p, cs, sn);
        }
        c.kind = Kind::Oriented;
        true
    }

    /// The direction angle of line `(g0, g1)` in cluster `i`'s frame, when it holds both ends.
    fn line_angle(&self, i: usize, l: (usize, usize)) -> Option<f64> {
        let c = self.live(i);
        let (p, q) = (c.pts.get(&l.0)?, c.pts.get(&l.1)?);
        let v = sub(*q, *p);
        (len(v) > self.tol * 10.0).then(|| v[1].atan2(v[0]))
    }

    /// For an angle relation with one line in `i` and the other in `j`: the rotation that
    /// turns `j`'s frame so the relation holds against `i`'s frame.
    fn relative_turn(&self, i: usize, j: usize, r: &AngleRel) -> Option<f64> {
        if let (Some(ai), Some(bj)) = (self.line_angle(i, r.a), self.line_angle(j, r.b)) {
            return Some(ai + r.theta - bj);
        }
        if let (Some(bi), Some(aj)) = (self.line_angle(i, r.b), self.line_angle(j, r.a)) {
            return Some(bi - r.theta - aj);
        }
        None
    }

    /// Angle rule against fixed orientation: a rigid cluster holding one line of an angle
    /// relation whose other line lies in an oriented (or ground) cluster turns until the
    /// relation holds and becomes oriented.
    fn try_angle_orient(&mut self, i: usize) -> bool {
        if self.live(i).kind != Kind::Rigid {
            return false;
        }
        let rels = self.angles.clone();
        for r in &rels {
            for j in 0..self.clusters.len() {
                if j == i || self.clusters[j].is_none() || self.live(j).kind == Kind::Rigid {
                    continue;
                }
                self.work += 1;
                // `j`'s frame is the world's orientation; turn `i` onto it.
                let Some(turn) = self.relative_turn(j, i, r) else {
                    continue;
                };
                let (sn, cs) = turn.sin_cos();
                let c = self.clusters[i].as_mut().unwrap();
                for p in c.pts.values_mut() {
                    *p = rot(*p, cs, sn);
                }
                c.kind = Kind::Oriented;
                return true;
            }
        }
        false
    }

    /// Hinge rule: two rigid clusters sharing exactly one point, with an angle relation
    /// between a line of each, are one rigid cluster: `j` turns about the hinge point until the
    /// relation holds and is absorbed into `i`.
    fn try_hinge(&mut self, i: usize, j: usize) -> Option<usize> {
        if self.live(i).kind != Kind::Rigid || self.live(j).kind != Kind::Rigid {
            return None;
        }
        if self.refused.contains(&(i.min(j), i.max(j))) {
            return None;
        }
        let shared = self.shared(i, j);
        if shared.len() != 1 {
            return None;
        }
        let h = shared[0];
        let rels = self.angles.clone();
        let turn = rels.iter().find_map(|r| self.relative_turn(i, j, r))?;
        let (sn, cs) = turn.sin_cos();
        let (pi, pj) = (self.live(i).pts[&h], self.live(j).pts[&h]);
        let m = Motion {
            c: cs,
            s: sn,
            t: sub(pi, rot(pj, cs, sn)),
        };
        // Other angle relations between the two must agree with the turn.
        for r in &rels {
            if let Some(t) = self.relative_turn(i, j, r) {
                if wrap_angle(t - turn).abs() > 1e-9 {
                    self.refused.insert((i.min(j), i.max(j)));
                    return None;
                }
            }
        }
        self.absorb(i, j, m, Kind::Rigid).then_some(i)
    }

    /// Triangle rule with `i` as one of the three clusters. Returns the survivor.
    fn try_triangle(&mut self, i: usize) -> Option<usize> {
        let nb = self.neighbours(i);
        for &j in &nb {
            let sij = self.shared(i, j);
            if sij.len() != 1 {
                continue;
            }
            let a = sij[0];
            let nj = self.neighbours(j);
            for &k in &nj {
                if k == i || self.work > MAX_WORK {
                    continue;
                }
                let sjk = self.shared(j, k);
                let sik = self.shared(i, k);
                if sjk.len() != 1 || sik.len() != 1 {
                    continue;
                }
                let (b, c) = (sjk[0], sik[0]);
                if a == b || b == c || a == c {
                    continue;
                }
                if let Some(s) = self.triangle(i, j, k, a, b, c) {
                    return Some(s);
                }
            }
        }
        None
    }

    /// Clusters `p` (∋ a, c), `q` (∋ a, b), `r` (∋ b, c). At most one may be non-rigid; it
    /// becomes the frame.
    fn triangle(
        &mut self,
        p: usize,
        q: usize,
        r: usize,
        a: usize,
        b: usize,
        c: usize,
    ) -> Option<usize> {
        let ids = [(p, a, c, b), (q, b, a, c), (r, c, b, a)];
        let non_rigid: Vec<_> = ids
            .iter()
            .filter(|t| self.live(t.0).kind != Kind::Rigid)
            .collect();
        if non_rigid.len() > 1 {
            return None;
        }
        // The frame cluster knows two of the shared points (u, v); w is constructed.
        let (f, u, v, w) = match non_rigid.first() {
            Some(t) => **t,
            None => *ids.iter().min_by_key(|t| t.0).unwrap(),
        };
        let others: Vec<usize> = [p, q, r].into_iter().filter(|&x| x != f).collect();
        let dist_in = |s: &Self, cl: usize, g1: usize, g2: usize| {
            let c = s.live(cl);
            len(sub(c.pts[&g2], c.pts[&g1]))
        };
        // Sides |uw| and |vw| come from the two other clusters (each holds w and one of u, v).
        let mut ruw = None;
        let mut rvw = None;
        for &o in &others {
            let cl = self.live(o);
            if cl.pts.contains_key(&u) && cl.pts.contains_key(&w) {
                ruw = Some((o, dist_in(self, o, u, w)));
            } else if cl.pts.contains_key(&v) && cl.pts.contains_key(&w) {
                rvw = Some((o, dist_in(self, o, v, w)));
            }
        }
        let ((ou, du), (ov, dv)) = (ruw?, rvw?);
        let fu = self.live(f).pts[&u];
        let fv = self.live(f).pts[&v];
        let d = len(sub(fv, fu));
        if d <= self.tol * 10.0 {
            return None;
        }
        let e = scale(sub(fv, fu), 1.0 / d);
        let along = (d * d + du * du - dv * dv) / (2.0 * d);
        let h2 = du * du - along * along;
        if h2 < -self.tol * 10.0 * (du + dv + d) {
            return None; // no triangle: leave the contradiction to the numerical solve
        }
        let h = h2.max(0.0).sqrt();
        // Owen's rule: keep the starting orientation of (u, v, w).
        let (cu, cv, cw) = (self.groups[u].cur, self.groups[v].cur, self.groups[w].cur);
        let sgn = if cross(sub(cv, cu), sub(cw, cu)) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let fw = add(add(fu, scale(e, along)), scale([-e[1], e[0]], sgn * h));
        let mu = {
            let cl = self.live(ou);
            Motion::from_pairs(cl.pts[&u], cl.pts[&w], fu, fw)
        };
        let mv = {
            let cl = self.live(ov);
            Motion::from_pairs(cl.pts[&v], cl.pts[&w], fv, fw)
        };
        let kind = self.live(f).kind;
        // Insert w into the frame first so both absorptions check against it.
        self.clusters[f].as_mut().unwrap().pts.insert(w, fw);
        self.index[w].insert(f);
        let ok_u = self.absorb(f, ou, mu, kind);
        let ok_v = ok_u && self.absorb(f, ov, mv, kind);
        if !ok_u || !ok_v {
            // Roll back the constructed point (the absorbed cluster, if any, stays merged:
            // it was consistent).
            if !ok_u {
                self.clusters[f].as_mut().unwrap().pts.remove(&w);
                self.index[w].remove(&f);
            }
            return if ok_u { Some(f) } else { None };
        }
        Some(f)
    }

    /// Loci rule: places one unplaced point from two loci on ground geometry.
    fn try_loci(&mut self) -> bool {
        let ground = self.live(0).pts.clone();
        if ground.is_empty() {
            return false;
        }
        #[derive(Clone, Copy)]
        enum Locus {
            Circle([f64; 2], f64),
            Line([f64; 2], [f64; 2]),
        }
        let mut loci: BTreeMap<usize, Vec<Locus>> = BTreeMap::new();
        for &(a, b, d) in &self.dists {
            for (p, q) in [(a, b), (b, a)] {
                if !ground.contains_key(&p) {
                    if let Some(c) = ground.get(&q) {
                        loci.entry(p).or_default().push(Locus::Circle(*c, d));
                    }
                }
            }
        }
        for &(a, b, axis) in &self.hv {
            let dir = if axis == Axis::H {
                [1.0, 0.0]
            } else {
                [0.0, 1.0]
            };
            for (p, q) in [(a, b), (b, a)] {
                if !ground.contains_key(&p) {
                    if let Some(c) = ground.get(&q) {
                        loci.entry(p).or_default().push(Locus::Line(*c, dir));
                    }
                }
            }
        }
        for &(p, a, b) in &self.online {
            if !ground.contains_key(&p) {
                if let (Some(pa), Some(pb)) = (ground.get(&a), ground.get(&b)) {
                    let d = sub(*pb, *pa);
                    if len(d) > self.tol * 10.0 {
                        loci.entry(p).or_default().push(Locus::Line(*pa, d));
                    }
                }
            }
        }
        self.work += loci.len();
        for (p, ls) in loci {
            if ls.len() < 2 || self.groups[p].bad {
                continue;
            }
            let cur = self.groups[p].cur;
            let tol = self.tol * 10.0;
            let solve2 = |l1: Locus, l2: Locus| -> Option<[f64; 2]> {
                match (l1, l2) {
                    (Locus::Line(p1, d1), Locus::Line(p2, d2)) => {
                        let den = cross(d1, d2);
                        if den.abs() <= 1e-9 * len(d1) * len(d2) {
                            return None;
                        }
                        let s = cross(sub(p2, p1), d2) / den;
                        Some(add(p1, scale(d1, s)))
                    }
                    (Locus::Line(p0, d), Locus::Circle(c, r))
                    | (Locus::Circle(c, r), Locus::Line(p0, d)) => {
                        let u = scale(d, 1.0 / len(d));
                        let foot = add(p0, scale(u, dot(sub(c, p0), u)));
                        let off = len(sub(foot, c));
                        let h2 = r * r - off * off;
                        if h2 < -tol * (r + 1.0) {
                            return None;
                        }
                        let h = h2.max(0.0).sqrt();
                        let (s1, s2) = (add(foot, scale(u, h)), sub(foot, scale(u, h)));
                        Some(if len(sub(s1, cur)) <= len(sub(s2, cur)) {
                            s1
                        } else {
                            s2
                        })
                    }
                    (Locus::Circle(c1, r1), Locus::Circle(c2, r2)) => {
                        let d = len(sub(c2, c1));
                        if d <= tol {
                            return None;
                        }
                        let e = scale(sub(c2, c1), 1.0 / d);
                        let along = (d * d + r1 * r1 - r2 * r2) / (2.0 * d);
                        let h2 = r1 * r1 - along * along;
                        if h2 < -tol * (r1 + r2 + d) {
                            return None;
                        }
                        let h = h2.max(0.0).sqrt();
                        let sgn = if cross(sub(c2, c1), sub(cur, c1)) >= 0.0 {
                            1.0
                        } else {
                            -1.0
                        };
                        Some(add(add(c1, scale(e, along)), scale([-e[1], e[0]], sgn * h)))
                    }
                }
            };
            let mut placed = None;
            'pairs: for i in 0..ls.len() {
                for j in i + 1..ls.len() {
                    if let Some(s) = solve2(ls[i], ls[j]) {
                        placed = Some(s);
                        break 'pairs;
                    }
                }
            }
            let Some(s) = placed else { continue };
            let consistent = ls.iter().all(|l| match *l {
                Locus::Circle(c, r) => (len(sub(s, c)) - r).abs() <= tol * (1.0 + r),
                Locus::Line(p0, d) => (cross(d, sub(s, p0)) / len(d)).abs() <= tol,
            });
            if !consistent {
                self.groups[p].bad = true;
                continue;
            }
            self.clusters[0].as_mut().unwrap().pts.insert(p, s);
            self.index[p].insert(0);
            return true;
        }
        false
    }

    /// Applies the rules until nothing changes (or the work budget runs out).
    fn saturate(&mut self) {
        let mut queue: BTreeSet<usize> = (0..self.clusters.len()).collect();
        loop {
            while let Some(i) = queue.pop_first() {
                if self.work > MAX_WORK {
                    return;
                }
                if self.clusters[i].is_none() {
                    continue;
                }
                if self.try_orient(i) || self.try_angle_orient(i) {
                    queue.insert(i);
                    queue.extend(self.neighbours(i));
                    continue;
                }
                let mut changed = None;
                for j in self.neighbours(i) {
                    if self.clusters[j].is_none() {
                        continue;
                    }
                    if let Some(s) = self.try_merge(i, j) {
                        changed = Some(s);
                        break;
                    }
                    if let Some(s) = self.try_hinge(i, j) {
                        changed = Some(s);
                        break;
                    }
                }
                if changed.is_none() {
                    changed = self.try_triangle(i);
                }
                if let Some(s) = changed {
                    queue.insert(s);
                    queue.extend(self.neighbours(s));
                }
            }
            if self.work > MAX_WORK || !self.try_loci() {
                return;
            }
            queue.insert(0);
            queue.extend(self.neighbours(0));
        }
    }

    fn finish(&self, is_free: &[bool]) -> Construction {
        let mut pos: Vec<Option<[f64; 2]>> = vec![None; self.groups.len()];
        for (g, p) in &self.live(0).pts {
            pos[*g] = Some(*p);
        }
        // Floating clusters, largest first, fitted to their starting positions.
        let mut floating: Vec<usize> = (1..self.clusters.len())
            .filter(|&i| self.clusters[i].as_ref().is_some_and(|c| c.pts.len() >= 2))
            .collect();
        floating.sort_by_key(|&i| (std::cmp::Reverse(self.live(i).pts.len()), i));
        for &i in &floating {
            let c = self.live(i);
            if c.pts.keys().any(|g| pos[*g].is_some()) {
                continue;
            }
            let ang = if c.kind == Kind::Rigid {
                self.frame_angle(i)
            } else {
                0.0
            };
            let (sn, cs) = ang.sin_cos();
            let k = c.pts.len() as f64;
            let (mut lc, mut wc) = ([0.0, 0.0], [0.0, 0.0]);
            for (g, p) in &c.pts {
                lc = add(lc, rot(*p, cs, sn));
                wc = add(wc, self.groups[*g].cur);
            }
            let t = sub(scale(wc, 1.0 / k), scale(lc, 1.0 / k));
            let m = Motion { c: cs, s: sn, t };
            for (g, p) in &c.pts {
                pos[*g] = Some(m.apply(*p));
            }
        }
        let mut out = Construction {
            clusters: self
                .clusters
                .iter()
                .flatten()
                .filter(|c| c.pts.len() >= 2)
                .count(),
            ..Construction::default()
        };
        for (g, grp) in self.groups.iter().enumerate() {
            let Some(p) = pos[g] else { continue };
            for m in &grp.members {
                let (vx, vy) = self.pt_vars(*m).unwrap();
                if !(is_free[vx] && is_free[vy]) {
                    continue;
                }
                if self.x[vx] != p[0] || self.x[vy] != p[1] {
                    out.points += 1;
                }
                out.values.push((vx, p[0]));
                out.values.push((vy, p[1]));
            }
        }
        out
    }
}
