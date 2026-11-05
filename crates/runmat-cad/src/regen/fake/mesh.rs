//! Display meshes of the test kernel's solids drawn on their true surfaces.
//!
//! The grid decides occupancy, so a curved face is a staircase of cell facets. The tool that
//! made the face knows its true surface ([`Exact`]), and the mesher rebuilds the boundary
//! representation from the staircase the way a B-rep tessellator works:
//!
//! 1. **Loops.** Each face's boundary is walked over its facets, giving closed loops of grid
//!    points with the face across each boundary segment.
//! 2. **Corners and curves.** A grid point where the face across changes is a corner; corners
//!    move onto the intersection of every surface meeting there. Between corners, a run of
//!    boundary points shared by two faces becomes one curve: its points move onto the
//!    intersection of the two surfaces, and the curve is resampled at a step tied to the
//!    surfaces' radius (the angular tolerance, and the chordal tolerance scaled to the body).
//!    Both faces use the same samples, so the mesh is watertight across every curve.
//! 3. **Faces.** Each face's loops are mapped to its surface chart (cut along the seam when
//!    a loop goes around a periodic surface), triangulated with a constrained Delaunay
//!    triangulation, refined with interior points on curved surfaces, and mapped back onto
//!    the surface. Vertices are shared within a face, so shading is smooth inside a face and
//!    creased only at face boundaries.
//!
//! Faces whose surface has no chart (lofted spans, drafted walls, scaled bodies) draw on their
//! facets, and the curves they share keep the grid points, so the mesh stays watertight.
//! Convex fillets and chamfers (blend faces laid over a grid edge) trim the faces they meet
//! back to their tangent lines and draw as a strip between them, with their end arcs added to
//! the faces at either end.

use super::cdt;
use super::exact::Exact;
use super::*;
use crate::regen::kernel::{BodyMesh, FaceMesh, TessellationOptions};
use std::collections::HashMap;
use std::f64::consts::FRAC_PI_2;

type P3 = [f64; 3];

fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: P3, b: P3) -> P3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn mul(a: P3, s: f64) -> P3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn len(a: P3) -> f64 {
    dot(a, a).sqrt()
}
fn lerp(a: P3, b: P3, t: f64) -> P3 {
    add(a, mul(sub(b, a), t))
}

/// The chart of a grid plane: `u`, `v` along the next two axes, so `∂u × ∂v` is `+axis`.
pub(super) fn grid_plane(p: &PlaneKey) -> Exact {
    let a = p.axis;
    let (u, v) = ((a + 1) % 3, (a + 2) % 3);
    let mut o = [0.0; 3];
    o[a] = p.offset;
    let (mut e1, mut e2) = ([0.0; 3], [0.0; 3]);
    e1[u] = 1.0;
    e2[v] = 1.0;
    Exact::plane(o, e1, e2)
}

/// One cell facet: its face, its corners counter-clockwise seen from outside, and the cell it
/// bounds.
struct Facet {
    face: usize,
    v: [usize; 4],
    cell: [usize; 3],
}

/// How a face draws.
enum Draw {
    /// On its surface chart.
    Chart(Arc<Exact>),
    /// On its facets.
    Facets,
    /// A blend face, drawn from its decoration.
    Blend,
}

/// A curve shared by two faces, in the orientation of its lower face.
struct Run {
    faces: (usize, usize),
    /// Grid points, first to last (a closed run repeats its first point at the end).
    grid: Vec<usize>,
    /// The drawn points, as indices into the point table.
    pts: Vec<usize>,
}

/// A face's loop: runs in order, each with whether it is walked against its stored
/// orientation.
type Loop = Vec<(usize, bool)>;

struct Ctx<'a> {
    s: &'a FakeSolid,
    /// Drawn point positions: grid points first, then curve samples.
    pos: Vec<P3>,
    raw: Vec<P3>,
    draw: Vec<Draw>,
    runs: Vec<Run>,
    loops: Vec<Vec<Loop>>,
    /// Faces meeting at each grid point that is a corner.
    corner_faces: HashMap<usize, BTreeSet<usize>>,
    /// The largest angle one curve segment may turn through.
    dtheta: f64,
    /// The cell size scale: the body's diagonal.
    diag: f64,
}

fn grid_index(g: &[f64], x: f64) -> usize {
    g.partition_point(|v| *v < x - EPS)
}

/// The display mesh of `s` on its true surfaces, or `None` when the solid has a feature this
/// mesher does not draw (the caller then draws the facets).
pub(super) fn tessellate(s: &FakeSolid, options: &TessellationOptions) -> Option<BodyMesh> {
    let (lo, hi) = s.bounds()?;
    let diag = len(sub(hi, lo)).max(1e-12);
    let tol = options.chord_tolerance.clamp(1e-5 * diag, 1e-3 * diag);
    let world = |c: [usize; 3]| [s.grid[0][c[0]], s.grid[1][c[1]], s.grid[2][c[2]]];

    // Facets, one per cell, wound outwards; grid points numbered as they are met.
    let mut facets: Vec<Facet> = Vec::new();
    let mut dense: HashMap<[usize; 3], usize> = HashMap::new();
    let mut raw: Vec<P3> = Vec::new();
    for (fi, f) in s.faces.iter().enumerate() {
        for (p, r) in f.rect_planes() {
            let a = p.axis;
            let (u, v) = ((a + 1) % 3, (a + 2) % 3);
            let ia = grid_index(&s.grid[a], p.offset);
            let (u0, u1) = (
                grid_index(&s.grid[u], r.lo[u]),
                grid_index(&s.grid[u], r.hi[u]),
            );
            let (v0, v1) = (
                grid_index(&s.grid[v], r.lo[v]),
                grid_index(&s.grid[v], r.hi[v]),
            );
            for iu in u0..u1 {
                for iv in v0..v1 {
                    let corner = |du: usize, dv: usize| {
                        let mut c = [0usize; 3];
                        c[a] = ia;
                        c[u] = iu + du;
                        c[v] = iv + dv;
                        c
                    };
                    let cs = [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)];
                    let mut ids = cs.map(|c| {
                        *dense.entry(c).or_insert_with(|| {
                            raw.push(world(c));
                            raw.len() - 1
                        })
                    });
                    if p.dir < 0 {
                        ids = [ids[0], ids[3], ids[2], ids[1]];
                    }
                    let mut cell = corner(0, 0);
                    cell[a] = if p.dir > 0 { ia.saturating_sub(1) } else { ia };
                    facets.push(Facet {
                        face: fi,
                        v: ids,
                        cell,
                    });
                }
            }
        }
    }
    if facets.is_empty() {
        return None;
    }
    // Directed facet edges.
    let mut half: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (k, f) in facets.iter().enumerate() {
        for e in 0..4 {
            half.entry((f.v[e], f.v[(e + 1) % 4])).or_default().push(k);
        }
    }

    // How each face draws.
    let blend_faces: BTreeSet<usize> = s
        .decorations
        .iter()
        .flat_map(|d| d.pieces.iter().map(|p| p.2 as usize))
        .collect();
    let draw: Vec<Draw> = s
        .faces
        .iter()
        .enumerate()
        .map(|(fi, f)| {
            if blend_faces.contains(&fi) || !f.is_grid() {
                Draw::Blend
            } else if let Some(p) = f.plane {
                Draw::Chart(Arc::new(grid_plane(&p)))
            } else if let Some(e) = &f.exact {
                Draw::Chart(e.clone())
            } else {
                Draw::Facets
            }
        })
        .collect();

    // Boundary loops per face: (point, face across the segment leaving it).
    let mut face_loops: Vec<Vec<Vec<(usize, usize)>>> = vec![Vec::new(); s.faces.len()];
    {
        let in_face = |a: usize, b: usize, face: usize| -> Option<usize> {
            half.get(&(a, b))
                .and_then(|l| l.iter().copied().find(|k| facets[*k].face == face))
        };
        let next_in = |k: usize, a: usize| -> usize {
            let f = &facets[k];
            let i = f.v.iter().position(|x| *x == a).expect("corner of facet");
            f.v[(i + 1) % 4]
        };
        let mut seen: BTreeSet<(usize, usize, usize)> = BTreeSet::new();
        for (k0, f0) in facets.iter().enumerate() {
            let face = f0.face;
            for e0 in 0..4 {
                let (a0, b0) = (f0.v[e0], f0.v[(e0 + 1) % 4]);
                if in_face(b0, a0, face).is_some() || seen.contains(&(a0, b0, face)) {
                    continue;
                }
                let mut lp: Vec<(usize, usize)> = Vec::new();
                let (mut k, mut a, mut b) = (k0, a0, b0);
                let mut guard = 0;
                loop {
                    guard += 1;
                    if guard > 4 * facets.len() + 16 {
                        return None;
                    }
                    seen.insert((a, b, face));
                    let across = half
                        .get(&(b, a))
                        .and_then(|l| {
                            l.iter()
                                .copied()
                                .find(|j| facets[*j].cell == facets[k].cell)
                                .or_else(|| l.first().copied())
                        })
                        .map(|j| facets[j].face)?;
                    lp.push((a, across));
                    // The next boundary segment from b: turn through the face's facets.
                    let mut kk = k;
                    let mut c = next_in(kk, b);
                    let mut turns = 0;
                    while let Some(g) = in_face(c, b, face) {
                        kk = g;
                        c = next_in(kk, b);
                        turns += 1;
                        if turns > 8 {
                            return None;
                        }
                    }
                    k = kk;
                    a = b;
                    b = c;
                    if (a, b) == (a0, b0) {
                        break;
                    }
                }
                face_loops[face].push(lp);
            }
        }
    }

    // Corners: points where the face across changes, or a loop passes twice.
    let mut corner_faces: HashMap<usize, BTreeSet<usize>> = HashMap::new();
    let mut is_corner: BTreeSet<usize> = BTreeSet::new();
    for (face, loops) in face_loops.iter().enumerate() {
        for lp in loops {
            let n = lp.len();
            let mut count: BTreeMap<usize, usize> = BTreeMap::new();
            for (p, _) in lp {
                *count.entry(*p).or_default() += 1;
            }
            for i in 0..n {
                let (p, out) = lp[i];
                let (_, inn) = lp[(i + n - 1) % n];
                if out != inn || count[&p] > 1 {
                    is_corner.insert(p);
                }
                let _ = face;
            }
        }
    }
    for (face, loops) in face_loops.iter().enumerate() {
        for lp in loops {
            let n = lp.len();
            for i in 0..n {
                let (p, out) = lp[i];
                if is_corner.contains(&p) {
                    let e = corner_faces.entry(p).or_default();
                    e.insert(face);
                    e.insert(out);
                    e.insert(lp[(i + n - 1) % n].1);
                }
            }
        }
    }

    let dtheta = {
        // The angle a chord of the tolerance subtends on a unit radius scales with the radius;
        // `segment_angle` applies it per radius.
        options
            .angle_tolerance
            .clamp(1f64.to_radians(), 30f64.to_radians())
    };
    let mut ctx = Ctx {
        s,
        pos: raw.clone(),
        raw,
        draw,
        runs: Vec::new(),
        loops: vec![Vec::new(); s.faces.len()],
        corner_faces,
        dtheta,
        diag,
    };
    let tol_abs = tol;

    // Snap corners.
    let corners: Vec<usize> = is_corner.iter().copied().collect();
    for p in corners {
        let faces = ctx.corner_faces[&p].clone();
        if let Some(surfs) = ctx.surfaces(&faces) {
            ctx.pos[p] = snap_or(ctx.raw[p], &surfs, ctx.diag);
        }
    }

    // Runs.
    let mut run_of: HashMap<(usize, usize, Vec<usize>), usize> = HashMap::new();
    for face in 0..s.faces.len() {
        for lp in &face_loops[face] {
            let n = lp.len();
            let start = (0..n).find(|i| is_corner.contains(&lp[*i].0));
            let mut pieces: Vec<(usize, Vec<usize>)> = Vec::new();
            match start {
                None => {
                    let mut pts: Vec<usize> = lp.iter().map(|x| x.0).collect();
                    pts.push(pts[0]);
                    pieces.push((lp[0].1, pts));
                }
                Some(s0) => {
                    let mut cur: Vec<usize> = vec![lp[s0].0];
                    let mut across = lp[s0].1;
                    for step in 1..=n {
                        let (p, out) = lp[(s0 + step) % n];
                        cur.push(p);
                        if is_corner.contains(&p) || step == n {
                            pieces.push((across, std::mem::take(&mut cur)));
                            cur.push(p);
                            across = out;
                        }
                    }
                }
            }
            let mut l: Loop = Vec::new();
            for (g, pts) in pieces {
                let closed = pts.first() == pts.last() && !is_corner.contains(&pts[0]);
                let (lo_f, hi_f) = (face.min(g), face.max(g));
                let mut canon = if face <= g {
                    pts.clone()
                } else {
                    pts.iter().rev().copied().collect()
                };
                if closed {
                    canon.pop();
                    let m = (0..canon.len()).min_by_key(|i| canon[*i]).unwrap_or(0);
                    canon.rotate_left(m);
                    canon.push(canon[0]);
                }
                let key = (lo_f, hi_f, canon.clone());
                let id = *run_of.entry(key).or_insert_with(|| {
                    ctx.runs.push(Run {
                        faces: (lo_f, hi_f),
                        grid: canon.clone(),
                        pts: Vec::new(),
                    });
                    ctx.runs.len() - 1
                });
                let reversed = face > g;
                l.push((id, reversed));
            }
            ctx.loops[face].push(l);
        }
    }
    // Closed runs walked from a different start in one face than the canonical one are the
    // same curve; the face's loop is the whole run either way.
    for r in 0..ctx.runs.len() {
        ctx.sample_run(r, tol_abs);
    }
    // Blends trim their faces and add their own.
    let blend_meshes = blends::apply(&mut ctx)?;

    // Faces.
    let mut faces = vec![FaceMesh::default(); s.faces.len()];
    for face in 0..s.faces.len() {
        let mesh = match &ctx.draw[face] {
            Draw::Chart(e) => {
                let e = e.clone();
                ctx.face_on_chart(face, &e, &facets, tol_abs)
                    .unwrap_or_else(|| ctx.face_on_facets(face, &facets))
            }
            Draw::Facets => ctx.face_on_facets(face, &facets),
            Draw::Blend => blend_meshes.get(&face).cloned().unwrap_or_default(),
        };
        faces[face] = mesh;
    }

    // Edges: each topology edge along the curve of its two faces.
    let edges = s
        .topology
        .edges
        .iter()
        .map(|e| ctx.edge_polyline(e))
        .collect();
    let vertices = s
        .topology
        .vertices
        .iter()
        .map(|v| {
            let c = [0, 1, 2].map(|a| grid_index(&s.grid[a], v.point[a]));
            match dense.get(&c) {
                Some(id) if len(sub(ctx.raw[*id], v.point)) < 1e-9 => ctx.pos[*id],
                _ => v.point,
            }
        })
        .collect();
    Some(BodyMesh {
        faces,
        edges,
        vertices,
    })
}

/// The point nearest `p0` on every surface at once (least-norm Newton steps).
fn snap(p0: P3, surfs: &[Arc<Exact>], diag: f64) -> Option<P3> {
    let mut p = p0;
    let k = surfs.len();
    if k == 0 {
        return Some(p);
    }
    for _ in 0..60 {
        let mut n: Vec<P3> = Vec::with_capacity(k);
        let mut f: Vec<f64> = Vec::with_capacity(k);
        for s in surfs {
            let uv = s.invert(p);
            let q = s.eval(uv[0], uv[1]);
            let nn = s.normal(uv[0], uv[1]);
            n.push(nn);
            f.push(dot(nn, sub(p, q)));
        }
        // (N Nᵀ + λI) y = f; dp = −Nᵀ y.
        let lam = 1e-9;
        let mut m = vec![vec![0.0; k + 1]; k];
        for i in 0..k {
            for j in 0..k {
                m[i][j] = dot(n[i], n[j]) + if i == j { lam } else { 0.0 };
            }
            m[i][k] = f[i];
        }
        let y = solve(m)?;
        let mut dp = [0.0; 3];
        for i in 0..k {
            dp = sub(dp, mul(n[i], y[i]));
        }
        p = add(p, dp);
        if len(dp) < 1e-14 * diag {
            break;
        }
    }
    if len(sub(p, p0)) > 0.05 * diag || p.iter().any(|x| !x.is_finite()) {
        return None;
    }
    Some(p)
}

/// Gaussian elimination with partial pivoting on an augmented matrix.
fn solve(mut m: Vec<Vec<f64>>) -> Option<Vec<f64>> {
    let k = m.len();
    for c in 0..k {
        let piv = (c..k).max_by(|a, b| m[*a][c].abs().total_cmp(&m[*b][c].abs()))?;
        if m[piv][c].abs() < 1e-300 {
            return None;
        }
        m.swap(c, piv);
        for r in 0..k {
            if r != c {
                let t = m[r][c] / m[c][c];
                for j in c..=k {
                    m[r][j] -= t * m[c][j];
                }
            }
        }
    }
    Some((0..k).map(|i| m[i][k] / m[i][i]).collect())
}

/// [`snap`], keeping the start point when the surfaces have no common point near it.
fn snap_or(p0: P3, surfs: &[Arc<Exact>], diag: f64) -> P3 {
    snap(p0, surfs, diag).unwrap_or(p0)
}

impl Ctx<'_> {
    /// The surfaces of `faces`, when every one draws on a chart.
    fn surfaces(&self, faces: &BTreeSet<usize>) -> Option<Vec<Arc<Exact>>> {
        let mut out = Vec::new();
        for f in faces {
            match self.draw.get(*f)? {
                Draw::Chart(e) => out.push(e.clone()),
                _ => return None,
            }
        }
        Some(out)
    }

    fn chart(&self, f: usize) -> Option<Arc<Exact>> {
        match &self.draw[f] {
            Draw::Chart(e) => Some(e.clone()),
            _ => None,
        }
    }

    /// The largest step along a curve of radius `r` for the tolerances.
    fn step_for(&self, r: f64, tol: f64) -> f64 {
        if !r.is_finite() {
            return f64::INFINITY;
        }
        let chord = if tol < r {
            2.0 * (1.0 - tol / r).acos()
        } else {
            FRAC_PI_2
        };
        r * self.dtheta.min(chord).max(1e-3)
    }

    fn sample_run(&mut self, r: usize, tol: f64) {
        let (fa, fb) = self.runs[r].faces;
        let grid = self.runs[r].grid.clone();
        let closed = grid.first() == grid.last() && grid.len() > 2;
        let (sa, sb) = (self.chart(fa), self.chart(fb));
        let (Some(sa), Some(sb)) = (sa, sb) else {
            self.runs[r].pts = grid;
            return;
        };
        let curved = sa.curved() || sb.curved();
        if !curved {
            // Two planes meet on a line: the ends are enough when the run is straight.
            let (p0, p1) = (self.pos[grid[0]], self.pos[grid[grid.len() - 1]]);
            let d = sub(p1, p0);
            let l = len(d);
            let straight = !closed
                && l > 0.0
                && grid.iter().all(|g| {
                    let q = sub(self.raw[*g], self.raw[grid[0]]);
                    let dr = sub(self.raw[grid[grid.len() - 1]], self.raw[grid[0]]);
                    len(cross(q, dr)) <= 1e-9 * len(dr).max(1e-300) * self.diag
                });
            self.runs[r].pts = if straight {
                vec![grid[0], grid[grid.len() - 1]]
            } else {
                grid
            };
            return;
        }
        let surfs = vec![sa.clone(), sb.clone()];
        // Interior points onto the curve; the ends are corners, already placed.
        let mut pts: Vec<P3> = Vec::with_capacity(grid.len());
        for (i, g) in grid.iter().enumerate() {
            let end = !closed && (i == 0 || i == grid.len() - 1);
            pts.push(if end {
                self.pos[*g]
            } else {
                snap_or(self.raw[*g], &surfs, self.diag)
            });
        }
        if closed {
            let q = snap_or(self.raw[grid[0]], &surfs, self.diag);
            pts[0] = q;
            let n = pts.len();
            pts[n - 1] = q;
            self.pos[grid[0]] = q;
        }
        let clean = clean_polyline(&pts);
        let total: f64 = clean.windows(2).map(|w| len(sub(w[1], w[0]))).sum();
        if total <= 1e-12 * self.diag {
            self.runs[r].pts = grid;
            return;
        }
        // Straight curves (a generator line, a tangent line) need only their ends.
        if !closed {
            let (p0, p1) = (clean[0], clean[clean.len() - 1]);
            let chord = sub(p1, p0);
            let cl = len(chord);
            if cl > 1e-12 {
                let dev = clean
                    .iter()
                    .map(|q| len(cross(sub(*q, p0), chord)) / cl)
                    .fold(0.0, f64::max);
                if dev <= tol {
                    self.runs[r].pts = vec![grid[0], grid[grid.len() - 1]];
                    return;
                }
            }
        }
        let radius = clean
            .iter()
            .flat_map(|q| {
                let mut v = Vec::new();
                for s in [&sa, &sb] {
                    if s.curved() {
                        let rr = s.radii(*q);
                        v.push(rr[0].min(rr[1]));
                    }
                }
                v
            })
            .fold(f64::INFINITY, f64::min);
        let h = self.step_for(radius, tol).max(total / 4000.0);
        let mut n = (total / h).ceil() as usize;
        n = n.max(if closed { 12 } else { 1 });
        // A closed curve around a periodic surface (a hole's rim) is sampled at even steps
        // of the surface's angle from its seam, so rims of one surface line up.
        if closed {
            // A multiple of four, so the quadrant points (the silhouettes of the standard
            // views) are samples.
            let n4 = n.div_ceil(4) * 4;
            if let Some(out) = self.sample_closed_by_angle(r, &clean, n4, &surfs, &sa, &sb) {
                self.runs[r].pts = out;
                return;
            }
        }
        let mut cum = vec![0.0];
        for w in clean.windows(2) {
            let l = cum[cum.len() - 1] + len(sub(w[1], w[0]));
            cum.push(l);
        }
        let mut out = vec![grid[0]];
        for i in 1..n {
            let t = total * i as f64 / n as f64;
            let k = cum.partition_point(|c| *c < t).clamp(1, clean.len() - 1);
            let seg = cum[k] - cum[k - 1];
            let f = if seg > 0.0 {
                (t - cum[k - 1]) / seg
            } else {
                0.0
            };
            let guess = lerp(clean[k - 1], clean[k], f);
            let q = snap_or(guess, &surfs, self.diag);
            self.pos.push(q);
            self.raw.push(q);
            out.push(self.pos.len() - 1);
        }
        out.push(grid[grid.len() - 1]);
        self.runs[r].pts = out;
    }

    fn sample_closed_by_angle(
        &mut self,
        r: usize,
        clean: &[P3],
        n: usize,
        surfs: &[Arc<Exact>],
        sa: &Arc<Exact>,
        sb: &Arc<Exact>,
    ) -> Option<Vec<usize>> {
        let s = [sa, sb].into_iter().find(|s| s.period_u().is_some())?;
        let per = s.period_u()?;
        let mut uv: Vec<[f64; 2]> = clean.iter().map(|q| s.invert(*q)).collect();
        for k in 1..uv.len() {
            while uv[k][0] - uv[k - 1][0] > per / 2.0 {
                uv[k][0] -= per;
            }
            while uv[k][0] - uv[k - 1][0] < -per / 2.0 {
                uv[k][0] += per;
            }
        }
        let span = uv[uv.len() - 1][0] - uv[0][0];
        if (span.abs() - per).abs() > 0.05 * per {
            return None;
        }
        let dir = span.signum();
        // Even angles from the seam (u = 0), in the curve's direction; v interpolated.
        let at = |u: f64| -> f64 {
            // u within the unwrapped range, by linear search.
            for w in uv.windows(2) {
                let (a, b) = (w[0], w[1]);
                let (lo, hi) = (a[0].min(b[0]), a[0].max(b[0]));
                if u >= lo - 1e-15 && u <= hi + 1e-15 {
                    let t = if (b[0] - a[0]).abs() > 1e-300 {
                        (u - a[0]) / (b[0] - a[0])
                    } else {
                        0.0
                    };
                    return a[1] + t * (b[1] - a[1]);
                }
            }
            uv[0][1]
        };
        let u0 = uv[0][0];
        // The first sample at or after the start in the curve's direction, on the even grid.
        let step = per / n as f64;
        let k0 = if dir > 0.0 {
            (u0 / step).ceil()
        } else {
            (u0 / step).floor()
        };
        let grid0 = self.runs[r].grid[0];
        let mut out = Vec::with_capacity(n + 1);
        for i in 0..n {
            let u = (k0 + dir * i as f64) * step;
            // Keep u inside the unwrapped range.
            let mut uu = u;
            let (lo, hi) = (u0.min(u0 + span), u0.max(u0 + span));
            while uu < lo {
                uu += per;
            }
            while uu > hi {
                uu -= per;
            }
            let guess = s.eval(uu, at(uu));
            let q = snap_or(guess, surfs, self.diag);
            if i == 0 {
                self.pos[grid0] = q;
                out.push(grid0);
            } else {
                self.pos.push(q);
                self.raw.push(q);
                out.push(self.pos.len() - 1);
            }
        }
        out.push(grid0);
        Some(out)
    }

    /// A face's loops as point indices (each loop once around, no repeated end).
    fn loop_points(&self, face: usize) -> Vec<Vec<usize>> {
        self.loops[face]
            .iter()
            .map(|lp| {
                let mut out: Vec<usize> = Vec::new();
                for (r, rev) in lp {
                    let pts = &self.runs[*r].pts;
                    let it: Box<dyn Iterator<Item = &usize>> = if *rev {
                        Box::new(pts.iter().rev())
                    } else {
                        Box::new(pts.iter())
                    };
                    for p in it {
                        if out.last() != Some(p) {
                            out.push(*p);
                        }
                    }
                }
                if out.len() > 1 && out.first() == out.last() {
                    out.pop();
                }
                out
            })
            .collect()
    }

    fn face_on_facets(&self, face: usize, facets: &[Facet]) -> FaceMesh {
        let mut mesh = FaceMesh::default();
        let mut local: HashMap<usize, u32> = HashMap::new();
        let mut id = |mesh: &mut FaceMesh, p: usize| -> u32 {
            *local.entry(p).or_insert_with(|| {
                mesh.positions.push(self.pos[p]);
                (mesh.positions.len() - 1) as u32
            })
        };
        for f in facets.iter().filter(|f| f.face == face) {
            let v = f.v.map(|p| id(&mut mesh, p));
            mesh.triangles.push([v[0], v[1], v[2]]);
            mesh.triangles.push([v[0], v[2], v[3]]);
        }
        mesh
    }

    /// The face on its chart; `None` when its loops do not form a polygon there.
    fn face_on_chart(
        &self,
        face: usize,
        e: &Exact,
        facets: &[Facet],
        tol: f64,
    ) -> Option<FaceMesh> {
        let loops = self.loop_points(face);
        if loops.is_empty() || loops.iter().any(|l| l.len() < 2) {
            return None;
        }
        // Chart points, unwrapped along periodic u.
        let period = e.period_u();
        let mut uv: Vec<[f64; 2]> = Vec::new();
        let mut ids: Vec<usize> = Vec::new();
        let mut poly_loops: Vec<Vec<usize>> = Vec::new();
        let mut windings: Vec<i64> = Vec::new();
        for l in &loops {
            let mut idx = Vec::with_capacity(l.len());
            let mut prev: Option<f64> = None;
            let mut start_u = 0.0;
            for (k, p) in l.iter().enumerate() {
                let mut c = e.invert(self.pos[*p]);
                if let (Some(per), Some(pu)) = (period, prev) {
                    while c[0] - pu > per / 2.0 {
                        c[0] -= per;
                    }
                    while c[0] - pu < -per / 2.0 {
                        c[0] += per;
                    }
                }
                if k == 0 {
                    start_u = c[0];
                }
                prev = Some(c[0]);
                uv.push(c);
                ids.push(*p);
                idx.push(uv.len() - 1);
            }
            let w = match period {
                Some(per) => {
                    let last = prev.unwrap_or(start_u);
                    let mut close = start_u - last;
                    while close > per / 2.0 {
                        close -= per;
                    }
                    while close < -per / 2.0 {
                        close += per;
                    }
                    ((last + close - start_u) / per).round() as i64
                }
                None => 0,
            };
            windings.push(w);
            poly_loops.push(idx);
        }
        let winding: Vec<usize> = (0..loops.len()).filter(|i| windings[*i] != 0).collect();
        let area = |uv: &[[f64; 2]], l: &[usize]| -> f64 {
            let n = l.len();
            (0..n)
                .map(|i| {
                    let (a, b) = (uv[l[i]], uv[l[(i + 1) % n]]);
                    a[0] * b[1] - a[1] * b[0]
                })
                .sum::<f64>()
                / 2.0
        };
        let mut outer_loops: Vec<Vec<usize>>;
        match winding.len() {
            0 => {
                let outer = (0..poly_loops.len())
                    .max_by(|a, b| {
                        area(&uv, &poly_loops[*a])
                            .abs()
                            .total_cmp(&area(&uv, &poly_loops[*b]).abs())
                    })
                    .unwrap_or(0);
                if let Some(per) = period {
                    // Holes next to the outer loop, not a period away.
                    let cu = |l: &[usize], uv: &[[f64; 2]]| {
                        l.iter().map(|i| uv[*i][0]).sum::<f64>() / l.len() as f64
                    };
                    let c0 = cu(&poly_loops[outer], &uv);
                    for (li, l) in poly_loops.iter().enumerate() {
                        if li == outer {
                            continue;
                        }
                        let shift = ((cu(l, &uv) - c0) / per).round() * per;
                        for i in l {
                            uv[*i][0] -= shift;
                        }
                    }
                }
                outer_loops = vec![poly_loops[outer].clone()];
                for (li, l) in poly_loops.iter().enumerate() {
                    if li != outer {
                        outer_loops.push(l.clone());
                    }
                }
            }
            2 => {
                let per = period?;
                let (l1, l2) = (winding[0], winding[1]);
                if windings[l1] + windings[l2] != 0 || windings[l1].abs() != 1 {
                    return None;
                }
                let holes: Vec<usize> = (0..loops.len()).filter(|i| windings[*i] == 0).collect();
                // Cut at the point of loop 1 farthest (in u) from every hole.
                let a = &poly_loops[l1];
                let hole_us: Vec<f64> = holes
                    .iter()
                    .flat_map(|h| poly_loops[*h].iter().map(|i| uv[*i][0]))
                    .collect();
                let gap = |u: f64| {
                    hole_us
                        .iter()
                        .map(|h| {
                            let d = (u - h).rem_euclid(per);
                            d.min(per - d)
                        })
                        .fold(f64::INFINITY, f64::min)
                };
                let i1 = (0..a.len())
                    .max_by(|x, y| gap(uv[a[*x]][0]).total_cmp(&gap(uv[a[*y]][0])))
                    .unwrap_or(0);
                let w1 = windings[l1] as f64;
                let u0 = uv[a[i1]][0];
                // Loop 1 from the cut once around: u from u0 to u0 + w1·per.
                let mut strip: Vec<usize> = Vec::new();
                for s in 0..a.len() {
                    strip.push(a[(i1 + s) % a.len()]);
                }
                // Make the unwrapped sequence continuous from the cut.
                let mut seq: Vec<[f64; 2]> = strip.iter().map(|i| uv[*i]).collect();
                for k in 1..seq.len() {
                    while seq[k][0] - seq[k - 1][0] > per / 2.0 {
                        seq[k][0] -= per;
                    }
                    while seq[k][0] - seq[k - 1][0] < -per / 2.0 {
                        seq[k][0] += per;
                    }
                }
                let mut poly: Vec<usize> = Vec::new();
                let push = |uv: &mut Vec<[f64; 2]>, ids: &mut Vec<usize>, c: [f64; 2], p: usize| {
                    uv.push(c);
                    ids.push(p);
                    uv.len() - 1
                };
                for (k, c) in seq.iter().enumerate() {
                    let pid = ids[strip[k]];
                    let i = push(&mut uv, &mut ids, *c, pid);
                    poly.push(i);
                }
                let end_u = u0 + w1 * per;
                let pid = ids[strip[0]];
                let i = push(&mut uv, &mut ids, [end_u, seq[0][1]], pid);
                poly.push(i);
                // Loop 2 from its point nearest the cut, shifted to start near end_u.
                let b = &poly_loops[l2];
                let i2 = (0..b.len())
                    .min_by(|x, y| {
                        let dx = (uv[b[*x]][0] - u0).rem_euclid(per);
                        let dy = (uv[b[*y]][0] - u0).rem_euclid(per);
                        dx.min(per - dx).total_cmp(&dy.min(per - dy))
                    })
                    .unwrap_or(0);
                let mut seq2: Vec<[f64; 2]> =
                    (0..b.len()).map(|s| uv[b[(i2 + s) % b.len()]]).collect();
                let shift0 = ((end_u - seq2[0][0]) / per).round() * per;
                seq2[0][0] += shift0;
                for k in 1..seq2.len() {
                    while seq2[k][0] - seq2[k - 1][0] > per / 2.0 {
                        seq2[k][0] -= per;
                    }
                    while seq2[k][0] - seq2[k - 1][0] < -per / 2.0 {
                        seq2[k][0] += per;
                    }
                }
                let strip2: Vec<usize> = (0..b.len()).map(|s| b[(i2 + s) % b.len()]).collect();
                for (k, c) in seq2.iter().enumerate() {
                    let pid = ids[strip2[k]];
                    let i = push(&mut uv, &mut ids, *c, pid);
                    poly.push(i);
                }
                let last = [seq2[0][0] - w1 * per, seq2[0][1]];
                let pid = ids[strip2[0]];
                let i = push(&mut uv, &mut ids, last, pid);
                poly.push(i);
                outer_loops = vec![poly];
                let mid = u0 + w1 * per / 2.0;
                for h in holes {
                    let l = poly_loops[h].clone();
                    let c = l.iter().map(|i| uv[*i][0]).sum::<f64>() / l.len() as f64;
                    let shift = ((c - mid) / per).round() * per;
                    for i in &l {
                        uv[*i][0] -= shift;
                    }
                    outer_loops.push(l);
                }
            }
            _ => return None,
        }
        // Counter-clockwise outer loop: mirror u if needed.
        let mirror = area(&uv, &outer_loops[0]) < 0.0;
        if mirror {
            for c in &mut uv {
                c[0] = -c[0];
            }
        }
        for l in &outer_loops[1..] {
            if area(&uv, l) > 0.0 {
                return None;
            }
        }
        // Interior points on curved surfaces, on a triangular lattice in a chart stretched
        // so the lattice step is the curve step along u and the (longer) step along v.
        let mut interior: Vec<usize> = Vec::new();
        let mut vk = 1.0;
        // A ruled face (straight along v) bounded by one polygon needs no interior points:
        // its boundary samples already follow the curve, and the triangles between them run
        // along the rulings.
        let ruled = !e.radii(self.pos[ids[outer_loops[0][0]]])[1].is_finite();
        if e.curved() && !(ruled && outer_loops.len() == 1) {
            let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
            for l in &outer_loops {
                for i in l {
                    for a in 0..2 {
                        lo[a] = lo[a].min(uv[*i][a]);
                        hi[a] = hi[a].max(uv[*i][a]);
                    }
                }
            }
            let probe = self.pos[ids[outer_loops[0][0]]];
            let rr = e.radii(probe);
            let mut ru = rr[0];
            // The radius about the axis varies over a revolved face; use its smallest.
            if let Exact::Revolved { .. } = e {
                for l in &outer_loops {
                    for i in l {
                        ru = ru.min(e.radii(self.pos[ids[*i]])[0]);
                    }
                }
            }
            let metric = e.u_metric(probe).max(1e-12);
            let (uext, vext) = (hi[0] - lo[0], hi[1] - lo[1]);
            let mut hu = self.step_for(ru, tol) / metric;
            let mut hv = if rr[1].is_finite() {
                self.step_for(rr[1], tol)
            } else {
                (10.0 * hu * metric).min(vext / 2.0).max(hu * metric)
            };
            if !hu.is_finite() {
                hu = uext.max(1e-300);
            }
            if !hv.is_finite() {
                hv = vext.max(1e-300);
            }
            // At most about 20 000 lattice points per face.
            let count = (uext / hu) * (vext / hv);
            if count > 20_000.0 {
                let g = (count / 20_000.0).sqrt();
                hu *= g;
                hv *= g;
            }
            vk = hu / hv;
            for c in &mut uv {
                c[1] *= vk;
            }
            let (vlo, vhi) = (lo[1] * vk, hi[1] * vk);
            if uext > 1.5 * hu || vhi - vlo > 1.5 * hu {
                let segs: Vec<([f64; 2], [f64; 2])> = outer_loops
                    .iter()
                    .flat_map(|l| {
                        let n = l.len();
                        (0..n).map(move |k| (l[k], l[(k + 1) % n]))
                    })
                    .map(|(a, b)| (uv[a], uv[b]))
                    .collect();
                let min_h = 0.5 * hu;
                let row = hu * 3f64.sqrt() / 2.0;
                let (ulo, uhi) = if mirror {
                    (-hi[0], -lo[0])
                } else {
                    (lo[0], hi[0])
                };
                let nrow = ((vhi - vlo) / row).floor() as usize;
                let ncol = (uext / hu).ceil() as usize + 1;
                for j in 1..=nrow {
                    let v = vlo + j as f64 * row;
                    if v >= vhi {
                        break;
                    }
                    let shift = if j % 2 == 1 { 0.5 * hu } else { 0.0 };
                    for i in 0..ncol {
                        let u = ulo + shift + i as f64 * hu;
                        if u <= ulo || u >= uhi {
                            continue;
                        }
                        let c = [u, v];
                        if !inside(&segs, c) {
                            continue;
                        }
                        if segs
                            .iter()
                            .any(|(a, b)| seg_dist2(c, *a, *b) < min_h * min_h)
                        {
                            continue;
                        }
                        uv.push(c);
                        ids.push(usize::MAX - 1 - interior.len());
                        interior.push(uv.len() - 1);
                    }
                }
            }
        }
        let tris = cdt::triangulate(&uv, &outer_loops, &interior)?;
        // Outward: the facets' normals decide the chart's sign.
        let mut sign = 0.0;
        for f in facets.iter().filter(|f| f.face == face) {
            let pa = self.raw[f.v[0]];
            let n = cross(sub(self.raw[f.v[1]], pa), sub(self.raw[f.v[3]], pa));
            let c = mul(
                add(
                    add(pa, self.raw[f.v[1]]),
                    add(self.raw[f.v[2]], self.raw[f.v[3]]),
                ),
                0.25,
            );
            let q = e.invert(c);
            sign += dot(n, e.normal(q[0], q[1])).signum();
        }
        let flip_all = sign < 0.0;
        let mut mesh = FaceMesh::default();
        let mut local: HashMap<usize, u32> = HashMap::new();
        let mut id_of = |mesh: &mut FaceMesh, k: usize| -> u32 {
            let key = ids[k];
            *local.entry(key).or_insert_with(|| {
                let p = if key >= self.pos.len() {
                    let c = uv[k];
                    let u = if mirror { -c[0] } else { c[0] };
                    e.eval(u, c[1] / vk)
                } else {
                    self.pos[key]
                };
                mesh.positions.push(p);
                (mesh.positions.len() - 1) as u32
            })
        };
        for t in tris {
            let mut v = t.map(|k| id_of(&mut mesh, k));
            // The chart is counter-clockwise in (±u, v); its 3D sense is ∂u × ∂v (mirrored:
            // reversed), and the facets say which way is out.
            if mirror != flip_all {
                v.swap(1, 2);
            }
            if v[0] == v[1] || v[1] == v[2] || v[0] == v[2] {
                continue;
            }
            mesh.triangles.push(v);
        }
        Some(mesh)
    }

    fn edge_polyline(&self, e: &EdgeTopo) -> Vec<P3> {
        let (a, b) = (e.faces[0] as usize, e.faces[1] as usize);
        let key = (a.min(b), a.max(b));
        let cands: Vec<&Run> = self.runs.iter().filter(|r| r.faces == key).collect();
        let closed = len(sub(e.geom.end, e.geom.start)) < 1e-12;
        let pick = cands.iter().find(|r| {
            let (s, t) = (self.raw[r.grid[0]], self.raw[r.grid[r.grid.len() - 1]]);
            if closed {
                r.grid.first() == r.grid.last()
            } else {
                (len(sub(s, e.geom.start)) < 1e-9 && len(sub(t, e.geom.end)) < 1e-9)
                    || (len(sub(s, e.geom.end)) < 1e-9 && len(sub(t, e.geom.start)) < 1e-9)
            }
        });
        match pick.or(if cands.len() == 1 {
            cands.first()
        } else {
            None
        }) {
            Some(r) => r.pts.iter().map(|p| self.pos[*p]).collect(),
            None => {
                if closed {
                    Vec::new()
                } else {
                    vec![e.geom.start, e.geom.end]
                }
            }
        }
    }
}

fn seg_dist2(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let l2 = d[0] * d[0] + d[1] * d[1];
    let t = if l2 > 0.0 {
        (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let q = [a[0] + t * d[0] - p[0], a[1] + t * d[1] - p[1]];
    q[0] * q[0] + q[1] * q[1]
}

fn inside(segs: &[([f64; 2], [f64; 2])], q: [f64; 2]) -> bool {
    let mut c = false;
    for (a, b) in segs {
        if (a[1] > q[1]) != (b[1] > q[1]) {
            let x = a[0] + (q[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
            if q[0] < x {
                c = !c;
            }
        }
    }
    c
}

/// Points of a snapped staircase without repeats or backtracks.
fn clean_polyline(pts: &[P3]) -> Vec<P3> {
    let total: f64 = pts.windows(2).map(|w| len(sub(w[1], w[0]))).sum();
    let eps = 1e-6 * total.max(1e-300);
    let mut out: Vec<P3> = Vec::with_capacity(pts.len());
    for (i, p) in pts.iter().enumerate() {
        let last = i == pts.len() - 1;
        if let Some(q) = out.last() {
            if len(sub(*p, *q)) < eps && !last {
                continue;
            }
            if last && len(sub(*p, *q)) < eps && out.len() > 1 {
                out.pop();
            }
        }
        out.push(*p);
    }
    // Drop points that turn back on the curve.
    let mut changed = true;
    let mut guard = 0;
    while changed && guard < 8 {
        changed = false;
        guard += 1;
        let mut k = 1;
        while k + 1 < out.len() {
            let (a, b, c) = (out[k - 1], out[k], out[k + 1]);
            if dot(sub(b, a), sub(c, b)) < 0.0 {
                out.remove(k);
                changed = true;
            } else {
                k += 1;
            }
        }
    }
    out
}

mod blends {
    //! Convex fillets and chamfers on the display mesh.
    //!
    //! The grid keeps a blend as a decoration over a sharp edge between two grid planes. On
    //! the display it becomes the face a B-rep kernel would build: the two planes are trimmed
    //! back to the tangent lines, the end faces gain the blend's end arc in place of the sharp
    //! corner, and the blend draws as a strip between its two end arcs.
    use super::*;

    /// Walked points of a loop entry.
    fn walked(ctx: &Ctx<'_>, (r, rev): (usize, bool)) -> Vec<usize> {
        let p = &ctx.runs[r].pts;
        if rev {
            p.iter().rev().copied().collect()
        } else {
            p.clone()
        }
    }

    /// Trims the faces each convex blend meets and returns the blends' own meshes, or `None`
    /// when a blend lies where this mesher cannot place it (the caller then draws facets).
    pub(super) fn apply(ctx: &mut Ctx<'_>) -> Option<BTreeMap<usize, FaceMesh>> {
        let mut out = BTreeMap::new();
        let s = ctx.s;
        let mut used_corners: BTreeSet<usize> = BTreeSet::new();
        for dec in &s.decorations {
            if dec.size <= EPS || dec.pieces.is_empty() {
                continue;
            }
            let normals = dec.planes.map(|p| (p.axis, p.dir as f64));
            let [(a0, s0), (a1, s1)] = normals;
            let mut mid = dec.at;
            mid[dec.along] = 0.5 * (dec.lo + dec.hi);
            let probe = |d0: f64, d1: f64| {
                let mut q = mid;
                let e = 1e-3 * dec.size;
                q[a0] += d0 * s0 * e;
                q[a1] += d1 * s1 * e;
                s.contains(q)
            };
            let convex = probe(-1.0, -1.0) && !probe(-1.0, 1.0) && !probe(1.0, -1.0);
            if !convex {
                // A concave blend keeps the sharp edge on the display.
                continue;
            }
            if dec.curved || dec.pieces.len() != 1 {
                return None;
            }
            let (plo, phi, bface) = dec.pieces[0];
            let bface = bface as usize;
            let point = |theta: f64, along: f64| -> P3 {
                let size = dec.size;
                let mut p = dec.at;
                p[dec.along] = along;
                p[a0] -= s0 * size;
                p[a1] -= s1 * size;
                let (c, sn) = if dec.role == Role::Chamfer {
                    let t = theta / FRAC_PI_2;
                    (1.0 - t, t)
                } else {
                    (theta.cos(), theta.sin())
                };
                p[a0] += s0 * size * c;
                p[a1] += s1 * size * sn;
                p
            };
            // The run of the sharp edge, between the faces on the two planes.
            let on_plane = |f: usize, k: usize| {
                s.faces[f]
                    .plane
                    .map(|p| p.same(&dec.planes[k]))
                    .unwrap_or(false)
            };
            let along = dec.along;
            let ridx = (0..ctx.runs.len()).find(|r| {
                let run = &ctx.runs[*r];
                let (fa, fb) = run.faces;
                let pair =
                    (on_plane(fa, 0) && on_plane(fb, 1)) || (on_plane(fa, 1) && on_plane(fb, 0));
                if !pair || run.grid.len() < 2 {
                    return false;
                }
                let (x, y) = (ctx.raw[run.grid[0]], ctx.raw[run.grid[run.grid.len() - 1]]);
                let line = (0..3)
                    .filter(|a| *a != along)
                    .all(|a| (x[a] - dec.at[a]).abs() < EPS && (y[a] - dec.at[a]).abs() < EPS);
                let (l, h) = (x[along].min(y[along]), x[along].max(y[along]));
                line && (l - plo).abs() < EPS && (h - phi).abs() < EPS
            })?;
            let (fa, fb) = ctx.runs[ridx].faces;
            let (f0, f1) = if on_plane(fa, 0) { (fa, fb) } else { (fb, fa) };
            let grid = ctx.runs[ridx].grid.clone();
            let ends = [grid[0], grid[grid.len() - 1]];
            if ends.iter().any(|c| !used_corners.insert(*c)) {
                return None;
            }
            // The end arcs, θ from face 0 to face 1.
            let n = if dec.role == Role::Chamfer {
                1
            } else {
                ((FRAC_PI_2 / ctx.dtheta).ceil() as usize).max(2)
            };
            let mut arcs: Vec<(usize, Vec<usize>)> = Vec::new();
            for c in ends {
                let at = ctx.raw[c][along];
                let ids: Vec<usize> = (0..=n)
                    .map(|i| {
                        ctx.pos.push(point(FRAC_PI_2 * i as f64 / n as f64, at));
                        ctx.raw.push(ctx.pos[ctx.pos.len() - 1]);
                        ctx.pos.len() - 1
                    })
                    .collect();
                arcs.push((c, ids));
            }
            // The end face at each corner, and the runs it meets the planes on.
            for (c, ids) in &arcs {
                let faces = ctx.corner_faces.get(c)?.clone();
                let others: Vec<usize> = faces
                    .iter()
                    .copied()
                    .filter(|f| *f != f0 && *f != f1)
                    .collect();
                if others.len() != 1 {
                    return None;
                }
                let e = others[0];
                let ep = s.faces[e].plane?;
                if ep.axis != along {
                    return None;
                }
                // Trim the end runs: (e, f0) ends at the tangent point on face 0 and (e, f1)
                // at the one on face 1.
                for (fk, tangent) in [(f0, ids[0]), (f1, ids[n])] {
                    let key = (e.min(fk), e.max(fk));
                    let r = (0..ctx.runs.len()).find(|r| {
                        let run = &ctx.runs[*r];
                        run.faces == key
                            && (run.pts.first() == Some(c) || run.pts.last() == Some(c))
                    })?;
                    let pts = &mut ctx.runs[r].pts;
                    let at_start = pts.first() == Some(c);
                    if !at_start {
                        pts.reverse();
                    }
                    // Drop points the blend covers, then start at the tangent point.
                    let t = ctx.pos[tangent];
                    let cpos = ctx.pos[*c];
                    let reach = len(sub(t, cpos));
                    let keep: Vec<usize> = pts
                        .iter()
                        .copied()
                        .skip(1)
                        .filter(|p| len(sub(ctx.pos[*p], cpos)) > reach + EPS)
                        .collect();
                    if keep.is_empty() {
                        return None;
                    }
                    let mut np = vec![tangent];
                    np.extend(keep);
                    if !at_start {
                        np.reverse();
                    }
                    *pts = np;
                }
                // The end face's loop: insert the arc where the corner was.
                let arc_run = {
                    ctx.runs.push(Run {
                        faces: (e.min(bface), e.max(bface)),
                        grid: vec![*c],
                        pts: Vec::new(),
                    });
                    ctx.runs.len() - 1
                };
                let mut placed = false;
                for li in 0..ctx.loops[e].len() {
                    let lp = ctx.loops[e][li].clone();
                    let m = lp.len();
                    for k in 0..m {
                        let w = walked(ctx, lp[k]);
                        let w2 = walked(ctx, lp[(k + 1) % m]);
                        let (Some(last), Some(first)) = (w.last(), w2.first()) else {
                            continue;
                        };
                        let a = *last;
                        let b = *first;
                        if a == b {
                            continue;
                        }
                        let ok = (a == ids[0] && b == ids[n]) || (a == ids[n] && b == ids[0]);
                        if !ok {
                            continue;
                        }
                        let seq: Vec<usize> = if a == ids[0] {
                            ids.clone()
                        } else {
                            ids.iter().rev().copied().collect()
                        };
                        // Stored in e's walking direction (e is the run's lower face).
                        ctx.runs[arc_run].pts = seq;
                        ctx.loops[e][li].insert(k + 1, (arc_run, false));
                        placed = true;
                        break;
                    }
                    if placed {
                        break;
                    }
                }
                if !placed {
                    return None;
                }
            }
            // The sharp edge becomes two tangent lines.
            let tangent_of = |ctx: &Ctx<'_>, p: usize, face0: bool| -> usize {
                let at = ctx.raw[p][along];
                let (_, ids) = arcs
                    .iter()
                    .find(|(c, _)| (ctx.raw[*c][along] - at).abs() < EPS)
                    .expect("an end arc at each end");
                if face0 {
                    ids[0]
                } else {
                    ids[n]
                }
            };
            for (fk, face0) in [(f0, true), (f1, false)] {
                for li in 0..ctx.loops[fk].len() {
                    for k in 0..ctx.loops[fk][li].len() {
                        let (r, rev) = ctx.loops[fk][li][k];
                        if r != ridx {
                            continue;
                        }
                        let w = walked(ctx, (r, rev));
                        let pts = vec![
                            tangent_of(ctx, w[0], face0),
                            tangent_of(ctx, w[w.len() - 1], face0),
                        ];
                        ctx.runs.push(Run {
                            faces: (fk.min(bface), fk.max(bface)),
                            grid: w.clone(),
                            pts,
                        });
                        let nr = ctx.runs.len() - 1;
                        ctx.loops[fk][li][k] = (nr, false);
                    }
                }
            }
            // The blend's own strip between its end arcs.
            let (lo_ids, hi_ids) = if ctx.raw[arcs[0].0][along] <= ctx.raw[arcs[1].0][along] {
                (&arcs[0].1, &arcs[1].1)
            } else {
                (&arcs[1].1, &arcs[0].1)
            };
            let mut mesh = FaceMesh::default();
            for i in 0..=n {
                mesh.positions.push(ctx.pos[lo_ids[i]]);
                mesh.positions.push(ctx.pos[hi_ids[i]]);
            }
            let [p0, p1, p2] = [0, 1, 2].map(|i| mesh.positions[i]);
            let nrm = cross(sub(p1, p0), sub(p2, p0));
            let mut outv = [0.0; 3];
            outv[a0] += s0;
            outv[a1] += s1;
            let flip = dot(nrm, outv) < 0.0;
            for i in 0..n as u32 {
                let (a, b, c, d) = (2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3);
                if flip {
                    mesh.triangles.extend([[a, c, b], [b, c, d]]);
                } else {
                    mesh.triangles.extend([[a, b, c], [b, d, c]]);
                }
            }
            out.insert(bface, mesh);
        }
        Some(out)
    }
}
