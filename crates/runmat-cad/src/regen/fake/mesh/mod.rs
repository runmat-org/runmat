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

mod sampling;
mod faces;

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

impl Ctx<'_> {}

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

mod blends;
