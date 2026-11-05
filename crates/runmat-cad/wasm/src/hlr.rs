//! Hidden-line removal for drawing views: every edge of the drawn model, projected under the
//! view's orientation and split into visible and hidden runs, as ISO 128 and ASME Y14.3 draw
//! them (visible solid, hidden dashed).
//!
//! - **Edges.** Each topological edge of the display. An edge the tessellator left without a
//!   polyline (a circle on the reference kernel) is rebuilt from the boundary segments its two
//!   faces' meshes share.
//! - **Silhouettes.** A curved face adds the mesh edges where it turns from facing the viewer to
//!   facing away (a hole's walls in a side view).
//! - **Visibility.** Each edge is sampled along its length; a sample is hidden when a ray from
//!   it toward the viewer meets a triangle of any other face. Triangles are bucketed on a grid
//!   in view space, so a sample tests only the triangles over it.
//!
//! Output lines are flat `[u, v, u, v, …]` in view millimetres (model millimetres projected).

use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

/// A view's projection: screen right `u`, screen up `v`, and `depth` into the screen.
#[derive(Clone, Copy, Debug)]
pub struct Axes {
    pub u: [f64; 3],
    pub v: [f64; 3],
    pub depth: [f64; 3],
}

/// The orientations the sheet canvas draws, third-angle.
pub fn axes(orientation: &str) -> Axes {
    let a = |u, v, depth| Axes { u, v, depth };
    match orientation {
        "back" => a([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
        "top" => a([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
        "bottom" => a([1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
        "right" => a([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]),
        "left" => a([0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        "isometric" => a(
            [
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
                0.0,
            ],
            [-0.4082, 0.4082, 0.8165],
            [0.5774, -0.5774, 0.5774],
        ),
        _ => a([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
    }
}

/// A view's orientation from the drawing's views (as JSON): a standard view's own, a projected
/// view's direction from its parent, a section or detail its parent's.
pub fn orientation<'a>(views: &'a Value, name: &str, depth: usize) -> &'a str {
    let Some(view) = views
        .as_array()
        .and_then(|v| v.iter().find(|v| v["name"] == name))
    else {
        return "front";
    };
    let kind = &view["kind"];
    match kind["view"].as_str() {
        Some("standard") => kind["orientation"].as_str().unwrap_or("front"),
        Some("projected") => match kind["direction"].as_str() {
            Some("up") => "top",
            Some("down") => "bottom",
            Some("left") => "left",
            Some("right") => "right",
            _ => "front",
        },
        _ if depth < 8 => kind["from"]
            .as_str()
            .map(|from| orientation(views, from, depth + 1))
            .unwrap_or("front"),
        _ => "front",
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ViewLines {
    pub visible: Vec<Vec<f64>>,
    pub hidden: Vec<Vec<f64>>,
}

type P3 = [f64; 3];
/// A mesh edge by its two end keys (smaller first).
type EdgeKey = ([i64; 3], [i64; 3]);

fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// One face's triangles in model millimetres.
struct Face {
    tris: Vec<[P3; 3]>,
}

/// One edge: its polyline (model mm) and the faces it bounds (global face indices).
struct Edge {
    points: Vec<P3>,
    faces: [usize; 2],
}

struct Model {
    faces: Vec<Face>,
    edges: Vec<Edge>,
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default()
}

fn transform(t: &[f64], p: P3) -> P3 {
    if t.len() < 12 {
        return p;
    }
    [
        t[0] * p[0] + t[1] * p[1] + t[2] * p[2] + t[9],
        t[3] * p[0] + t[4] * p[1] + t[5] * p[2] + t[10],
        t[6] * p[0] + t[7] * p[1] + t[8] * p[2] + t[11],
    ]
}

fn key(p: P3) -> [i64; 3] {
    p.map(|x| (x * 1.0e4).round() as i64)
}

/// The segments of a face's mesh boundary (edges used by one triangle), by their end keys.
fn boundary(face: &Face) -> HashMap<EdgeKey, (P3, P3)> {
    let mut count: HashMap<EdgeKey, (usize, P3, P3)> = HashMap::new();
    for t in &face.tris {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let (ka, kb) = (key(a), key(b));
            let id = if ka <= kb { (ka, kb) } else { (kb, ka) };
            count.entry(id).or_insert((0, a, b)).0 += 1;
        }
    }
    count
        .into_iter()
        .filter(|(_, (n, _, _))| *n == 1)
        .map(|(k, (_, a, b))| (k, (a, b)))
        .collect()
}

fn add_bodies(model: &mut Model, bodies: &Value, t: &[f64]) {
    for body in bodies.as_array().into_iter().flatten() {
        let base = model.faces.len();
        let faces = body["faces"].as_array().cloned().unwrap_or_default();
        for f in &faces {
            let pos = floats(&f["positions"]);
            let at = |i: usize| -> P3 {
                transform(t, [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]]).map(|x| x * 1000.0)
            };
            let idx: Vec<usize> = floats(&f["triangles"])
                .iter()
                .map(|x| *x as usize)
                .collect();
            let tris = idx
                .chunks_exact(3)
                .filter(|c| c.iter().all(|i| 3 * i + 2 < pos.len()))
                .map(|c| [at(c[0]), at(c[1]), at(c[2])])
                .collect();
            model.faces.push(Face { tris });
        }
        for e in body["edges"].as_array().into_iter().flatten() {
            let fa = e["faces"][0].as_u64().unwrap_or(0) as usize + base;
            let fb = e["faces"][1].as_u64().unwrap_or(0) as usize + base;
            let pts = floats(&e["points"]);
            let points: Vec<P3> = pts
                .chunks_exact(3)
                .map(|c| transform(t, [c[0], c[1], c[2]]).map(|x| x * 1000.0))
                .collect();
            if points.len() >= 2 {
                model.edges.push(Edge {
                    points,
                    faces: [fa, fb],
                });
                continue;
            }
            // No polyline: the boundary segments both faces' meshes share.
            let (Some(a), Some(b)) = (model.faces.get(fa), model.faces.get(fb)) else {
                continue;
            };
            let (ba, bb) = (boundary(a), boundary(b));
            for (k, (p, q)) in &ba {
                if bb.contains_key(k) {
                    model.edges.push(Edge {
                        points: vec![*p, *q],
                        faces: [fa, fb],
                    });
                }
            }
        }
    }
}

fn model_of(display: &Value) -> Model {
    let mut model = Model {
        faces: Vec::new(),
        edges: Vec::new(),
    };
    let data = &display["data"];
    if display["kind"] == "assembly" {
        let prototypes = data["prototypes"].as_array().cloned().unwrap_or_default();
        for occ in data["occurrences"].as_array().into_iter().flatten() {
            let t = floats(&occ["transform"]);
            let Some(proto) = occ["prototype"]
                .as_u64()
                .and_then(|i| prototypes.get(i as usize))
            else {
                continue;
            };
            match proto["shape"].as_str() {
                Some("mesh") => add_bodies(&mut model, &proto["part"]["bodies"], &t),
                Some("box") => {
                    let (lo, hi) = (
                        floats(&proto["bounds"]["min"]),
                        floats(&proto["bounds"]["max"]),
                    );
                    if lo.len() == 3 && hi.len() == 3 {
                        let c = |k: usize| -> P3 {
                            let p =
                                [0, 1, 2].map(|a| if (k >> a) & 1 == 1 { hi[a] } else { lo[a] });
                            transform(&t, p).map(|x| x * 1000.0)
                        };
                        let face = model.faces.len();
                        model.faces.push(Face { tris: Vec::new() });
                        for (i, j) in [
                            (0, 1),
                            (2, 3),
                            (4, 5),
                            (6, 7),
                            (0, 2),
                            (1, 3),
                            (4, 6),
                            (5, 7),
                            (0, 4),
                            (1, 5),
                            (2, 6),
                            (3, 7),
                        ] {
                            model.edges.push(Edge {
                                points: vec![c(i), c(j)],
                                faces: [face, face],
                            });
                        }
                    }
                }
                _ => {}
            }
        }
    } else {
        add_bodies(&mut model, &data["bodies"], &[]);
    }
    model
}

/// The silhouette of each curved face under `depth`: mesh edges inside the face whose two
/// triangles face opposite ways.
fn silhouettes(model: &Model, depth: P3) -> Vec<Edge> {
    let mut out = Vec::new();
    for (fi, face) in model.faces.iter().enumerate() {
        let mut by_edge: HashMap<EdgeKey, Vec<(f64, P3, P3)>> = HashMap::new();
        for t in &face.tris {
            let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
            let facing = dot(n, depth);
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                let (ka, kb) = (key(a), key(b));
                let id = if ka <= kb { (ka, kb) } else { (kb, ka) };
                by_edge.entry(id).or_default().push((facing, a, b));
            }
        }
        for uses in by_edge.values() {
            if let [(f1, a, b), (f2, _, _)] = uses.as_slice() {
                let scale = 1e-9;
                if (f1 > &scale && f2 < &-scale) || (f1 < &-scale && f2 > &scale) {
                    out.push(Edge {
                        points: vec![*a, *b],
                        faces: [fi, fi],
                    });
                }
            }
        }
    }
    out
}

/// Triangles bucketed on a grid in view space.
struct Grid {
    lo: [f64; 2],
    cell: f64,
    n: usize,
    cells: Vec<Vec<(usize, usize)>>,
}

impl Grid {
    fn new(model: &Model, axes: &Axes) -> Grid {
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for f in &model.faces {
            for t in &f.tris {
                for p in t {
                    let q = [dot(*p, axes.u), dot(*p, axes.v)];
                    for k in 0..2 {
                        lo[k] = lo[k].min(q[k]);
                        hi[k] = hi[k].max(q[k]);
                    }
                }
            }
        }
        let n = 64;
        let cell = ((hi[0] - lo[0]).max(hi[1] - lo[1]) / n as f64).max(1e-6);
        let mut grid = Grid {
            lo,
            cell,
            n,
            cells: vec![Vec::new(); n * n],
        };
        for (fi, f) in model.faces.iter().enumerate() {
            for (ti, t) in f.tris.iter().enumerate() {
                let qs = t.map(|p| [dot(p, axes.u), dot(p, axes.v)]);
                let (x0, x1) = grid.span(qs.iter().map(|q| q[0]), 0);
                let (y0, y1) = grid.span(qs.iter().map(|q| q[1]), 1);
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        grid.cells[y * n + x].push((fi, ti));
                    }
                }
            }
        }
        grid
    }
    fn index(&self, v: f64, axis: usize) -> usize {
        (((v - self.lo[axis]) / self.cell).floor().max(0.0) as usize).min(self.n - 1)
    }
    fn span(&self, vs: impl Iterator<Item = f64>, axis: usize) -> (usize, usize) {
        let (mut a, mut b) = (f64::MAX, f64::MIN);
        for v in vs {
            a = a.min(v);
            b = b.max(v);
        }
        (self.index(a, axis), self.index(b, axis))
    }
}

/// Whether the ray from `p` toward the viewer (against `depth`) meets a triangle of a face
/// other than `own` before it leaves the model.
fn hidden(model: &Model, grid: &Grid, axes: &Axes, p: P3, own: [usize; 2]) -> bool {
    let toward = axes.depth.map(|x| -x);
    let cell = grid.index(dot(p, axes.v), 1) * grid.n + grid.index(dot(p, axes.u), 0);
    let tol = 1e-4 * grid.cell * grid.n as f64;
    grid.cells[cell].iter().any(|&(fi, ti)| {
        if own.contains(&fi) {
            return false;
        }
        let [a, b, c] = model.faces[fi].tris[ti];
        // Möller–Trumbore.
        let (e1, e2) = (sub(b, a), sub(c, a));
        let h = cross(toward, e2);
        let det = dot(e1, h);
        if det.abs() < 1e-12 {
            return false;
        }
        let s = sub(p, a);
        let u = dot(s, h) / det;
        let q = cross(s, e1);
        let v = dot(toward, q) / det;
        // Inclusive: a sample over a shared triangle edge or vertex is covered by its faces.
        let edge = 1e-7;
        if u < -edge || v < -edge || u + v > 1.0 + edge {
            return false;
        }
        dot(e2, q) / det > tol
    })
}

/// Every edge of the drawn model under `axes`, split into visible and hidden runs.
pub fn view_lines(display: &Value, axes: &Axes) -> ViewLines {
    let model = model_of(display);
    let grid = Grid::new(&model, axes);
    let mut out = ViewLines::default();
    let project = |p: P3| [dot(p, axes.u), dot(p, axes.v)];
    let mut edges: Vec<&Edge> = model.edges.iter().collect();
    let sil = silhouettes(&model, axes.depth);
    edges.extend(sil.iter());
    // A segment seen end-on draws as a point; leave it out.
    let mut seen: BTreeMap<[i64; 4], bool> = BTreeMap::new();
    for edge in edges {
        for w in edge.points.windows(2) {
            let (a, b) = (w[0], w[1]);
            let (pa, pb) = (project(a), project(b));
            let len = ((pb[0] - pa[0]).powi(2) + (pb[1] - pa[1]).powi(2)).sqrt();
            if len < 1e-6 {
                continue;
            }
            let k = {
                let r = |x: f64| (x * 1.0e3).round() as i64;
                let (s, e) = if (r(pa[0]), r(pa[1])) <= (r(pb[0]), r(pb[1])) {
                    (pa, pb)
                } else {
                    (pb, pa)
                };
                [r(s[0]), r(s[1]), r(e[0]), r(e[1])]
            };
            let steps = ((len / 0.5).ceil() as usize).clamp(1, 128);
            // Classify each piece by its midpoint, then join runs.
            let mut run: Option<(bool, Vec<f64>)> = None;
            let lerp = |t: f64| -> P3 { [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t) };
            for s in 0..steps {
                let (t0, t1) = (s as f64 / steps as f64, (s + 1) as f64 / steps as f64);
                let h = hidden(&model, &grid, axes, lerp((t0 + t1) / 2.0), edge.faces);
                let (p0, p1) = (project(lerp(t0)), project(lerp(t1)));
                match &mut run {
                    Some((state, pts)) if *state == h => pts.extend([p1[0], p1[1]]),
                    _ => {
                        if let Some((state, pts)) = run.take() {
                            push(&mut out, state, pts);
                        }
                        run = Some((h, vec![p0[0], p0[1], p1[0], p1[1]]));
                    }
                }
            }
            if let Some((state, pts)) = run.take() {
                // One projected segment drawn twice (a through-cut's top and bottom edges, a
                // hole's two rims) draws once: a copy adds nothing unless it is the first
                // visible one (a hidden line under it is dropped by `merge_overlaps`).
                let visible = !state;
                match seen.get(&k) {
                    Some(true) => continue,
                    Some(false) if !visible => continue,
                    _ => {}
                }
                seen.insert(k, visible);
                push(&mut out, state, pts);
            }
        }
    }
    merge_overlaps(&mut out);
    out
}

fn push(out: &mut ViewLines, hidden: bool, pts: Vec<f64>) {
    if hidden {
        out.hidden.push(pts)
    } else {
        out.visible.push(pts)
    }
}

/// Hidden runs that lie on a visible line are drawn by the visible line.
fn merge_overlaps(out: &mut ViewLines) {
    let on_visible = |x: f64, y: f64, visible: &[Vec<f64>]| {
        visible.iter().any(|l| {
            l.chunks_exact(2).collect::<Vec<_>>().windows(2).any(|w| {
                let (a, b) = ((w[0][0], w[0][1]), (w[1][0], w[1][1]));
                let (dx, dy) = (b.0 - a.0, b.1 - a.1);
                let len2 = dx * dx + dy * dy;
                if len2 < 1e-12 {
                    return false;
                }
                let t = (((x - a.0) * dx + (y - a.1) * dy) / len2).clamp(0.0, 1.0);
                let (px, py) = (a.0 + t * dx, a.1 + t * dy);
                (px - x).powi(2) + (py - y).powi(2) < 1e-6
            })
        })
    };
    let visible = out.visible.clone();
    out.hidden.retain(|l| {
        let n = l.len() / 2;
        let mid = |i: usize| (l[2 * i], l[2 * i + 1]);
        let (a, b) = (mid(0), mid(n - 1));
        let c = ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
        !(on_visible(a.0, a.1, &visible)
            && on_visible(b.0, b.1, &visible)
            && on_visible(c.0, c.1, &visible))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Desktop's showcase bracket: an 80 x 50 x 10 plate, a 20 x 10 through-cut at
    /// (20..40, 20..30) and a 4 mm through-hole at (60, 10).
    pub(super) const BRACKET: &str = include_str!(
        "../../../../../runmat-private-cad/desktop/src/app/components/cad/fixtures/bracket.part"
    );

    fn lines(orientation: &str) -> ViewLines {
        let part = crate::runtime::compile_part("/p/bracket.part", BRACKET)
            .unwrap_or_else(|d| panic!("{d:?}"));
        let display = serde_json::json!({ "kind": "part", "data": part.display });
        view_lines(&display, &axes(orientation))
    }

    /// Segments of `lines` (flat u, v pairs) as (u0, v0, u1, v1).
    fn segments(lines: &[Vec<f64>]) -> Vec<[f64; 4]> {
        lines
            .iter()
            .flat_map(|l| {
                l.chunks_exact(2)
                    .collect::<Vec<_>>()
                    .windows(2)
                    .map(|w| [w[0][0], w[0][1], w[1][0], w[1][1]])
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < 0.05
    }

    /// Length of the segments at constant `u = at` (vertical) within `v` in [lo, hi].
    fn vertical(segs: &[[f64; 4]], at: f64, lo: f64, hi: f64) -> f64 {
        segs.iter()
            .filter(|s| near(s[0], at) && near(s[2], at))
            .map(|s| (s[1].max(s[3]).min(hi) - s[1].min(s[3]).max(lo)).max(0.0))
            .sum()
    }

    #[test]
    fn the_top_view_shows_the_cut_and_the_hole() {
        let top = lines("top");
        let visible = segments(&top.visible);
        // The cut's rectangle: four sides, 20 + 10 + 20 + 10 mm.
        let cut: f64 = visible
            .iter()
            .filter(|s| {
                (20.0 - 0.05..=40.05).contains(&s[0])
                    && (20.0 - 0.05..=30.05).contains(&s[1])
                    && (20.0 - 0.05..=40.05).contains(&s[2])
                    && (20.0 - 0.05..=30.05).contains(&s[3])
            })
            .map(|s| ((s[2] - s[0]).powi(2) + (s[3] - s[1]).powi(2)).sqrt())
            .sum();
        assert!((cut - 60.0).abs() < 0.5, "cut outline {cut} mm of 60");
        // The hole: a closed outline of radius 2 about (60, 10), its circumference.
        let hole: Vec<&[f64; 4]> = visible
            .iter()
            .filter(|s| {
                let r = |u: f64, v: f64| ((u - 60.0).powi(2) + (v - 10.0).powi(2)).sqrt();
                (r(s[0], s[1]) - 2.0).abs() < 0.1 && (r(s[2], s[3]) - 2.0).abs() < 0.1
            })
            .collect();
        let around: f64 = hole
            .iter()
            .map(|s| ((s[2] - s[0]).powi(2) + (s[3] - s[1]).powi(2)).sqrt())
            .sum();
        assert!(hole.len() >= 8, "{} hole segments", hole.len());
        assert!(
            (around - 4.0 * std::f64::consts::PI).abs() < 1.0,
            "hole outline {around} mm"
        );
    }

    #[test]
    fn the_front_and_right_views_show_the_cut_and_the_hole_hidden() {
        let front = lines("front");
        let hidden = segments(&front.hidden);
        // The cut's walls at x = 20 and 40 and the hole's silhouette at x = 58 and 62, through
        // the 10 mm plate, dashed.
        for x in [20.0, 40.0, 58.0, 62.0] {
            let len = vertical(&hidden, x, 0.0, 10.0);
            assert!(len > 9.0, "front: hidden line at x = {x} is {len} mm");
        }
        assert!(
            vertical(&segments(&front.visible), 20.0, 0.0, 10.0) < 0.5,
            "the cut is behind the front face"
        );
        let right = lines("right");
        let hidden = segments(&right.hidden);
        // Seen from the right, u is y: the cut at y = 20 and 30, the hole at y = 8 and 12.
        for y in [20.0, 30.0, 8.0, 12.0] {
            let len = vertical(&hidden, y, 0.0, 10.0);
            assert!(len > 9.0, "right: hidden line at y = {y} is {len} mm");
        }
    }
}
