//! Constrained Delaunay triangulation of a polygon with holes, for the display mesher.
//!
//! The polygon (an outer loop counter-clockwise, holes clockwise) is first ear clipped, with
//! each hole bridged to the outer boundary (Eberly's method, holes in order of their rightmost
//! point). Interior points are then inserted one at a time, and every edge that is not a
//! boundary edge is flipped until each triangle's circumcircle is empty of its neighbours'
//! opposite points (Lawson). The result is the constrained Delaunay triangulation of the
//! boundary and the interior points: no boundary edge is crossed and, among triangulations
//! that keep the boundary, the smallest angle is as large as it can be.
//!
//! Coordinates are normalised to the unit box first, so the tolerances are scale free.

use std::collections::{BTreeSet, HashMap};

type P2 = [f64; 2];

fn orient(a: P2, b: P2, c: P2) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Whether `d` lies strictly inside the circumcircle of counter-clockwise `a, b, c`.
fn in_circle(a: P2, b: P2, c: P2, d: P2) -> bool {
    let (adx, ady) = (a[0] - d[0], a[1] - d[1]);
    let (bdx, bdy) = (b[0] - d[0], b[1] - d[1]);
    let (cdx, cdy) = (c[0] - d[0], c[1] - d[1]);
    let det = (adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
        - (bdx * bdx + bdy * bdy) * (adx * cdy - cdx * ady)
        + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady);
    det > 1e-14
}

const EPS: f64 = 1e-12;

fn in_triangle(a: P2, b: P2, c: P2, p: P2) -> bool {
    orient(a, b, p) >= -EPS && orient(b, c, p) >= -EPS && orient(c, a, p) >= -EPS
}

/// One simple polygon (with repeated indices at bridges) from an outer loop and its holes.
fn bridge(pts: &[P2], outer: &[usize], holes: &[Vec<usize>]) -> Option<Vec<usize>> {
    let mut poly: Vec<usize> = outer.to_vec();
    let mut order: Vec<&Vec<usize>> = holes.iter().filter(|h| h.len() >= 3).collect();
    let max_x = |h: &Vec<usize>| {
        h.iter()
            .map(|&i| pts[i][0])
            .fold(f64::NEG_INFINITY, f64::max)
    };
    order.sort_by(|a, b| max_x(b).total_cmp(&max_x(a)));
    for hole in order {
        let (mi, &m) = hole
            .iter()
            .enumerate()
            .max_by(|a, b| {
                pts[*a.1][0]
                    .total_cmp(&pts[*b.1][0])
                    .then(pts[*b.1][1].total_cmp(&pts[*a.1][1]))
            })
            .expect("hole has points");
        let mp = pts[m];
        // The nearest edge the ray from M towards +x hits.
        let n = poly.len();
        let mut best: Option<(f64, usize)> = None;
        for k in 0..n {
            let (a, b) = (pts[poly[k]], pts[poly[(k + 1) % n]]);
            if (a[1] > mp[1]) == (b[1] > mp[1]) && a[1] != mp[1] && b[1] != mp[1] {
                continue;
            }
            if (a[1] - b[1]).abs() < EPS {
                // A horizontal edge on the ray: its nearer end.
                if (a[1] - mp[1]).abs() > EPS {
                    continue;
                }
                for (x, kk) in [(a[0], k), (b[0], (k + 1) % n)] {
                    if x >= mp[0] - EPS && best.map(|bb| x - mp[0] < bb.0).unwrap_or(true) {
                        best = Some((x - mp[0], kk + n));
                    }
                }
                continue;
            }
            let t = (mp[1] - a[1]) / (b[1] - a[1]);
            if !(-EPS..=1.0 + EPS).contains(&t) {
                continue;
            }
            let x = a[0] + t * (b[0] - a[0]);
            if x >= mp[0] - EPS && best.map(|bb| x - mp[0] < bb.0).unwrap_or(true) {
                best = Some((x - mp[0], k));
            }
        }
        let (dist, k) = best?;
        let pk = if k >= n {
            k - n
        } else {
            let (a, b) = (poly[k], poly[(k + 1) % n]);
            if pts[a][0] > pts[b][0] {
                k
            } else {
                (k + 1) % n
            }
        };
        let i_pt = [mp[0] + dist, mp[1]];
        let p = pts[poly[pk]];
        // A reflex vertex inside triangle M, I, P blocks the bridge; take the one closest in
        // angle to the ray.
        let mut chosen = pk;
        if k < n && (p[0] - i_pt[0]).abs() + (p[1] - i_pt[1]).abs() > EPS {
            let (tri_a, tri_b, tri_c) = if orient(mp, i_pt, p) >= 0.0 {
                (mp, i_pt, p)
            } else {
                (mp, p, i_pt)
            };
            let mut best_r: Option<(f64, f64, usize)> = None;
            for j in 0..n {
                if j == pk {
                    continue;
                }
                let q = pts[poly[j]];
                let prev = pts[poly[(j + n - 1) % n]];
                let next = pts[poly[(j + 1) % n]];
                let reflex = orient(prev, q, next) <= 0.0;
                if reflex && in_triangle(tri_a, tri_b, tri_c, q) && q[0] >= mp[0] - EPS {
                    let ang = (q[1] - mp[1]).abs().atan2(q[0] - mp[0]);
                    let d = (q[0] - mp[0]).hypot(q[1] - mp[1]);
                    if best_r
                        .map(|b| ang < b.0 - EPS || (ang < b.0 + EPS && d < b.1))
                        .unwrap_or(true)
                    {
                        best_r = Some((ang, d, j));
                    }
                }
            }
            if let Some((_, _, j)) = best_r {
                chosen = j;
            }
        }
        // A vertex that appears more than once (an earlier bridge): use the copy whose
        // interior wedge holds M.
        let copies: Vec<usize> = (0..n).filter(|j| poly[*j] == poly[chosen]).collect();
        if copies.len() > 1 {
            let wedge = |j: usize| {
                let (a, p, b) = (
                    pts[poly[(j + n - 1) % n]],
                    pts[poly[j]],
                    pts[poly[(j + 1) % n]],
                );
                if orient(a, p, b) > 0.0 {
                    orient(a, p, mp) > 0.0 && orient(p, b, mp) > 0.0
                } else {
                    orient(a, p, mp) > 0.0 || orient(p, b, mp) > 0.0
                }
            };
            if let Some(j) = copies.iter().copied().find(|j| wedge(*j)) {
                chosen = j;
            }
        }
        // poly[..=chosen], hole from M around back to M, poly[chosen..].
        let mut merged = Vec::with_capacity(poly.len() + hole.len() + 2);
        merged.extend_from_slice(&poly[..=chosen]);
        for s in 0..=hole.len() {
            merged.push(hole[(mi + s) % hole.len()]);
        }
        merged.extend_from_slice(&poly[chosen..]);
        poly = merged;
    }
    Some(poly)
}

/// Ear clipping of a simple polygon (counter-clockwise, indices may repeat at bridges).
fn ear_clip(pts: &[P2], poly: &[usize]) -> Vec<[usize; 3]> {
    let n = poly.len();
    let mut out = Vec::with_capacity(n.saturating_sub(2));
    if n < 3 {
        return out;
    }
    let mut next: Vec<usize> = (0..n).map(|i| (i + 1) % n).collect();
    let mut prev: Vec<usize> = (0..n).map(|i| (i + n - 1) % n).collect();
    let mut alive = n;
    let mut cur = 0;
    let mut stall = 0;
    let p = |i: usize| pts[poly[i]];
    while alive > 3 {
        let (a, b, c) = (prev[cur], cur, next[cur]);
        let convex = orient(p(a), p(b), p(c)) > EPS;
        let mut ear = convex;
        if ear {
            let mut j = next[c];
            while j != a {
                let q = poly[j];
                if q != poly[a] && q != poly[b] && q != poly[c] {
                    let (pj, pn) = (p(prev[j]), p(next[j]));
                    let reflex = orient(pj, p(j), pn) <= EPS;
                    if reflex && in_triangle(p(a), p(b), p(c), p(j)) {
                        ear = false;
                        break;
                    }
                }
                j = next[j];
            }
        }
        if ear || stall > 2 * alive {
            // A stalled clip (a numerically flat ear) is dropped; a real one is kept.
            if orient(p(a), p(b), p(c)) > 0.0 {
                out.push([poly[a], poly[b], poly[c]]);
            }
            next[a] = c;
            prev[c] = a;
            alive -= 1;
            cur = c;
            stall = 0;
        } else {
            cur = next[cur];
            stall += 1;
        }
    }
    let (a, b, c) = (prev[cur], cur, next[cur]);
    if orient(p(a), p(b), p(c)) > 0.0 {
        out.push([poly[a], poly[b], poly[c]]);
    }
    out
}

struct Mesh<'a> {
    pts: &'a [P2],
    tris: Vec<[usize; 3]>,
    edge: HashMap<(usize, usize), usize>,
    fixed: BTreeSet<(usize, usize)>,
}

impl Mesh<'_> {
    fn key(a: usize, b: usize) -> (usize, usize) {
        (a.min(b), a.max(b))
    }

    fn set(&mut self, t: usize, tri: [usize; 3]) {
        self.tris[t] = tri;
        for k in 0..3 {
            self.edge.insert((tri[k], tri[(k + 1) % 3]), t);
        }
    }

    fn push(&mut self, tri: [usize; 3]) -> usize {
        self.tris.push(tri);
        let t = self.tris.len() - 1;
        self.set(t, tri);
        t
    }

    /// Flip the edge `a → b` of triangle `t` when it is not locally Delaunay; the edges to
    /// re-check after a flip are pushed on `stack`.
    fn legalize(&mut self, stack: &mut Vec<(usize, usize)>) {
        let mut guard = 0usize;
        while let Some((a, b)) = stack.pop() {
            guard += 1;
            if guard > 50_000_000 {
                return;
            }
            if self.fixed.contains(&Self::key(a, b)) {
                continue;
            }
            let (Some(&t1), Some(&t2)) = (self.edge.get(&(a, b)), self.edge.get(&(b, a))) else {
                continue;
            };
            let c = third(self.tris[t1], a, b);
            let d = third(self.tris[t2], b, a);
            let (Some(c), Some(d)) = (c, d) else {
                continue;
            };
            let p = self.pts;
            if !in_circle(p[a], p[b], p[c], p[d]) {
                continue;
            }
            // The flipped pair must stay counter-clockwise (the quad is convex).
            if orient(p[a], p[d], p[c]) <= EPS || orient(p[d], p[b], p[c]) <= EPS {
                continue;
            }
            self.edge.remove(&(a, b));
            self.edge.remove(&(b, a));
            self.set(t1, [a, d, c]);
            self.set(t2, [d, b, c]);
            stack.extend([(a, d), (d, b), (b, c), (c, a)]);
        }
    }

    fn insert(&mut self, q: usize) {
        let p = self.pts;
        let pq = p[q];
        let Some(t) = (0..self.tris.len()).find(|&t| {
            let [a, b, c] = self.tris[t];
            in_triangle(p[a], p[b], p[c], pq)
        }) else {
            return;
        };
        let [a, b, c] = self.tris[t];
        // Too close to an edge: leave the point out rather than make a sliver.
        let scale = |u: usize, v: usize| (p[v][0] - p[u][0]).hypot(p[v][1] - p[u][1]);
        for (u, v) in [(a, b), (b, c), (c, a)] {
            let l = scale(u, v).max(1e-300);
            if orient(p[u], p[v], pq).abs() / l < 1e-6 {
                return;
            }
        }
        self.set(t, [a, b, q]);
        self.push([b, c, q]);
        self.push([c, a, q]);
        let mut stack = vec![(a, b), (b, c), (c, a)];
        self.legalize(&mut stack);
    }
}

fn third(t: [usize; 3], a: usize, b: usize) -> Option<usize> {
    for k in 0..3 {
        if t[k] == a && t[(k + 1) % 3] == b {
            return Some(t[(k + 2) % 3]);
        }
    }
    None
}

/// Triangles (counter-clockwise, as indices into `pts`) covering the polygon `loops[0]`
/// (counter-clockwise) minus `loops[1..]` (clockwise), with `interior` points inserted.
/// `None` when the polygon could not be bridged.
pub(super) fn triangulate(
    pts: &[P2],
    loops: &[Vec<usize>],
    interior: &[usize],
) -> Option<Vec<[usize; 3]>> {
    let outer = loops.first()?;
    if outer.len() < 3 {
        return None;
    }
    // Normalise to the unit box.
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for &i in outer {
        for a in 0..2 {
            lo[a] = lo[a].min(pts[i][a]);
            hi[a] = hi[a].max(pts[i][a]);
        }
    }
    let s = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-300);
    let norm: Vec<P2> = pts
        .iter()
        .map(|p| [(p[0] - lo[0]) / s, (p[1] - lo[1]) / s])
        .collect();
    let poly = bridge(&norm, outer, &loops[1..])?;
    let tris = ear_clip(&norm, &poly);
    let mut mesh = Mesh {
        pts: &norm,
        tris: Vec::with_capacity(tris.len() + 2 * interior.len()),
        edge: HashMap::new(),
        fixed: BTreeSet::new(),
    };
    for l in loops {
        for k in 0..l.len() {
            mesh.fixed.insert(Mesh::key(l[k], l[(k + 1) % l.len()]));
        }
    }
    for t in tris {
        mesh.push(t);
    }
    // Sorted, so the flips (and the triangulation of cocircular points) are deterministic.
    let mut stack: Vec<(usize, usize)> = mesh.edge.keys().copied().collect();
    stack.sort_unstable();
    mesh.legalize(&mut stack);
    for &q in interior {
        mesh.insert(q);
    }
    Some(mesh.tris)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area(pts: &[P2], tris: &[[usize; 3]]) -> f64 {
        tris.iter()
            .map(|t| orient(pts[t[0]], pts[t[1]], pts[t[2]]) / 2.0)
            .sum()
    }

    #[test]
    fn square_with_a_round_hole_covers_its_area() {
        let mut pts = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let n = 48;
        let mut hole = Vec::new();
        for k in 0..n {
            let a = -(k as f64) * std::f64::consts::TAU / n as f64;
            pts.push([5.0 + 2.0 * a.cos(), 5.0 + 2.0 * a.sin()]);
            hole.push(4 + k);
        }
        let tris = triangulate(&pts, &[vec![0, 1, 2, 3], hole], &[]).unwrap();
        let poly_hole = 0.5 * n as f64 * 4.0 * (std::f64::consts::TAU / n as f64).sin();
        assert!((area(&pts, &tris) - (100.0 - poly_hole)).abs() < 1e-9);
        assert_eq!(tris.len(), 4 + n + 2 - 2);
        assert!(tris
            .iter()
            .all(|t| orient(pts[t[0]], pts[t[1]], pts[t[2]]) > 0.0));
    }

    #[test]
    fn a_second_hole_bridges_past_the_first_ones_bridge() {
        // The second hole's ray meets the first hole's bridge, whose far end appears twice.
        let mut pts = vec![[0.0, 1.5], [80.0, 1.5], [80.0, 50.0], [0.0, 50.0]];
        let mut hole = Vec::new();
        for k in 0..24 {
            let a = -(k as f64) * std::f64::consts::TAU / 24.0;
            pts.push([60.0 + 2.0 * a.cos(), 10.0 + 2.0 * a.sin()]);
            hole.push(pts.len() - 1);
        }
        let base = pts.len();
        pts.extend([[40.0, 20.0], [20.0, 20.0], [20.0, 30.0], [40.0, 30.0]]);
        let cut = vec![base, base + 1, base + 2, base + 3];
        let tris = triangulate(&pts, &[vec![0, 1, 2, 3], hole, cut], &[]).unwrap();
        assert_eq!(tris.len(), 4 + 24 + 4 + 4 - 2);
        let poly_hole = 0.5 * 24.0 * 4.0 * (std::f64::consts::TAU / 24.0).sin();
        assert!((area(&pts, &tris) - (80.0 * 48.5 - poly_hole - 200.0)).abs() < 1e-9);
    }

    #[test]
    fn interior_points_are_inserted() {
        let mut pts = vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let mut inner = Vec::new();
        for i in 1..4 {
            for j in 1..4 {
                pts.push([i as f64, j as f64 + 0.01 * i as f64]);
                inner.push(pts.len() - 1);
            }
        }
        let tris = triangulate(&pts, &[vec![0, 1, 2, 3]], &inner).unwrap();
        assert!((area(&pts, &tris) - 16.0).abs() < 1e-9);
        assert_eq!(tris.len(), 2 * 9 + 2);
    }
}
