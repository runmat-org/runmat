//! Angular dimensions drawn as arcs (ISO 129-1 and ASME Y14.5): an arc centred where the two
//! faces' edge-on lines meet, arrowheads on both ends, extension lines carrying each face's line
//! out to the arc, and the angle's figure over the arc's middle. The arc goes on the side of the
//! vertex clear of the view, nearest first, further out while its figure would overlap
//! something.

use super::marks::{line, mark, text, unit};
use super::solver::BoxR;
use super::views::ViewPlan;
use super::{floats, Ctx, Prim, Std, Xy};

/// The arc dimension of angular annotation `aid` on view `v`, with the box its figure takes;
/// `None` when its faces aren't both planes seen edge-on, or their lines are parallel.
pub fn arc(ctx: &Ctx<'_>, v: &ViewPlan, id: &str, aid: &str, figure: &str) -> Option<(Vec<Prim>, BoxR)> {
    let anchors: Vec<Xy> = ctx.data["anchors"][aid].as_array()?.iter().filter_map(|c| {
        let f = floats(c);
        (f.len() == 3).then(|| v.model_to_sheet([f[0], f[1], f[2]]))
    }).collect();
    let normals: Vec<[f64; 3]> = ctx.data["normals"][aid].as_array()?.iter().filter_map(|n| {
        let f = floats(n);
        (f.len() == 3).then(|| [f[0], f[1], f[2]])
    }).collect();
    if anchors.len() != 2 || normals.len() != 2 {
        return None;
    }
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    // Each face's line on the sheet: square to its normal as the view sees it.
    let mut dirs = [[0.0; 2]; 2];
    for (k, n) in normals.iter().enumerate() {
        let (nu, nv) = (dot(*n, v.axes.u), dot(*n, v.axes.v));
        let l = (nu * nu + nv * nv).sqrt();
        // A face the view doesn't see edge-on shows no line to measure from.
        if l < 0.99 {
            return None;
        }
        dirs[k] = [-nv / l, nu / l];
    }
    let (p, q) = (anchors[0], anchors[1]);
    let cross = |a: Xy, b: Xy| a[0] * b[1] - a[1] * b[0];
    let den = cross(dirs[0], dirs[1]);
    if den.abs() < 1e-6 {
        return None;
    }
    let t = cross([q[0] - p[0], q[1] - p[1]], dirs[1]) / den;
    let vertex = [p[0] + dirs[0][0] * t, p[1] + dirs[0][1] * t];
    // Rays from the vertex toward each face.
    let toward = |a: Xy, d: Xy| if (a[0] - vertex[0]) * d[0] + (a[1] - vertex[1]) * d[1] >= 0.0 { d } else { [-d[0], -d[1]] };
    let rays = [toward(p, dirs[0]), toward(q, dirs[1])];
    let st = &ctx.std;
    let g = v.geom_box();
    let reach = [super::marks::len(vertex, p), super::marks::len(vertex, q)];
    let mut best: Option<(Vec<Prim>, BoxR)> = None;
    // The arc between the faces' rays, or between their continuations past the vertex (the
    // vertical angle, the same size), whichever lies clear of the view; out from the vertex
    // until its figure is clear of every label, view and table.
    for outward in [true, false] {
        let r = if outward { [[-rays[0][0], -rays[0][1]], [-rays[1][0], -rays[1][1]]] } else { rays };
        for step in 0..8 {
            let radius = st.first + 2.0 + step as f64 * st.step;
            let (prims, b, mid) = arc_dimension(st, id, vertex, r[0], r[1], radius, figure);
            let clear_of_view = !g.hits(&BoxR::new(mid[0], mid[1], mid[0], mid[1]));
            if !clear_of_view {
                break;
            }
            let mut prims = prims;
            // Extension lines carry each face's line to the arc when the arc runs past it.
            for k in 0..2 {
                let from = if outward { st.gap } else { reach[k] + st.gap };
                let to = radius + st.over;
                if to > from {
                    let (s, e) = ([vertex[0] + r[k][0] * from, vertex[1] + r[k][1] * from], [vertex[0] + r[k][0] * to, vertex[1] + r[k][1] * to]);
                    prims.push(mark(id, "dimink", line(vec![s[0], s[1], e[0], e[1]])));
                }
            }
            let free = ctx.occ.free(&b, id, 0.5, &[]);
            if free {
                return Some((prims, b));
            }
            if best.is_none() {
                best = Some((prims, b));
            }
        }
    }
    best
}

/// An arc of `radius` about `c` from ray `a` to ray `b` (the shorter way), arrowheads at both
/// ends pointing out along it, and `t` over its middle. Returns the primitives, the figure's
/// box and the arc's middle.
pub fn arc_dimension(st: &Std, id: &str, c: Xy, a: Xy, b: Xy, radius: f64, t: &str) -> (Vec<Prim>, BoxR, Xy) {
    let (a0, mut a1) = (a[1].atan2(a[0]), b[1].atan2(b[0]));
    let mut sweep = a1 - a0;
    while sweep > std::f64::consts::PI {
        sweep -= 2.0 * std::f64::consts::PI;
    }
    while sweep < -std::f64::consts::PI {
        sweep += 2.0 * std::f64::consts::PI;
    }
    a1 = a0 + sweep;
    let at = |ang: f64| [c[0] + radius * ang.cos(), c[1] + radius * ang.sin()];
    let n = ((sweep.abs() / (std::f64::consts::PI / 36.0)).ceil() as usize).max(4);
    let pts: Vec<f64> = (0..=n).flat_map(|k| at(a0 + sweep * k as f64 / n as f64)).collect();
    let mut out = vec![mark(id, "dimink", line(pts))];
    // Arrowheads at both ends, tangent to the arc, pointing out of it.
    let tangent = |ang: f64, s: f64| [-ang.sin() * s, ang.cos() * s];
    let sign = sweep.signum();
    out.push(st.arrowhead(id, at(a0), tangent(a0, -sign)));
    out.push(st.arrowhead(id, at(a1), tangent(a1, sign)));
    let mid_ang = a0 + sweep / 2.0;
    let mid = at(mid_ang);
    let w = Std::text_width(t, st.text);
    let out_dir = unit(c, mid);
    // The figure just outside the arc's middle, upright.
    let fc = [mid[0] + out_dir[0] * (st.text * 0.6 + w * 0.5 * out_dir[0].abs() + 1.0), mid[1] + out_dir[1] * (st.text * 0.6 + 1.0)];
    out.push(mark(id, "dimtext", text(fc[0], fc[1] - st.text / 2.0, t, st.text, "middle")));
    let b = BoxR::new(fc[0] - w / 2.0, fc[1] - st.text * 0.7, fc[0] + w / 2.0, fc[1] + st.text * 0.7);
    (out, b, mid)
}
