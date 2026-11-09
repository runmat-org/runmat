//! Face meshes: faces drawn on their facets, faces triangulated on their charts, and edge
//! polylines.

use super::*;

impl Ctx<'_> {
    pub(super) fn face_on_facets(&self, face: usize, facets: &[Facet]) -> FaceMesh {
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
    pub(super) fn face_on_chart(
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

    pub(super) fn edge_polyline(&self, e: &EdgeTopo) -> Vec<P3> {
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
