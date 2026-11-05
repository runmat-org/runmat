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
            let pair = (on_plane(fa, 0) && on_plane(fb, 1)) || (on_plane(fa, 1) && on_plane(fb, 0));
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
                    run.faces == key && (run.pts.first() == Some(c) || run.pts.last() == Some(c))
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
