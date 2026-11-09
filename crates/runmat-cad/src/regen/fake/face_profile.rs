//! A planar grid face as a profile (a [`ToolOp::profile_face`]): the union of the face's rects
//! traced into closed boundary loops, outer first, in the profile frame. Curved and faceted
//! faces, and frames not parallel to the face, are refused.

use super::*;

/// The loops of planar face `face` of `s` in `frame` (which lies in the face's plane).
pub(super) fn face_loops(
    s: &FakeSolid,
    face: u32,
    frame: &Frame,
) -> Result<Vec<Vec<ProfileSegment>>, KernelError> {
    let f = s.faces.get(face as usize).ok_or(KernelError::Failed {
        message: "the profile face is not a face of its body".into(),
    })?;
    let plane = f.plane.ok_or(KernelError::NotSupported {
        what: "a curved or faceted face as a profile on the grid kernel".into(),
    })?;
    if frame.z[plane.axis].abs() < 1.0 - 1e-9 {
        return Err(KernelError::Failed {
            message: "the profile frame is not in the face's plane".into(),
        });
    }
    let (a, b) = ((plane.axis + 1) % 3, (plane.axis + 2) % 3);
    let mut us: Vec<f64> = Vec::new();
    let mut vs: Vec<f64> = Vec::new();
    for r in &f.rects {
        us.extend([r.lo[a], r.hi[a]]);
        vs.extend([r.lo[b], r.hi[b]]);
    }
    let dedup = |xs: &mut Vec<f64>| {
        xs.sort_by(f64::total_cmp);
        xs.dedup_by(|x, y| (*x - *y).abs() < EPS);
    };
    dedup(&mut us);
    dedup(&mut vs);
    if us.len() < 2 || vs.len() < 2 {
        return Err(KernelError::Failed {
            message: "the profile face has no area".into(),
        });
    }
    let (nu, nv) = (us.len() - 1, vs.len() - 1);
    let filled = |i: isize, j: isize| -> bool {
        if i < 0 || j < 0 || i as usize >= nu || j as usize >= nv {
            return false;
        }
        let (i, j) = (i as usize, j as usize);
        let c = [(us[i] + us[i + 1]) / 2.0, (vs[j] + vs[j + 1]) / 2.0];
        f.rects.iter().any(|r| {
            c[0] > r.lo[a] && c[0] < r.hi[a] && c[1] > r.lo[b] && c[1] < r.hi[b]
        })
    };
    // Boundary edges with the face on their left (counter-clockwise about filled cells in a, b).
    let mut edges: Vec<((usize, usize), (usize, usize))> = Vec::new();
    for i in 0..nu {
        for j in 0..nv {
            let (x, y) = (i as isize, j as isize);
            if !filled(x, y) {
                continue;
            }
            if !filled(x, y - 1) {
                edges.push(((i, j), (i + 1, j)));
            }
            if !filled(x + 1, y) {
                edges.push(((i + 1, j), (i + 1, j + 1)));
            }
            if !filled(x, y + 1) {
                edges.push(((i + 1, j + 1), (i, j + 1)));
            }
            if !filled(x - 1, y) {
                edges.push(((i, j + 1), (i, j)));
            }
        }
    }
    let mut used = vec![false; edges.len()];
    let mut loops: Vec<Vec<(usize, usize)>> = Vec::new();
    for k in 0..edges.len() {
        if used[k] {
            continue;
        }
        used[k] = true;
        let start = edges[k].0;
        let mut pts = vec![start];
        let mut at = edges[k].1;
        while at != start {
            pts.push(at);
            let Some(n) = (0..edges.len()).find(|n| !used[*n] && edges[*n].0 == at) else {
                return Err(KernelError::Failed {
                    message: "the profile face's boundary does not close".into(),
                });
            };
            used[n] = true;
            at = edges[n].1;
        }
        // Drop the corners where the boundary runs straight on.
        let m = pts.len();
        let straight = |p: (usize, usize), q: (usize, usize), r: (usize, usize)| {
            (p.0 == q.0 && q.0 == r.0) || (p.1 == q.1 && q.1 == r.1)
        };
        let corners: Vec<(usize, usize)> = (0..m)
            .filter(|k| !straight(pts[(k + m - 1) % m], pts[*k], pts[(k + 1) % m]))
            .map(|k| pts[k])
            .collect();
        loops.push(corners);
    }
    let world = |p: (usize, usize)| {
        let mut w = [0.0; 3];
        w[plane.axis] = plane.offset;
        w[a] = us[p.0];
        w[b] = vs[p.1];
        let d = [w[0] - frame.origin[0], w[1] - frame.origin[1], w[2] - frame.origin[2]];
        let dot = |e: [f64; 3]| d[0] * e[0] + d[1] * e[1] + d[2] * e[2];
        [dot(frame.x), dot(frame.y)]
    };
    let area = |l: &[(usize, usize)]| -> f64 {
        (0..l.len())
            .map(|k| {
                let (p, q) = (l[k], l[(k + 1) % l.len()]);
                us[p.0] * vs[q.1] - us[q.0] * vs[p.1]
            })
            .sum::<f64>()
            .abs()
    };
    loops.sort_by(|x, y| area(y).total_cmp(&area(x)));
    let mut k = 0usize;
    Ok(loops
        .into_iter()
        .map(|l| {
            (0..l.len())
                .map(|i| {
                    let seg = ProfileSegment {
                        curve: ToolOp::face_edge_curve(k),
                        kind: SegmentKind::Line,
                        start: world(l[i]),
                        end: world(l[(i + 1) % l.len()]),
                    };
                    k += 1;
                    seg
                })
                .collect()
        })
        .collect())
}
