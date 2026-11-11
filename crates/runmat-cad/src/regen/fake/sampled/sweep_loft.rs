//! Sweeps and lofts as sampled tools.

use super::*;

/// A translational sweep of profile loops (local z = 0) along a polyline path given in local
/// coordinates. Side faces are numbered by path segment when the path has corners.
pub(in crate::regen::fake) fn sweep(
    to_world: Transform,
    loops: &[Vec<(Curve2, HistorySource)>],
    path: &[[f64; 3]],
    path_curves: &[u32],
    cap_labels: [HistorySource; 2],
) -> Result<Sampled, KernelError> {
    if path.len() < 2 {
        return Err(KernelError::Failed {
            message: "a sweep path needs at least two points".into(),
        });
    }
    let offsets: Vec<[f64; 3]> = path.iter().map(|p| sub3(*p, path[0])).collect();
    let n = offsets.len() - 1;
    for j in 0..n {
        if (offsets[j + 1][2] - offsets[j][2]).abs() < 1e-12 {
            return Err(unsupported("sweep path segments in the profile plane"));
        }
    }
    // Path curve per segment: given, or one per segment.
    let curve_of: Vec<u32> = if path_curves.len() == n {
        path_curves.to_vec()
    } else {
        (0..n as u32).collect()
    };
    let distinct = {
        let mut v = curve_of.clone();
        v.sort();
        v.dedup();
        v.len()
    };
    let mut surfaces: Vec<Surf> = Vec::new();
    let mut curves = Vec::new();
    let mut sides = Vec::new();
    for l in loops {
        let mut cl = Vec::new();
        let mut sl = Vec::new();
        for (c, lab) in l {
            cl.push(c.clone());
            let mut per = Vec::new();
            let mut by_curve: std::collections::BTreeMap<u32, usize> =
                std::collections::BTreeMap::new();
            for j in 0..n {
                if let Some(si) = by_curve.get(&curve_of[j]) {
                    // A further sample of the same path curve: the same face.
                    per.push(*si);
                    if let SurfaceKind::Plane { .. } = surfaces[*si].hint {
                        let seg = sub3(offsets[j + 1], offsets[j]);
                        let prev = sub3(offsets[j], offsets[j.saturating_sub(1)]);
                        if norm3(seg) > 0.0 && j > 0 && {
                            let c = dot3(seg, prev) / (norm3(seg) * norm3(prev).max(1e-300));
                            (c - 1.0).abs() > 1e-9
                        } {
                            surfaces[*si].hint = SurfaceKind::Other;
                        }
                    }
                    continue;
                }
                let mut label = lab.clone();
                if distinct > 1 {
                    if let HistorySource::Generated { ordinal, .. } = &mut label {
                        *ordinal = Some(curve_of[j]);
                    }
                }
                by_curve.insert(curve_of[j], surfaces.len());
                per.push(surfaces.len());
                // A straight curve swept along a straight segment is planar.
                let hint = match c.line() {
                    Some((a, b)) => {
                        let l = [b[0] - a[0], b[1] - a[1], 0.0];
                        let d = sub3(offsets[j + 1], offsets[j]);
                        let nrm = [
                            l[1] * d[2] - l[2] * d[1],
                            l[2] * d[0] - l[0] * d[2],
                            l[0] * d[1] - l[1] * d[0],
                        ];
                        let m = norm3(nrm).max(1e-300);
                        plane_hint([nrm[0] / m, nrm[1] / m, nrm[2] / m], &to_world)
                    }
                    None => SurfaceKind::Other,
                };
                surfaces.push(Surf {
                    label,
                    hint,
                    plane: None,
                });
            }
            sl.push(per);
        }
        curves.push(cl);
        sides.push(sl);
    }
    let [c0, c1] = cap_labels;
    let caps = [surfaces.len(), surfaces.len() + 1];
    surfaces.push(Surf {
        label: c0,
        hint: plane_hint([0.0, 0.0, -1.0], &to_world),
        plane: None,
    });
    surfaces.push(Surf {
        label: c1,
        hint: plane_hint([0.0, 0.0, 1.0], &to_world),
        plane: None,
    });
    let indexed: Vec<Vec<(Curve2, usize)>> = curves
        .iter()
        .map(|l| l.iter().map(|c| (c.clone(), 0)).collect())
        .collect();
    let counts: Vec<Vec<usize>> = curves
        .iter()
        .map(|l| l.iter().map(|c| c.samples()).collect())
        .collect();
    let polys = loop_polys(&indexed, &counts);
    let pts: Vec<[f64; 3]> = offsets
        .iter()
        .flat_map(|o| {
            polys
                .iter()
                .flatten()
                .map(move |q| [q[0] + o[0], q[1] + o[1], o[2]])
                .collect::<Vec<_>>()
        })
        .map(|p| to_world.apply(p))
        .collect();
    let (lo, hi) = bbox(pts);
    let mut s = Sampled {
        to_local: to_world.inverse(),
        shape: Shape::Sweep {
            loops: curves,
            polys,
            offsets,
            sides,
            caps,
        },
        surfaces,
        lo,
        hi,
        exact: Vec::new(),
    };
    s.recompute_exact();
    Ok(s)
}

/// A loft through sections at local heights `z` (strictly increasing after sorting), each with
/// the same loops and curve counts, sampled as ruled strips between sections. Each first-section
/// curve makes one side face through every section, as OCCT's smooth loft does.
pub(in crate::regen::fake) fn loft(
    to_world: Transform,
    sections: Vec<(f64, Vec<Vec<Curve2>>)>,
    labels: &[Vec<HistorySource>],
    cap_labels: [HistorySource; 2],
) -> Result<Sampled, KernelError> {
    if sections.len() < 2 {
        return Err(KernelError::Failed {
            message: "a loft needs at least two sections".into(),
        });
    }
    let shape0: Vec<usize> = sections[0].1.iter().map(|l| l.len()).collect();
    for (_, s) in &sections {
        let sh: Vec<usize> = s.iter().map(|l| l.len()).collect();
        if sh != shape0 {
            return Err(unsupported(
                "loft sections with different loop or curve counts",
            ));
        }
    }
    let mut order: Vec<usize> = (0..sections.len()).collect();
    let increasing = sections[sections.len() - 1].0 > sections[0].0;
    if !increasing {
        order.reverse();
    }
    let z: Vec<f64> = order.iter().map(|i| sections[*i].0).collect();
    if z.windows(2).any(|w| w[1] - w[0] < 1e-9) {
        return Err(unsupported(
            "loft sections that are not stacked along the normal",
        ));
    }
    let spans = z.len() - 1;
    // Equal sample counts per curve across sections.
    let mut counts: Vec<Vec<usize>> = shape0.iter().map(|n| vec![1; *n]).collect();
    for (_, s) in &sections {
        for (li, l) in s.iter().enumerate() {
            for (ci, c) in l.iter().enumerate() {
                counts[li][ci] = counts[li][ci].max(c.samples());
            }
        }
    }
    let curves: Vec<Vec<Vec<Vec<[f64; 2]>>>> = order
        .iter()
        .map(|i| {
            sections[*i]
                .1
                .iter()
                .enumerate()
                .map(|(li, l)| {
                    l.iter()
                        .enumerate()
                        .map(|(ci, c)| c.polyline(counts[li][ci]))
                        .collect()
                })
                .collect()
        })
        .collect();
    let mut surfaces = Vec::new();
    let mut sides = Vec::new();
    for (li, l) in labels.iter().enumerate() {
        let mut sl = Vec::new();
        for lab in l.iter().take(shape0[li]) {
            sl.push(vec![surfaces.len(); spans]);
            surfaces.push(Surf {
                label: lab.clone(),
                hint: SurfaceKind::Other,
                plane: None,
            });
        }
        sides.push(sl);
    }
    let [mut c0, mut c1] = cap_labels;
    if !increasing {
        std::mem::swap(&mut c0, &mut c1);
    }
    let caps = [surfaces.len(), surfaces.len() + 1];
    surfaces.push(Surf {
        label: c0,
        hint: plane_hint([0.0, 0.0, -1.0], &to_world),
        plane: None,
    });
    surfaces.push(Surf {
        label: c1,
        hint: plane_hint([0.0, 0.0, 1.0], &to_world),
        plane: None,
    });
    let pts: Vec<[f64; 3]> = curves
        .iter()
        .zip(&z)
        .flat_map(|(sec, zz)| {
            sec.iter()
                .flatten()
                .flatten()
                .map(|q| [q[0], q[1], *zz])
                .collect::<Vec<_>>()
        })
        .map(|p| to_world.apply(p))
        .collect();
    let (lo, hi) = bbox(pts);
    let mut s = Sampled {
        to_local: to_world.inverse(),
        shape: Shape::Loft {
            z,
            curves,
            sides,
            caps,
        },
        surfaces,
        lo,
        hi,
        exact: Vec::new(),
    };
    s.recompute_exact();
    Ok(s)
}
