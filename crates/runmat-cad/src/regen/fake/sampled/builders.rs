//! Prisms and revolves as sampled tools, with the hints and helpers they share.

use super::*;

pub(super) fn sweep_coords(offsets: &[[f64; 3]], j: usize, p: [f64; 3]) -> (f64, [f64; 2]) {
    let o = offsets[j];
    let d = sub3(offsets[j + 1], o);
    let rel = sub3(p, o);
    let s = if d[2].abs() > 1e-12 {
        rel[2] / d[2]
    } else {
        -1.0
    };
    (s, [rel[0] - s * d[0], rel[1] - s * d[1]])
}

pub(super) fn span_of(z: &[f64], v: f64) -> usize {
    let mut k = 0;
    while k + 2 < z.len() && v > z[k + 1] {
        k += 1;
    }
    k
}

pub(super) fn loft_polys(
    z: &[f64],
    curves: &[Vec<Vec<Vec<[f64; 2]>>>],
    v: f64,
) -> Vec<Vec<[f64; 2]>> {
    let k = span_of(z, v);
    let t = ((v - z[k]) / (z[k + 1] - z[k])).clamp(0.0, 1.0);
    curves[k]
        .iter()
        .zip(&curves[k + 1])
        .map(|(l0, l1)| {
            let mut pts = Vec::new();
            for (c0, c1) in l0.iter().zip(l1) {
                for (a, b) in c0.iter().zip(c1).take(c0.len() - 1) {
                    pts.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
                }
            }
            pts
        })
        .collect()
}

pub(super) fn moved_box(lo: [f64; 3], hi: [f64; 3], t: &Transform) -> ([f64; 3], [f64; 3]) {
    let mut nlo = [f64::INFINITY; 3];
    let mut nhi = [f64::NEG_INFINITY; 3];
    for i in 0..8 {
        let c = [
            if i & 1 == 0 { lo[0] } else { hi[0] },
            if i & 2 == 0 { lo[1] } else { hi[1] },
            if i & 4 == 0 { lo[2] } else { hi[2] },
        ];
        let w = t.apply(c);
        for a in 0..3 {
            nlo[a] = nlo[a].min(w[a]);
            nhi[a] = nhi[a].max(w[a]);
        }
    }
    (nlo, nhi)
}

pub(in crate::regen::fake) fn move_hint(h: &SurfaceKind, t: &Transform) -> SurfaceKind {
    match *h {
        SurfaceKind::Plane { normal } => SurfaceKind::Plane {
            normal: t.apply_vector(normal),
        },
        SurfaceKind::Cylinder {
            axis_point,
            axis_direction,
            radius,
        } => SurfaceKind::Cylinder {
            axis_point: t.apply(axis_point),
            axis_direction: t.apply_vector(axis_direction),
            radius,
        },
        SurfaceKind::Other => SurfaceKind::Other,
    }
}

/// The box around world points.
pub(super) fn bbox(points: impl IntoIterator<Item = [f64; 3]>) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    (lo, hi)
}

pub(super) fn unsupported(what: &str) -> KernelError {
    KernelError::NotSupported {
        what: format!("fake kernel: {}", what),
    }
}

pub(super) fn plane_hint(local_normal: [f64; 3], to_world: &Transform) -> SurfaceKind {
    SurfaceKind::Plane {
        normal: to_world.apply_vector(local_normal),
    }
}

/// Side-face hint of a straight profile curve in the profile plane, swept along local z.
pub(super) fn side_hint(c: &Curve2, to_world: &Transform) -> SurfaceKind {
    match c {
        Curve2::Line { a, b } => {
            let d = [b[0] - a[0], b[1] - a[1]];
            let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-300);
            // Outward for a counter-clockwise outer loop.
            plane_hint([d[1] / l, -d[0] / l, 0.0], to_world)
        }
        Curve2::Arc { c, r, .. } | Curve2::Circle { c, r } => SurfaceKind::Cylinder {
            axis_point: to_world.apply([c[0], c[1], 0.0]),
            axis_direction: to_world.apply_vector([0.0, 0.0, 1.0]),
            radius: *r,
        },
        Curve2::Poly { .. } => SurfaceKind::Other,
    }
}

/// A prism of any profile on any frame: `to_world` places the profile plane (local z = 0);
/// the prism spans local `z0..z1` with `labels[0]` at `z0`.
pub(in crate::regen::fake) fn prism(
    to_world: Transform,
    loops: &[Vec<(Curve2, HistorySource)>],
    z0: f64,
    z1: f64,
    cap_labels: [HistorySource; 2],
) -> Sampled {
    let mut surfaces = Vec::new();
    let mut l2 = Vec::new();
    for l in loops {
        let mut v = Vec::new();
        for (c, lab) in l {
            v.push((c.clone(), surfaces.len()));
            surfaces.push(Surf {
                label: lab.clone(),
                hint: side_hint(c, &to_world),
                plane: None,
            });
        }
        l2.push(v);
    }
    let (z0, z1, caps_l) = if z0 <= z1 {
        (z0, z1, cap_labels)
    } else {
        let [a, b] = cap_labels;
        (z1, z0, [b, a])
    };
    let [c0, c1] = caps_l;
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
    let counts: Vec<Vec<usize>> = l2
        .iter()
        .map(|l| l.iter().map(|(c, _)| c.samples()).collect())
        .collect();
    let polys = loop_polys(&l2, &counts);
    let pts: Vec<[f64; 3]> = polys
        .iter()
        .flatten()
        .flat_map(|q| [[q[0], q[1], z0], [q[0], q[1], z1]])
        .map(|p| to_world.apply(p))
        .collect();
    let (lo, hi) = bbox(pts);
    let mut s = Sampled {
        to_local: to_world.inverse(),
        shape: Shape::Prism {
            loops: l2,
            polys,
            z0,
            z1,
            caps,
        },
        surfaces,
        lo,
        hi,
        exact: Vec::new(),
    };
    s.recompute_exact();
    s
}

/// A revolve of profile loops (local z = 0 plane of `to_world`) about the local in-plane axis
/// through `ap` along `ad`, by `angle` radians (negative: the other way).
pub(in crate::regen::fake) fn revolve(
    to_world: Transform,
    loops: &[Vec<(Curve2, HistorySource)>],
    ap: [f64; 2],
    ad: [f64; 2],
    angle: f64,
    cap_labels: Option<[HistorySource; 2]>,
) -> Result<Sampled, KernelError> {
    let l = (ad[0] * ad[0] + ad[1] * ad[1]).sqrt();
    if l < 1e-12 {
        return Err(unsupported("revolve axis out of the profile plane"));
    }
    let ad = [ad[0] / l, ad[1] / l];
    let mut surfaces = Vec::new();
    let mut l2 = Vec::new();
    let mut centroid = [0.0, 0.0];
    let mut count = 0.0;
    for lp in loops {
        let mut v = Vec::new();
        for (c, lab) in lp {
            for q in c.polyline(c.samples()) {
                centroid[0] += q[0];
                centroid[1] += q[1];
                count += 1.0;
            }
            v.push((c.clone(), surfaces.len()));
            let hint = match c.line() {
                Some((a, b)) => {
                    let d = [b[0] - a[0], b[1] - a[1]];
                    let along = (d[0] * ad[0] + d[1] * ad[1]).abs() / (d2(a, b).max(1e-300));
                    let r = {
                        let rel = [a[0] - ap[0], a[1] - ap[1]];
                        let h = rel[0] * ad[0] + rel[1] * ad[1];
                        d2(rel, [h * ad[0], h * ad[1]])
                    };
                    if (along - 1.0).abs() < 1e-9 {
                        SurfaceKind::Cylinder {
                            axis_point: to_world.apply([ap[0], ap[1], 0.0]),
                            axis_direction: to_world.apply_vector([ad[0], ad[1], 0.0]),
                            radius: r,
                        }
                    } else if along < 1e-9 {
                        plane_hint([ad[0], ad[1], 0.0], &to_world)
                    } else {
                        SurfaceKind::Other
                    }
                }
                None => SurfaceKind::Other,
            };
            surfaces.push(Surf {
                label: lab.clone(),
                hint,
                plane: None,
            });
        }
        l2.push(v);
    }
    if count == 0.0 {
        return Err(unsupported("revolve of an empty profile"));
    }
    centroid = [centroid[0] / count, centroid[1] / count];
    let rel = [centroid[0] - ap[0], centroid[1] - ap[1]];
    let h = rel[0] * ad[0] + rel[1] * ad[1];
    let perp = [rel[0] - h * ad[0], rel[1] - h * ad[1]];
    let pl = (perp[0] * perp[0] + perp[1] * perp[1]).sqrt();
    if pl < 1e-12 {
        return Err(unsupported("revolve profile centred on its axis"));
    }
    let e1 = [perp[0] / pl, perp[1] / pl];
    let sense = if angle < 0.0 { -1.0 } else { 1.0 };
    let angle = angle.abs().min(TAU);
    let full = angle >= TAU - 1e-9;
    let caps = match (full, cap_labels) {
        (false, Some([s, e])) => {
            let i = surfaces.len();
            // The start cap lies in the profile plane, facing against the sweep.
            surfaces.push(Surf {
                label: s,
                hint: plane_hint([0.0, 0.0, -sense], &to_world),
                plane: None,
            });
            // The end cap is the profile plane turned through the angle.
            let end_normal = Transform::rotation_about(
                to_world.apply([ap[0], ap[1], 0.0]),
                to_world.apply_vector([ad[0], ad[1], 0.0]),
                sense * angle,
            )
            .apply_vector(to_world.apply_vector([0.0, 0.0, sense]));
            surfaces.push(Surf {
                label: e,
                hint: SurfaceKind::Plane { normal: end_normal },
                plane: None,
            });
            Some([i, i + 1])
        }
        _ => None,
    };
    let counts: Vec<Vec<usize>> = l2
        .iter()
        .map(|l| l.iter().map(|(c, _)| c.samples()).collect())
        .collect();
    let polys = loop_polys(&l2, &counts);
    let mut s = Sampled {
        to_local: to_world.inverse(),
        shape: Shape::Revolve {
            loops: l2,
            polys: polys.clone(),
            ap,
            ad,
            e1,
            sense,
            angle,
            caps,
        },
        surfaces,
        lo: [0.0; 3],
        hi: [0.0; 3],
        exact: Vec::new(),
    };
    // Bounding box: profile samples turned through the sweep.
    let steps = 48;
    let mut pts = Vec::new();
    let ad3 = [ad[0], ad[1], 0.0];
    let e13 = [e1[0], e1[1], 0.0];
    let e23 = [
        sense * (ad3[1] * e13[2] - ad3[2] * e13[1]),
        sense * (ad3[2] * e13[0] - ad3[0] * e13[2]),
        sense * (ad3[0] * e13[1] - ad3[1] * e13[0]),
    ];
    for q in polys.iter().flatten() {
        let rel = [q[0] - ap[0], q[1] - ap[1]];
        let h = rel[0] * ad[0] + rel[1] * ad[1];
        let r = d2(rel, [h * ad[0], h * ad[1]]);
        for k in 0..=steps {
            let a = angle * k as f64 / steps as f64;
            let (sn, cs) = a.sin_cos();
            let p = [
                ap[0] + h * ad3[0] + r * (cs * e13[0] + sn * e23[0]),
                ap[1] + h * ad3[1] + r * (cs * e13[1] + sn * e23[1]),
                r * (cs * e13[2] + sn * e23[2]),
            ];
            pts.push(to_world.apply(p));
        }
    }
    let (mut lo, mut hi) = bbox(pts);
    let pad = 1e-6 * (0..3).map(|a| hi[a] - lo[a]).fold(1.0, f64::max);
    for a in 0..3 {
        lo[a] -= pad;
        hi[a] += pad;
    }
    s.lo = lo;
    s.hi = hi;
    s.recompute_exact();
    Ok(s)
}
