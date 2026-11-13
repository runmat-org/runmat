//! Images of whole bodies under rigid motions or reflections, and extrusions of planar faces.

use super::*;

/// The image of a whole solid under `t` (any rigid motion or reflection). Faces become
/// [`HistorySource::Copy`] of the operand's faces at `instance`, or continue them
/// ([`HistorySource::Input`]) when the body itself moves.
pub(in crate::regen::fake) fn image(
    solid: &FakeSolid,
    operand: u32,
    t: &Transform,
    instance: Option<&InstanceKey>,
) -> Result<Sampled, KernelError> {
    if !solid.decorations.is_empty() {
        return Err(unsupported(
            "moving blended bodies by anything but right-angle motions",
        ));
    }
    let surfaces = solid
        .faces
        .iter()
        .enumerate()
        .map(|(i, f)| Surf {
            label: {
                let from = OperandEntity {
                    operand,
                    entity: i as u32,
                };
                match instance {
                    Some(k) => HistorySource::Copy {
                        from,
                        instance: k.clone(),
                    },
                    None => HistorySource::Input { from },
                }
            },
            hint: move_hint(&f.hint(), t),
            plane: None,
        })
        .collect();
    let (lo, hi) = solid
        .bounds()
        .map(|(l, h)| moved_box(l, h, t))
        .ok_or_else(|| unsupported("moving an empty body"))?;
    let mut s = Sampled {
        to_local: t.inverse(),
        shape: Shape::Image {
            solid: Arc::new(solid.clone()),
        },
        surfaces,
        lo,
        hi,
        exact: Vec::new(),
    };
    s.recompute_exact();
    Ok(s)
}

/// The thin wall's inner side of a straight curve: the line moved `inside` into the profile.
pub(super) fn inner_line(c: &Curve2, inside: f64) -> Option<[f64; 2]> {
    let (a, b) = c.line()?;
    let d = [b[0] - a[0], b[1] - a[1]];
    let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-300);
    // Inward is left of a counter-clockwise outer loop.
    Some([a[0] - d[1] / l * inside, a[1] + d[0] / l * inside])
}

/// An extrusion: a prism whose caps follow end conditions, with draft and thin options.
/// `labels` are the side labels per loop and curve; `inner` the thin inner-wall labels.
#[allow(clippy::too_many_arguments)]
pub(in crate::regen::fake) fn extrusion(
    to_world: Transform,
    loops: &[Vec<(Curve2, HistorySource)>],
    inner: Option<&[Vec<HistorySource>]>,
    start: ZBound,
    end: ZBound,
    cap_labels: [HistorySource; 2],
    draft: Option<(f64, f64, f64)>,
    thin: Option<ThinBand<HistorySource>>,
) -> Result<Sampled, KernelError> {
    let mut surfaces = Vec::new();
    let mut l2 = Vec::new();
    let end_sign = match (start.fixed(), end.fixed()) {
        (Some(a), Some(b)) => (b - a).signum(),
        _ => 1.0,
    };
    for (li, l) in loops.iter().enumerate() {
        let mut v = Vec::new();
        for (ci, (c, lab)) in l.iter().enumerate() {
            let o = surfaces.len();
            let hint = match (draft, c) {
                (None, _) => side_hint(c, &to_world),
                (Some((h, te, ts)), Curve2::Line { a, b }) => {
                    let one_side = start
                        .fixed()
                        .zip(end.fixed())
                        .map(|(zs, ze)| (zs - h) * (ze - h) >= -1e-15);
                    match one_side {
                        Some(true) => {
                            let d = [b[0] - a[0], b[1] - a[1]];
                            let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-300);
                            let t = if end
                                .fixed()
                                .map(|z| (z - h) * end_sign >= 0.0)
                                .unwrap_or(true)
                            {
                                te
                            } else {
                                -ts
                            };
                            plane_hint([d[1] / l, -d[0] / l, t * end_sign], &to_world)
                        }
                        _ => SurfaceKind::Other,
                    }
                }
                (Some(_), _) => SurfaceKind::Other,
            };
            surfaces.push(Surf {
                label: lab.clone(),
                hint,
                plane: None,
            });
            let inn = match (inner, thin.as_ref()) {
                (Some(labels), Some(_)) => {
                    let k = surfaces.len();
                    let mut hint = side_hint(c, &to_world);
                    if let SurfaceKind::Plane { normal } = hint {
                        hint = SurfaceKind::Plane {
                            normal: [-normal[0], -normal[1], -normal[2]],
                        };
                    }
                    if draft.is_some() {
                        hint = SurfaceKind::Other;
                    }
                    surfaces.push(Surf {
                        label: labels[li][ci].clone(),
                        hint,
                        plane: None,
                    });
                    Some(k)
                }
                _ => None,
            };
            v.push((c.clone(), o, inn));
        }
        l2.push(v);
    }
    let caps = [surfaces.len(), surfaces.len() + 1];
    let [c0, c1] = cap_labels;
    let cap_hint = |b: &ZBound, n: f64| match b {
        ZBound::At(_) => plane_hint([0.0, 0.0, n], &to_world),
        _ => SurfaceKind::Other,
    };
    surfaces.push(Surf {
        label: c0,
        hint: cap_hint(&start, -end_sign),
        plane: None,
    });
    surfaces.push(Surf {
        label: c1,
        hint: cap_hint(&end, end_sign),
        plane: None,
    });
    let thin_spec = match thin {
        None => None,
        Some((i, o, None)) => Some((i, o, None)),
        Some((i, o, Some((cap, [a, b])))) => {
            let k = surfaces.len();
            surfaces.push(Surf {
                label: a,
                hint: plane_hint([0.0, 0.0, end_sign], &to_world),
                plane: None,
            });
            surfaces.push(Surf {
                label: b,
                hint: plane_hint([0.0, 0.0, -end_sign], &to_world),
                plane: None,
            });
            Some((i, o, Some((cap, [k, k + 1]))))
        }
    };
    let counts: Vec<Vec<usize>> = l2
        .iter()
        .map(|l| l.iter().map(|(c, _, _)| c.samples()).collect())
        .collect();
    let curve_loops: Vec<Vec<(Curve2, usize)>> = l2
        .iter()
        .map(|l| l.iter().map(|(c, o, _)| (c.clone(), *o)).collect())
        .collect();
    let polys = loop_polys(&curve_loops, &counts);
    let x = Extrusion {
        loops: l2,
        polys,
        start,
        end,
        caps,
        end_sign,
        draft,
        thin: thin_spec,
    };
    // The extent: caps over sample points of the profile, sides grown by any outward wall.
    let mut plo = [f64::INFINITY; 2];
    let mut phi = [f64::NEG_INFINITY; 2];
    for q in x.polys.iter().flatten() {
        for a in 0..2 {
            plo[a] = plo[a].min(q[a]);
            phi[a] = phi[a].max(q[a]);
        }
    }
    let mut samples: Vec<[f64; 2]> = x.polys.iter().flatten().copied().collect();
    for i in 0..=8 {
        for j in 0..=8 {
            let q = [
                plo[0] + (phi[0] - plo[0]) * i as f64 / 8.0,
                plo[1] + (phi[1] - plo[1]) * j as f64 / 8.0,
            ];
            if inside_polys(&x.polys, q) {
                samples.push(q);
            }
        }
    }
    let (mut zlo, mut zhi) = (f64::INFINITY, f64::NEG_INFINITY);
    for q in &samples {
        // Boundary points sit exactly on the profile; nudge them inside for the ray.
        let Some((a, b)) = x.caps_at(*q, &to_world) else {
            if inside_polys(&x.polys, *q) {
                return Err(KernelError::Failed {
                    message: "no face the extrude can end at intercepts the whole profile".into(),
                });
            }
            continue;
        };
        zlo = zlo.min(a.min(b));
        zhi = zhi.max(a.max(b));
    }
    if !zlo.is_finite() || zhi - zlo <= 1e-12 {
        return Err(KernelError::Failed {
            message: "the extrude has no length".into(),
        });
    }
    let mut grow = x.thin.map(|t| t.1).unwrap_or(0.0);
    if let Some((h, te, ts)) = x.draft {
        let reach = (zhi - h).abs().max((zlo - h).abs());
        grow += reach * (-te).max(-ts).max(0.0);
    }
    let pts: Vec<[f64; 3]> = [
        [plo[0] - grow, plo[1] - grow],
        [phi[0] + grow, plo[1] - grow],
        [plo[0] - grow, phi[1] + grow],
        [phi[0] + grow, phi[1] + grow],
    ]
    .iter()
    .flat_map(|q| [[q[0], q[1], zlo], [q[0], q[1], zhi]])
    .map(|p| to_world.apply(p))
    .collect();
    let (lo, hi) = bbox(pts);
    let mut out = Sampled {
        to_local: to_world.inverse(),
        shape: Shape::Extrusion(Box::new(x)),
        surfaces,
        lo,
        hi,
        exact: Vec::new(),
    };
    out.recompute_exact();
    Ok(out)
}
