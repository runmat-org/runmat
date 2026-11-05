//! Surface bodies on the grid test kernel: extruded, revolved, swept, lofted, filled, offset,
//! trimmed and knit sheets.

use super::*;

impl FakeKernel {
    pub(super) fn surface(
        &self,
        op: &SurfaceOp,
        operands: &[&BodyHandle],
    ) -> Result<KernelOutput, KernelError> {
        let kernel = Self::id_static();
        let keep_all = |n: usize| -> Vec<(SheetBuild, Vec<HistorySource>)> {
            (0..n)
                .filter_map(|i| {
                    operands[i].downcast::<FakeSheet>().map(|s| {
                        (
                            s.as_input(i as u32),
                            vec![HistorySource::Input {
                                from: OperandEntity {
                                    operand: i as u32,
                                    entity: 0,
                                },
                            }],
                        )
                    })
                })
                .collect()
        };
        match op {
            SurfaceOp::Planar { patches } => {
                sheet::output(&kernel, sheet::planar(patches), Vec::new(), Vec::new())
            }
            SurfaceOp::Swept { tool } => {
                let b = sheet::swept(tool)?;
                let body = tool
                    .body
                    .as_ref()
                    .map(|l| gen(l.role, l.tag.clone(), l.ordinal))
                    .unwrap_or_else(|| gen(Role::Body, KernelTag::None, None));
                sheet::output(&kernel, vec![(b, vec![body])], Vec::new(), Vec::new())
            }
            SurfaceOp::Offset { faces, distance } => {
                // Operands are kept; one new sheet per offset operand.
                let mut builds = keep_all(operands.len());
                let mut solids_kept = Vec::new();
                for (i, h) in operands.iter().enumerate() {
                    let chosen: BTreeSet<u32> = faces
                        .iter()
                        .filter(|e| e.operand as usize == i)
                        .map(|e| e.entity)
                        .collect();
                    let mut b = SheetBuild {
                        facets: Vec::new(),
                        face_of: Vec::new(),
                        face_sources: Vec::new(),
                        hints: Vec::new(),
                    };
                    let add_face = |b: &mut SheetBuild, fs: Vec<Facet>, f: u32| {
                        let face = b.face_sources.len() as u32;
                        b.face_sources.push(vec![gen(
                            Role::OffsetFace,
                            KernelTag::InputFace {
                                face: OperandEntity {
                                    operand: i as u32,
                                    entity: f,
                                },
                            },
                            None,
                        )]);
                        b.hints.push(match fs.first() {
                            Some(x) if fs.iter().all(|y| y.normal == x.normal) => {
                                SurfaceKind::Plane { normal: x.normal }
                            }
                            _ => SurfaceKind::Other,
                        });
                        for x in fs {
                            b.facets.push(x);
                            b.face_of.push(face);
                        }
                    };
                    if let Some(s) = h.downcast::<FakeSheet>() {
                        let base = s.as_input(i as u32);
                        let nf = s.topology.faces.len() as u32;
                        let all: Vec<Facet> = base.facets.clone();
                        let moved = sheet::offset_facets(&all, *distance);
                        for f in 0..nf {
                            if !chosen.is_empty() && !chosen.contains(&f) {
                                continue;
                            }
                            let fs: Vec<Facet> = moved
                                .iter()
                                .zip(&base.face_of)
                                .filter(|(_, k)| **k == f)
                                .map(|(x, _)| relabel_boundaries(x, Role::OffsetFace))
                                .collect();
                            add_face(&mut b, fs, f);
                        }
                    } else if let Some(so) = h.downcast::<FakeSolid>() {
                        solids_kept.push(i);
                        let sel: Vec<u32> = if chosen.is_empty() {
                            (0..so.faces.len() as u32).collect()
                        } else {
                            chosen.iter().copied().collect()
                        };
                        let selected: BTreeSet<u32> = sel.iter().copied().collect();
                        let mut all = Vec::new();
                        let mut owner = Vec::new();
                        for f in &sel {
                            let mut fs = sheet::solid_face_facets(
                                so,
                                *f,
                                gen(Role::Boundary, KernelTag::None, None),
                            )?;
                            sheet::label_by_neighbour(
                                so,
                                &selected,
                                &mut fs,
                                Role::Boundary,
                                i as u32,
                            );
                            owner.extend(std::iter::repeat_n(*f, fs.len()));
                            all.extend(fs);
                        }
                        let moved = sheet::offset_facets(&all, *distance);
                        for f in &sel {
                            let fs: Vec<Facet> = moved
                                .iter()
                                .zip(&owner)
                                .filter(|(_, k)| **k == *f)
                                .map(|(x, _)| x.clone())
                                .collect();
                            add_face(&mut b, fs, *f);
                        }
                    } else {
                        return Err(KernelError::ForeignOperand { operand: i as u32 });
                    }
                    if !b.facets.is_empty() {
                        builds.push((b, vec![gen(Role::Body, KernelTag::None, None)]));
                    }
                }
                // Solid operands pass through unchanged.
                let mut extra = Vec::new();
                for i in solids_kept {
                    let so = operands[i].downcast::<FakeSolid>().expect("solid");
                    let faces = (0..so.faces.len() as u32)
                        .map(|f| {
                            vec![HistorySource::Input {
                                from: OperandEntity {
                                    operand: i as u32,
                                    entity: f,
                                },
                            }]
                        })
                        .collect();
                    extra.push((
                        KernelBody {
                            handle: operands[i].clone(),
                            topology: so.topology.clone(),
                        },
                        faces,
                        vec![HistorySource::Input {
                            from: OperandEntity {
                                operand: i as u32,
                                entity: 0,
                            },
                        }],
                    ));
                }
                sheet::output(&kernel, builds, extra, Vec::new())
            }
            SurfaceOp::Knit { solid } => {
                let mut merged = SheetBuild {
                    facets: Vec::new(),
                    face_of: Vec::new(),
                    face_sources: Vec::new(),
                    hints: Vec::new(),
                };
                let mut body_sources = Vec::new();
                for (i, h) in operands.iter().enumerate() {
                    let s = h
                        .downcast::<FakeSheet>()
                        .ok_or(KernelError::ForeignOperand { operand: i as u32 })?;
                    let b = s.as_input(i as u32);
                    let base = merged.face_sources.len() as u32;
                    merged.facets.extend(b.facets);
                    merged.face_of.extend(b.face_of.iter().map(|f| f + base));
                    merged.face_sources.extend(b.face_sources);
                    merged.hints.extend(b.hints);
                    body_sources.push(HistorySource::Input {
                        from: OperandEntity {
                            operand: i as u32,
                            entity: 0,
                        },
                    });
                }
                let (probe, _) = FakeSheet::build(SheetBuild {
                    facets: merged.facets.clone(),
                    face_of: merged.face_of.clone(),
                    face_sources: merged.face_sources.clone(),
                    hints: merged.hints.clone(),
                });
                let open = probe.topology.edges.iter().any(|e| e.laminar);
                if *solid {
                    if open {
                        return Err(KernelError::Failed {
                            message: "the knitted sheets are not watertight; no solid".into(),
                        });
                    }
                    let polys: Vec<(Facet, HistorySource)> = merged
                        .facets
                        .iter()
                        .zip(&merged.face_of)
                        .map(|(f, k)| (f.clone(), merged.face_sources[*k as usize][0].clone()))
                        .collect();
                    let mut job = grid_job(
                        Vec::new(),
                        Vec::new(),
                        vec![Tool::Polys(Box::new(PolyTool::new(polys)))],
                        Mode::New,
                    );
                    job.body_label = Some(vec![body_sources[0].clone()]);
                    let res = run_grid(job)?;
                    let mut out = self.finish(&[], &[], res, Vec::new(), Vec::new())?;
                    if let Some(b) = out.history.bodies.first_mut() {
                        b.sources = body_sources;
                    }
                    return Ok(out);
                }
                sheet::output(
                    &kernel,
                    vec![(merged, body_sources)],
                    Vec::new(),
                    Vec::new(),
                )
            }
            SurfaceOp::Thicken { distance, both } => {
                let mut tools = Vec::new();
                let mut labels = Vec::new();
                for (i, h) in operands.iter().enumerate() {
                    let s = h
                        .downcast::<FakeSheet>()
                        .ok_or(KernelError::ForeignOperand { operand: i as u32 })?;
                    let (d0, d1) = if *both {
                        (-distance / 2.0, distance / 2.0)
                    } else {
                        (0.0, *distance)
                    };
                    let face_src = |f: u32| HistorySource::Input {
                        from: OperandEntity {
                            operand: i as u32,
                            entity: f,
                        },
                    };
                    let far = |f: u32| {
                        gen(
                            Role::ThickenFace,
                            KernelTag::InputFace {
                                face: OperandEntity {
                                    operand: i as u32,
                                    entity: f,
                                },
                            },
                            None,
                        )
                    };
                    let near_facets = sheet::offset_facets(&s.facets, d0);
                    let far_facets = sheet::offset_facets(&s.facets, d1);
                    let mut polys: Vec<(Facet, HistorySource)> = Vec::new();
                    // Centred: the side along +normal continues the sheet's faces; one-sided:
                    // the near side does.
                    let lab_near = |f: u32| if *both { far(f) } else { face_src(f) };
                    let lab_far = |f: u32| if *both { face_src(f) } else { far(f) };

                    for (k, f) in near_facets.iter().enumerate() {
                        polys.push((f.clone(), lab_near(s.face_of[k])));
                    }
                    for (k, f) in far_facets.iter().enumerate() {
                        polys.push((f.clone(), lab_far(s.face_of[k])));
                    }
                    // Rims along boundary edges.
                    for (e, segs) in &s.boundary_segments {
                        for &(fi, li, si) in segs {
                            let l0 = &near_facets[fi].loops[li];
                            let l1 = &far_facets[fi].loops[li];
                            let n = l0.len();
                            let q = vec![l0[si], l0[(si + 1) % n], l1[(si + 1) % n], l1[si]];
                            let nrm = crate::model::datum::normalize(crate::model::datum::cross(
                                crate::model::datum::sub(q[1], q[0]),
                                crate::model::datum::sub(q[3], q[0]),
                            ));
                            polys.push((
                                Facet {
                                    loops: vec![q],
                                    normal: nrm,
                                    labels: vec![vec![
                                        gen(Role::Boundary, KernelTag::None, None);
                                        4
                                    ]],
                                },
                                gen(
                                    Role::ThickenRim,
                                    KernelTag::InputEdge {
                                        edge: OperandEntity {
                                            operand: i as u32,
                                            entity: *e,
                                        },
                                    },
                                    None,
                                ),
                            ));
                        }
                    }
                    tools.push(Tool::Polys(Box::new(PolyTool::new(polys))));
                    labels.push(HistorySource::Input {
                        from: OperandEntity {
                            operand: i as u32,
                            entity: 0,
                        },
                    });
                }
                let mut job = grid_job(Vec::new(), Vec::new(), tools, Mode::New);
                job.separate_tools = true;
                job.body_label = Some(labels);
                let res = run_grid(job)?;
                self.finish(&[], &[], res, Vec::new(), Vec::new())
            }
            SurfaceOp::Trim { frame, keep } => {
                let mut builds = Vec::new();
                for (i, h) in operands.iter().enumerate() {
                    let s = h
                        .downcast::<FakeSheet>()
                        .ok_or(KernelError::ForeignOperand { operand: i as u32 })?;
                    let b = s.as_input(i as u32);
                    let cut = gen(Role::Boundary, KernelTag::Side { side: *keep }, None);
                    let mut nb = SheetBuild {
                        facets: Vec::new(),
                        face_of: Vec::new(),
                        face_sources: b.face_sources.clone(),
                        hints: b.hints.clone(),
                    };
                    for (k, f) in b.facets.iter().enumerate() {
                        for x in sheet::clip(std::slice::from_ref(f), frame, *keep, cut.clone()) {
                            nb.facets.push(x);
                            nb.face_of.push(b.face_of[k]);
                        }
                    }
                    // Faces trimmed away entirely have no image.
                    let alive: BTreeSet<u32> = nb.face_of.iter().copied().collect();
                    let mut remap = BTreeMap::new();
                    let mut sources = Vec::new();
                    let mut hints = Vec::new();
                    for f in 0..nb.face_sources.len() as u32 {
                        if alive.contains(&f) {
                            remap.insert(f, sources.len() as u32);
                            sources.push(nb.face_sources[f as usize].clone());
                            hints.push(nb.hints[f as usize]);
                        }
                    }
                    nb.face_of = nb.face_of.iter().map(|f| remap[f]).collect();
                    nb.face_sources = sources;
                    nb.hints = hints;
                    if !nb.facets.is_empty() {
                        builds.push((
                            nb,
                            vec![HistorySource::Input {
                                from: OperandEntity {
                                    operand: i as u32,
                                    entity: 0,
                                },
                            }],
                        ));
                    }
                }
                let mut out = sheet::output(&kernel, builds, Vec::new(), Vec::new())?;
                fill_deleted(&mut out, operands);
                Ok(out)
            }
            SurfaceOp::Extend { edges, distance } => {
                let mut builds = Vec::new();
                for (i, h) in operands.iter().enumerate() {
                    let s = h
                        .downcast::<FakeSheet>()
                        .ok_or(KernelError::ForeignOperand { operand: i as u32 })?;
                    let mut b = s.as_input(i as u32);
                    for e in edges.iter().filter(|e| e.operand as usize == i) {
                        let segs = s.boundary_segments.get(&e.entity).ok_or_else(|| {
                            KernelError::Failed {
                                message: "extend needs boundary edges".into(),
                            }
                        })?;
                        for &(fi, li, si) in segs {
                            let f = &s.facets[fi];
                            let l = &f.loops[li];
                            let (a, c) = (l[si], l[(si + 1) % l.len()]);
                            let along = crate::model::datum::sub(c, a);
                            // Outward in the facet's plane (loops run counter-clockwise about
                            // the normal for the outer loop).
                            let out_dir = crate::model::datum::normalize(
                                crate::model::datum::cross(along, f.normal),
                            );
                            let w = crate::model::datum::scale(out_dir, *distance);
                            let q = vec![
                                a,
                                crate::model::datum::add(a, w),
                                crate::model::datum::add(c, w),
                                c,
                            ];
                            let strip_label =
                                gen(Role::Extension, KernelTag::InputEdge { edge: *e }, None);
                            b.facets.push(Facet {
                                loops: vec![q],
                                normal: f.normal,
                                labels: vec![vec![
                                    strip_label.clone(),
                                    strip_label.clone(),
                                    strip_label.clone(),
                                    strip_label,
                                ]],
                            });
                            b.face_of.push(s.face_of[fi]);
                        }
                    }
                    builds.push((
                        b,
                        vec![HistorySource::Input {
                            from: OperandEntity {
                                operand: i as u32,
                                entity: 0,
                            },
                        }],
                    ));
                }
                sheet::output(&kernel, builds, Vec::new(), Vec::new())
            }
            SurfaceOp::DeleteHole { edges } => {
                let mut builds = Vec::new();
                let mut extra = Vec::new();
                for (i, h) in operands.iter().enumerate() {
                    let mine: Vec<u32> = edges
                        .iter()
                        .filter(|e| e.operand as usize == i)
                        .map(|e| e.entity)
                        .collect();
                    let Some(s) = h.downcast::<FakeSheet>() else {
                        if mine.is_empty() {
                            extra.push(kept_solid(operands, i)?);
                            continue;
                        }
                        return Err(KernelError::Failed {
                            message: "delete hole works on sheets".into(),
                        });
                    };
                    let mut b = s.as_input(i as u32);
                    // Inner loops of a facet close by dropping the loop; a hole whose rim runs
                    // over several facets of one face gets a patch on that face.
                    let mut drop: BTreeSet<(usize, usize)> = BTreeSet::new();
                    let mut loose = Vec::new();
                    for e in &mine {
                        let segs =
                            s.boundary_segments
                                .get(e)
                                .ok_or_else(|| KernelError::Failed {
                                    message: "delete hole needs boundary edges".into(),
                                })?;
                        for &(fi, li, si) in segs {
                            if li > 0 {
                                drop.insert((fi, li));
                            } else {
                                let l = &s.facets[fi].loops[li];
                                loose.push((l[si], l[(si + 1) % l.len()], s.face_of[fi]));
                            }
                        }
                    }
                    for &(fi, li) in drop.iter().rev() {
                        b.facets[fi].loops.remove(li);
                        b.facets[fi].labels.remove(li);
                    }
                    for lp in sheet::chain_loops(loose)? {
                        let face = lp[0].1;
                        if lp.iter().any(|(_, f)| *f != face) {
                            return Err(KernelError::Failed {
                                message: "the hole's rim runs over more than one face".into(),
                            });
                        }
                        let inner = gen(Role::SurfacePatch, KernelTag::None, None);
                        let tagged: Vec<_> = lp.iter().map(|(p, _)| (*p, inner.clone())).collect();
                        // The rim was traced round the hole; the patch faces the other way.
                        let rev: Vec<_> = tagged.into_iter().rev().collect();
                        for f in sheet::patch_facets(&rev, &inner) {
                            b.facets.push(f);
                            b.face_of.push(face);
                        }
                    }
                    builds.push((
                        b,
                        vec![HistorySource::Input {
                            from: OperandEntity {
                                operand: i as u32,
                                entity: 0,
                            },
                        }],
                    ));
                }
                sheet::output(&kernel, builds, extra, Vec::new())
            }
            SurfaceOp::Fill { edges } => {
                if edges
                    .iter()
                    .any(|(_, c)| *c != crate::model::surface::Continuity::G0)
                {
                    return Err(KernelError::NotSupported {
                        what: "fake kernel: fill surfaces tangent or curvature continuous \
                               with their faces"
                            .into(),
                    });
                }
                let mut segs = Vec::new();
                for (e, _) in edges {
                    let h = operands
                        .get(e.operand as usize)
                        .ok_or(KernelError::Failed {
                            message: "edge operand out of range".into(),
                        })?;
                    let label = gen(Role::Boundary, KernelTag::InputEdge { edge: *e }, None);
                    for (a, c) in sheet::edge_segments(h, e.entity)? {
                        segs.push((a, c, label.clone()));
                    }
                }
                let loops = sheet::chain_loops(segs)?;
                if loops.len() != 1 {
                    return Err(KernelError::Failed {
                        message: "a fill needs one closed loop of edges".into(),
                    });
                }
                let inner = gen(Role::SurfacePatch, KernelTag::None, Some(1));
                let facets = sheet::patch_facets(&loops[0], &inner);
                let hint = match facets.as_slice() {
                    [f] => SurfaceKind::Plane { normal: f.normal },
                    _ => SurfaceKind::Other,
                };
                let b = SheetBuild {
                    face_of: vec![0; facets.len()],
                    facets,
                    face_sources: vec![vec![gen(Role::SurfacePatch, KernelTag::None, None)]],
                    hints: vec![hint],
                };
                let mut builds = keep_all(operands.len());
                builds.push((b, vec![gen(Role::Body, KernelTag::None, None)]));
                let extra = (0..operands.len())
                    .filter(|i| operands[*i].downcast::<FakeSolid>().is_some())
                    .map(|i| kept_solid(operands, i))
                    .collect::<Result<Vec<_>, _>>()?;
                sheet::output(&kernel, builds, extra, Vec::new())
            }
            SurfaceOp::Ruled {
                edges,
                distance,
                angle,
            } => {
                let mut b = SheetBuild {
                    facets: Vec::new(),
                    face_of: Vec::new(),
                    face_sources: Vec::new(),
                    hints: Vec::new(),
                };
                for e in edges {
                    let h = operands
                        .get(e.operand as usize)
                        .ok_or(KernelError::Failed {
                            message: "edge operand out of range".into(),
                        })?;
                    let s = h
                        .downcast::<FakeSheet>()
                        .ok_or_else(|| KernelError::NotSupported {
                            what: "fake kernel: ruled surfaces along solid edges".into(),
                        })?;
                    let segs =
                        s.boundary_segments
                            .get(&e.entity)
                            .ok_or_else(|| KernelError::Failed {
                                message: "a ruled surface runs along boundary edges".into(),
                            })?;
                    let face = b.face_sources.len() as u32;
                    b.face_sources.push(vec![gen(
                        Role::Ruled,
                        KernelTag::InputEdge { edge: *e },
                        None,
                    )]);
                    let edge_label =
                        |k: u32| gen(Role::Boundary, KernelTag::InputEdge { edge: *e }, Some(k));
                    let mut normals = Vec::new();
                    for &(fi, li, si) in segs {
                        use crate::model::datum::{add, cross, normalize, scale, sub};
                        let f = &s.facets[fi];
                        let l = &f.loops[li];
                        let (a, c) = (l[si], l[(si + 1) % l.len()]);
                        // Outward from the face across the edge, in the face's plane.
                        let out_dir = normalize(cross(sub(c, a), f.normal));
                        let dir = add(scale(f.normal, angle.cos()), scale(out_dir, angle.sin()));
                        let w = scale(dir, *distance);
                        let q = vec![c, a, add(a, w), add(c, w)];
                        let n = normalize(cross(sub(q[1], q[0]), sub(q[3], q[0])));
                        normals.push(n);
                        b.facets.push(Facet {
                            loops: vec![q],
                            normal: n,
                            labels: vec![vec![
                                edge_label(0),
                                edge_label(2),
                                edge_label(1),
                                edge_label(3),
                            ]],
                        });
                        b.face_of.push(face);
                    }
                    b.hints.push(match normals.first() {
                        Some(n0)
                            if normals
                                .iter()
                                .all(|n| (0..3).all(|k| (n[k] - n0[k]).abs() < 1e-9)) =>
                        {
                            SurfaceKind::Plane { normal: *n0 }
                        }
                        _ => SurfaceKind::Other,
                    });
                }
                let mut builds = keep_all(operands.len());
                builds.push((b, vec![gen(Role::Body, KernelTag::None, None)]));
                let extra = (0..operands.len())
                    .filter(|i| operands[*i].downcast::<FakeSolid>().is_some())
                    .map(|i| kept_solid(operands, i))
                    .collect::<Result<Vec<_>, _>>()?;
                sheet::output(&kernel, builds, extra, Vec::new())
            }
            SurfaceOp::CutWithSurface { keep } => {
                let so = Self::solid(operands, 0)?;
                let sh = operands
                    .get(1)
                    .and_then(|h| h.downcast::<FakeSheet>())
                    .ok_or(KernelError::Failed {
                        message: "cut with surface needs a solid and a sheet".into(),
                    })?;
                let f0 = sh.facets.first().ok_or(KernelError::Failed {
                    message: "the surface is empty".into(),
                })?;
                let (n, o) = (f0.normal, f0.loops[0][0]);
                let on_plane = |p: [f64; 3]| {
                    ((p[0] - o[0]) * n[0] + (p[1] - o[1]) * n[1] + (p[2] - o[2]) * n[2]).abs()
                        < 1e-9
                };
                let planar = sh.facets.iter().all(|f| {
                    (0..3).all(|k| (f.normal[k] - n[k]).abs() < 1e-9)
                        && f.loops.iter().flatten().all(|p| on_plane(*p))
                });
                if !planar {
                    return Err(KernelError::NotSupported {
                        what: "fake kernel: cutting with a curved surface".into(),
                    });
                }
                let (lo, hi) = so.bounds().ok_or(KernelError::Failed {
                    message: "the body is empty".into(),
                })?;
                // The surface must reach across the body: every bounding corner, dropped onto
                // its plane, lies on it.
                for k in 0..8 {
                    let c = [
                        if k & 1 == 0 { lo[0] } else { hi[0] },
                        if k & 2 == 0 { lo[1] } else { hi[1] },
                        if k & 4 == 0 { lo[2] } else { hi[2] },
                    ];
                    let d = (c[0] - o[0]) * n[0] + (c[1] - o[1]) * n[1] + (c[2] - o[2]) * n[2];
                    let p = [c[0] - d * n[0], c[1] - d * n[1], c[2] - d * n[2]];
                    if !sh.facets.iter().any(|f| f.covers(p)) {
                        return Err(KernelError::Failed {
                            message: "the surface does not cut through the body".into(),
                        });
                    }
                }
                // Remove the other side: the body's box clipped to it.
                let away = match keep {
                    PlaneSide::Positive => n,
                    PlaneSide::Negative => [-n[0], -n[1], -n[2]],
                };
                let label = gen(Role::SplitFace, KernelTag::Side { side: *keep }, None);
                let pad = 1e-3;
                let boxed = Prism {
                    k: 2,
                    u: 0,
                    v: 1,
                    loops: vec![vec![
                        [lo[0] - pad, lo[1] - pad],
                        [hi[0] + pad, lo[1] - pad],
                        [hi[0] + pad, hi[1] + pad],
                        [lo[0] - pad, hi[1] + pad],
                    ]],
                    edge_labels: vec![vec![label.clone(); 4]],
                    lo: lo[2] - pad,
                    hi: hi[2] + pad,
                    lo_label: label.clone(),
                    hi_label: label.clone(),
                };
                let tool = Tool::Clipped(Box::new(Tool::Exact(boxed)), vec![(o, away, label)]);
                let res = run_grid(grid_job(vec![so], vec![0], vec![tool], Mode::Cut))?;
                let cut = self.finish(&[so], &[0], res, Vec::new(), Vec::new())?;
                let kept = sheet::output(
                    &kernel,
                    vec![(
                        sh.as_input(1),
                        vec![HistorySource::Input {
                            from: OperandEntity {
                                operand: 1,
                                entity: 0,
                            },
                        }],
                    )],
                    Vec::new(),
                    Vec::new(),
                )?;
                Ok(super::features::concat(vec![cut, kept]))
            }
            SurfaceOp::MidSurface { pairs } => {
                let so = Self::solid(operands, 0)?;
                let mut b = SheetBuild {
                    facets: Vec::new(),
                    face_of: Vec::new(),
                    face_sources: Vec::new(),
                    hints: Vec::new(),
                };
                let selected: BTreeSet<u32> = pairs
                    .iter()
                    .flat_map(|(a, c)| [a.entity, c.entity])
                    .collect();
                for (a, c) in pairs {
                    let (fa, fc) = (
                        so.faces.get(a.entity as usize),
                        so.faces.get(c.entity as usize),
                    );
                    let (Some(fa), Some(fc)) = (fa, fc) else {
                        return Err(KernelError::Failed {
                            message: "face out of range".into(),
                        });
                    };
                    let (Some(pa), Some(pc)) = (fa.plane, fc.plane) else {
                        return Err(KernelError::NotSupported {
                            what: "fake kernel: mid-surfaces of curved faces".into(),
                        });
                    };
                    if pa.axis != pc.axis || pa.dir == pc.dir {
                        return Err(KernelError::Failed {
                            message: "a mid-surface pair must be opposite parallel faces".into(),
                        });
                    }
                    let mid = (pa.offset + pc.offset) / 2.0;
                    let face = b.face_sources.len() as u32;
                    b.face_sources.push(vec![gen(
                        Role::MidSurface,
                        KernelTag::InputFace { face: *a },
                        None,
                    )]);
                    let mut n = [0.0; 3];
                    n[pa.axis] = pa.dir as f64;
                    b.hints.push(SurfaceKind::Plane { normal: n });
                    for ra in &fa.rects {
                        for rc in &fc.rects {
                            let mut lo = [0.0; 3];
                            let mut hi = [0.0; 3];
                            let mut ok = true;
                            for x in 0..3 {
                                if x == pa.axis {
                                    lo[x] = mid;
                                    hi[x] = mid;
                                } else {
                                    lo[x] = ra.lo[x].max(rc.lo[x]);
                                    hi[x] = ra.hi[x].min(rc.hi[x]);
                                    ok &= hi[x] > lo[x] + EPS;
                                }
                            }
                            if ok {
                                let pk = PlaneKey {
                                    axis: pa.axis,
                                    dir: pa.dir,
                                    offset: mid,
                                };
                                let mut fs = vec![sheet_rect(pk, Rect { lo, hi })];
                                // Label boundaries by the face across, measured on face `a`.
                                for f in &mut fs {
                                    for l in &mut f.loops {
                                        for p in l.iter_mut() {
                                            p[pa.axis] = pa.offset;
                                        }
                                    }
                                }
                                sheet::label_by_neighbour(
                                    so,
                                    &selected,
                                    &mut fs,
                                    Role::Boundary,
                                    0,
                                );
                                for f in &mut fs {
                                    for l in &mut f.loops {
                                        for p in l.iter_mut() {
                                            p[pa.axis] = mid;
                                        }
                                    }
                                }
                                for x in fs {
                                    b.facets.push(x);
                                    b.face_of.push(face);
                                }
                            }
                        }
                    }
                }
                // The solid stays; the mid-surface is a new sheet.
                let faces = (0..so.faces.len() as u32)
                    .map(|f| {
                        vec![HistorySource::Input {
                            from: OperandEntity {
                                operand: 0,
                                entity: f,
                            },
                        }]
                    })
                    .collect();
                sheet::output(
                    &kernel,
                    vec![(b, vec![gen(Role::Body, KernelTag::None, None)])],
                    vec![(
                        KernelBody {
                            handle: operands[0].clone(),
                            topology: so.topology.clone(),
                        },
                        faces,
                        vec![HistorySource::Input {
                            from: OperandEntity {
                                operand: 0,
                                entity: 0,
                            },
                        }],
                    )],
                    Vec::new(),
                )
            }
        }
    }
}
