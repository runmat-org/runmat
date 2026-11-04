//! Fake-kernel coverage for labelled tools (sheet metal, weldments, ribs, embosses, form
//! tools), direct edits (move, offset, rotate, delete and replace faces), scaling and surface
//! bodies.

use super::sheet::{self, Facet, FakeSheet, PolyTool, SheetBuild};
use super::*;

fn grid_job<'a>(
    operands: Vec<&'a FakeSolid>,
    operand_ids: Vec<u32>,
    tools: Vec<Tool>,
    mode: Mode,
) -> GridJob<'a> {
    GridJob {
        operands,
        operand_ids,
        tools,
        mode,
        split: None,
        separate_tools: false,
        body_label: None,
        modifier: None,
        merge_collinear: false,
        carve: Vec::new(),
    }
}

/// A solid operand passed through an operation unchanged.
fn kept_solid(operands: &[&BodyHandle], i: usize) -> Result<sheet::ExtraBody, KernelError> {
    let so = operands[i]
        .downcast::<FakeSolid>()
        .ok_or(KernelError::ForeignOperand { operand: i as u32 })?;
    let input = |entity: u32| {
        vec![HistorySource::Input {
            from: OperandEntity {
                operand: i as u32,
                entity,
            },
        }]
    };
    Ok((
        KernelBody {
            handle: operands[i].clone(),
            topology: so.topology.clone(),
        },
        (0..so.faces.len() as u32).map(input).collect(),
        input(0),
    ))
}

impl FakeKernel {
    /// Operations handled here; `None` for the rest.
    pub(super) fn execute_extra(
        &self,
        op: &KernelOp,
        operands: &[&BodyHandle],
    ) -> Option<Result<KernelOutput, KernelError>> {
        Some(match op {
            KernelOp::Labeled {
                tools,
                carve,
                combine,
                separate,
            } => self.labeled(tools, carve, *combine, *separate, operands),
            KernelOp::MoveFaces { faces, motion } => self.move_faces(faces, motion, operands),
            KernelOp::DeleteFaces { faces, heal } => self.delete_faces(faces, *heal, operands),
            KernelOp::ReplaceFaces { faces, with } => self.replace_faces(faces, *with, operands),
            KernelOp::Scale { center, factors } => self.scale(*center, *factors, operands),
            KernelOp::Surface(s) => self.surface(s, operands),
            _ => return None,
        })
    }

    fn solids<'a>(operands: &[&'a BodyHandle]) -> Result<Vec<&'a FakeSolid>, KernelError> {
        (0..operands.len())
            .map(|i| Self::solid(operands, i))
            .collect()
    }

    fn labeled(
        &self,
        tools: &[LabeledTool],
        carve: &[LabeledTool],
        combine: Combine,
        separate: bool,
        operands: &[&BodyHandle],
    ) -> Result<KernelOutput, KernelError> {
        let solids = Self::solids(operands)?;
        let ids: Vec<u32> = (0..solids.len() as u32).collect();
        let build = |lt: &LabeledTool| -> Result<Vec<Tool>, KernelError> {
            let base = self.tools(&ToolOp::new(lt.shape.clone(), combine), &solids, 0)?;
            let remap = |l: &HistorySource| -> Option<HistorySource> {
                let HistorySource::Generated { role, tag, .. } = l else {
                    return None;
                };
                let found = match (role, tag) {
                    (Role::Side, KernelTag::Curve { curve, .. }) => lt
                        .sides
                        .iter()
                        .find(|(c, _)| c == curve)
                        .map(|(_, f)| f.clone()),
                    (Role::StartCap, _) => lt.start.clone(),
                    (Role::EndCap, _) => lt.end.clone(),
                    _ => None,
                }?;
                Some(gen(found.role, found.tag, found.ordinal))
            };
            Ok(base
                .into_iter()
                .map(|mut t| {
                    t.map_labels(&remap);
                    if lt.clip.is_empty() {
                        t
                    } else {
                        Tool::Clipped(
                            Box::new(t),
                            lt.clip
                                .iter()
                                .map(|c| {
                                    (
                                        c.origin,
                                        c.normal,
                                        gen(c.label.role, c.label.tag.clone(), c.label.ordinal),
                                    )
                                })
                                .collect(),
                        )
                    }
                })
                .collect())
        };
        let mut built = Vec::new();
        let mut body_labels = Vec::new();
        for lt in tools {
            for t in build(lt)? {
                built.push(t);
                body_labels.push(
                    lt.body
                        .as_ref()
                        .map(|b| gen(b.role, b.tag.clone(), b.ordinal))
                        .unwrap_or_else(|| gen(Role::Body, KernelTag::None, None)),
                );
            }
        }
        let mut carved = Vec::new();
        for lt in carve {
            carved.extend(build(lt)?);
        }
        let mode = Self::mode(combine);
        let (ops, op_ids) = if mode == Mode::New {
            (Vec::new(), Vec::new())
        } else {
            (solids.clone(), ids.clone())
        };
        let mut job = grid_job(ops.clone(), op_ids.clone(), built, mode);
        job.separate_tools = separate;
        job.body_label = Some(body_labels);
        job.carve = carved;
        job.merge_collinear = tools.iter().all(|t| t.shape.merge_collinear());
        let res = run_grid(job)?;
        self.finish(&ops, &op_ids, res, Vec::new(), Vec::new())
    }

    fn move_faces(
        &self,
        faces: &[OperandEntity],
        motion: &FaceMotion,
        operands: &[&BodyHandle],
    ) -> Result<KernelOutput, KernelError> {
        let solids = Self::solids(operands)?;
        if solids.len() != 1 {
            return Err(KernelError::Failed {
                message: "moving faces works on one body".into(),
            });
        }
        let so = solids[0];
        let mut specs = Vec::new();
        for e in faces {
            let face = so.faces.get(e.entity as usize).ok_or(KernelError::Failed {
                message: "face out of range".into(),
            })?;
            let plane = face.plane.ok_or_else(|| KernelError::NotSupported {
                what: "fake kernel: moving a curved face".into(),
            })?;
            let mut n = [0.0; 3];
            n[plane.axis] = plane.dir as f64;
            let (lo, hi) = face_bounds(face);
            let (origin, pull, tan, shift) = match motion {
                FaceMotion::Offset { distance } => (lo, [0.0; 3], 0.0, *distance),
                FaceMotion::Translate { vector } => {
                    let along = vector[0] * n[0] + vector[1] * n[1] + vector[2] * n[2];
                    let tangential = (0..3).any(|a| (vector[a] - along * n[a]).abs() > 1e-12);
                    if tangential {
                        return Err(KernelError::NotSupported {
                            what: "fake kernel: sliding a face along itself".into(),
                        });
                    }
                    (lo, [0.0; 3], 0.0, along)
                }
                FaceMotion::Rotate {
                    axis_point,
                    axis_direction,
                    angle,
                } => {
                    let a = crate::model::datum::normalize(*axis_direction);
                    let in_plane = (a[plane.axis]).abs() < 1e-9
                        && (axis_point[plane.axis] - plane.offset).abs() < 1e-9;
                    if !in_plane {
                        return Err(KernelError::NotSupported {
                            what: "fake kernel: rotating a face about an axis off its plane".into(),
                        });
                    }
                    let pull = crate::model::datum::cross(a, n);
                    (*axis_point, pull, angle.tan(), 0.0)
                }
            };
            let mut max_h: f64 = 0.0;
            for i in 0..8 {
                let c = [
                    if i & 1 == 0 { lo[0] } else { hi[0] },
                    if i & 2 == 0 { lo[1] } else { hi[1] },
                    if i & 4 == 0 { lo[2] } else { hi[2] },
                ];
                let h: f64 = (0..3).map(|a| (c[a] - origin[a]) * pull[a]).sum();
                max_h = max_h.max(h.abs());
            }
            specs.push(DraftSpec {
                face: e.entity,
                plane,
                lo,
                hi,
                origin,
                pull,
                tan,
                reach: tan.abs() * max_h + 1.5 * shift.abs() + EPS,
                shift,
            });
        }
        let mut job = grid_job(vec![so], vec![0], Vec::new(), Mode::Keep);
        job.modifier = Some(Modifier::Draft(specs));
        let res = run_grid(job)?;
        self.finish(&[so], &[0], res, Vec::new(), Vec::new())
    }

    fn delete_faces(
        &self,
        faces: &[OperandEntity],
        heal: bool,
        operands: &[&BodyHandle],
    ) -> Result<KernelOutput, KernelError> {
        let solids = Self::solids(operands)?;
        if solids.len() != 1 {
            return Err(KernelError::Failed {
                message: "deleting faces works on one body".into(),
            });
        }
        let so = solids[0];
        let set: BTreeSet<u32> = faces.iter().map(|e| e.entity).collect();
        if !heal {
            // The rest of the body as a surface body, open where the faces were.
            let mut b = SheetBuild {
                facets: Vec::new(),
                face_of: Vec::new(),
                face_sources: Vec::new(),
                hints: Vec::new(),
            };
            for (fi, f) in so.faces.iter().enumerate() {
                if set.contains(&(fi as u32)) {
                    continue;
                }
                let face = b.face_sources.len() as u32;
                let mut fs = sheet::solid_face_facets(
                    so,
                    fi as u32,
                    gen(Role::Boundary, KernelTag::None, None),
                )?;
                b.face_sources.push(vec![HistorySource::Input {
                    from: OperandEntity {
                        operand: 0,
                        entity: fi as u32,
                    },
                }]);
                b.hints.push(f.hint());
                let keep: BTreeSet<u32> = (0..so.faces.len() as u32)
                    .filter(|k| set.contains(k) || *k == fi as u32)
                    .collect();
                // A boundary is where a deleted face was: tag it by that face.
                let others: BTreeSet<u32> = (0..so.faces.len() as u32)
                    .filter(|k| !set.contains(k))
                    .collect();
                let _ = keep;
                sheet::label_by_neighbour(so, &others, &mut fs, Role::Boundary, 0);
                for x in fs {
                    b.facets.push(x);
                    b.face_of.push(face);
                }
            }
            let deleted_faces: Vec<OperandEntity> = set
                .iter()
                .map(|f| OperandEntity {
                    operand: 0,
                    entity: *f,
                })
                .collect();
            let mut out = sheet::output(
                &Self::id_static(),
                vec![(
                    b,
                    vec![HistorySource::Input {
                        from: OperandEntity {
                            operand: 0,
                            entity: 0,
                        },
                    }],
                )],
                Vec::new(),
                Vec::new(),
            )?;
            out.history.deleted_faces = deleted_faces;
            return Ok(out);
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        let mut score = 0.0;
        let mut facets = Vec::new();
        for f in &set {
            let face = so.faces.get(*f as usize).ok_or(KernelError::Failed {
                message: "face out of range".into(),
            })?;
            if !face.is_grid() {
                return Err(KernelError::NotSupported {
                    what: "fake kernel: deleting a blend face".into(),
                });
            }
            let (l, h) = face_bounds(face);
            for a in 0..3 {
                lo[a] = lo[a].min(l[a]);
                hi[a] = hi[a].max(h[a]);
            }
            facets.extend(face.rect_planes());
        }
        let c = [
            (lo[0] + hi[0]) / 2.0,
            (lo[1] + hi[1]) / 2.0,
            (lo[2] + hi[2]) / 2.0,
        ];
        for (pk, r) in &facets {
            let m = (r.lo[pk.axis] + r.hi[pk.axis]) / 2.0;
            score += (pk.dir as f64) * (c[pk.axis] - m);
        }
        let mut job = grid_job(vec![so], vec![0], Vec::new(), Mode::Keep);
        job.modifier = Some(Modifier::Fill(FillSpec {
            faces: set,
            lo,
            hi,
            cavity: score > 0.0,
        }));
        let res = run_grid(job)?;
        self.finish(&[so], &[0], res, Vec::new(), Vec::new())
    }

    fn replace_faces(
        &self,
        faces: &[OperandEntity],
        with: OperandEntity,
        operands: &[&BodyHandle],
    ) -> Result<KernelOutput, KernelError> {
        let so = Self::solid(operands, 0)?;
        let sh = operands
            .get(with.operand as usize)
            .and_then(|h| h.downcast::<FakeSheet>())
            .ok_or(KernelError::ForeignOperand {
                operand: with.operand,
            })?;
        let target: Vec<&Facet> = sh.face_facets(with.entity).collect();
        let Some(first) = target.first() else {
            return Err(KernelError::Failed {
                message: "the replacement face is empty".into(),
            });
        };
        let Some((ax, s)) = axis_of(first.normal) else {
            return Err(KernelError::NotSupported {
                what: "fake kernel: replacing by a face that is not axis-aligned".into(),
            });
        };
        let off = first.loops[0][0][ax];
        if target
            .iter()
            .any(|f| f.loops.iter().flatten().any(|p| (p[ax] - off).abs() > 1e-9))
        {
            return Err(KernelError::NotSupported {
                what: "fake kernel: replacing by a curved face".into(),
            });
        }
        let _ = s;
        // Each replaced face moves onto the replacement's plane.
        let mut out: Option<KernelOutput> = None;
        let handles: Vec<BodyHandle> = vec![operands[0].clone()];
        let mut current = handles;
        for e in faces {
            let solid = current[0].downcast::<FakeSolid>().expect("fake solid");
            let face = solid
                .faces
                .get(e.entity as usize)
                .ok_or(KernelError::Failed {
                    message: "face out of range".into(),
                })?;
            let plane = face.plane.ok_or_else(|| KernelError::NotSupported {
                what: "fake kernel: replacing a curved face".into(),
            })?;
            if plane.axis != ax {
                return Err(KernelError::Failed {
                    message: "the replacement face is not parallel to the face it replaces".into(),
                });
            }
            let distance = (plane.dir as f64) * (off - plane.offset);
            let refs: Vec<&BodyHandle> = current.iter().collect();
            let o = self.move_faces(
                &[OperandEntity {
                    operand: 0,
                    entity: e.entity,
                }],
                &FaceMotion::Offset { distance },
                &refs,
            )?;
            current = o.bodies.iter().map(|b| b.handle.clone()).collect();
            out = Some(o);
            if faces.len() > 1 {
                return Err(KernelError::NotSupported {
                    what: "fake kernel: replacing several faces at once".into(),
                });
            }
        }
        let mut out = out.ok_or(KernelError::Failed {
            message: "no faces to replace".into(),
        })?;
        let _ = so;
        for r in &mut out.history.faces {
            if r.sources.len() == 1 {
                if let HistorySource::Input { from } = r.sources[0] {
                    if faces.iter().any(|f| f.entity == from.entity) {
                        r.sources = vec![gen(
                            Role::Replacement,
                            KernelTag::InputFace { face: from },
                            None,
                        )];
                        r.evolution = Evolution::Generated;
                    }
                }
            }
        }
        // The replaced face has no input image now.
        for f in faces {
            out.history.deleted_faces.push(OperandEntity {
                operand: 0,
                entity: f.entity,
            });
        }
        // The replacement sheet is consumed (history contract rule 3): it and its faces have
        // no image.
        out.history.deleted_bodies.push(with.operand);
        for f in 0..sh.topology.faces.len() as u32 {
            out.history.deleted_faces.push(OperandEntity {
                operand: with.operand,
                entity: f,
            });
        }
        out.history.deleted_faces.sort();
        out.history.deleted_bodies.sort();
        Ok(out)
    }

    fn scale(
        &self,
        center: [f64; 3],
        factors: [f64; 3],
        operands: &[&BodyHandle],
    ) -> Result<KernelOutput, KernelError> {
        let solids = Self::solids(operands)?;
        let ids: Vec<u32> = (0..solids.len() as u32).collect();
        if factors.iter().any(|f| *f <= 0.0 || !f.is_finite()) {
            return Err(KernelError::Failed {
                message: "scale factors must be positive".into(),
            });
        }
        let moved: Vec<FakeSolid> = solids
            .iter()
            .map(|s| {
                let mut c = (*s).clone();
                scale_solid(&mut c, center, factors);
                c
            })
            .collect();
        if moved.iter().any(|m| !m.decorations.is_empty()) {
            return Err(KernelError::NotSupported {
                what: "fake kernel: scaling a blended body".into(),
            });
        }
        let refs: Vec<&FakeSolid> = moved.iter().collect();
        let res = run_grid(grid_job(refs.clone(), ids.clone(), Vec::new(), Mode::Keep))?;
        self.finish(&refs, &ids, res, Vec::new(), Vec::new())
    }

    fn surface(
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

fn sheet_rect(pk: PlaneKey, r: Rect) -> Facet {
    let a = pk.axis;
    let (u, v) = ((a + 1) % 3, (a + 2) % 3);
    let mk = |x: f64, y: f64| {
        let mut p = [0.0; 3];
        p[a] = pk.offset;
        p[u] = x;
        p[v] = y;
        p
    };
    let mut pts = vec![
        mk(r.lo[u], r.lo[v]),
        mk(r.hi[u], r.lo[v]),
        mk(r.hi[u], r.hi[v]),
        mk(r.lo[u], r.hi[v]),
    ];
    if pk.dir < 0 {
        pts.reverse();
    }
    let mut n = [0.0; 3];
    n[a] = pk.dir as f64;
    Facet {
        loops: vec![pts],
        normal: n,
        labels: vec![vec![gen(Role::Boundary, KernelTag::None, None); 4]],
    }
}

/// Offset-surface boundaries: each continues the input boundary it offsets.
fn relabel_boundaries(f: &Facet, _role: Role) -> Facet {
    f.clone()
}

/// Input faces of sheet operands with no image are deleted.
fn fill_deleted(out: &mut KernelOutput, operands: &[&BodyHandle]) {
    let used: BTreeSet<(u32, u32)> = out
        .history
        .faces
        .iter()
        .flat_map(|r| r.sources.iter())
        .filter_map(|s| match s {
            HistorySource::Input { from } => Some((from.operand, from.entity)),
            _ => None,
        })
        .collect();
    for (i, h) in operands.iter().enumerate() {
        if let Some(s) = h.downcast::<FakeSheet>() {
            for f in 0..s.topology.faces.len() as u32 {
                if !used.contains(&(i as u32, f)) {
                    out.history.deleted_faces.push(OperandEntity {
                        operand: i as u32,
                        entity: f,
                    });
                }
            }
        }
    }
}

fn face_bounds(f: &FFace) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for r in &f.rects {
        for a in 0..3 {
            lo[a] = lo[a].min(r.lo[a]);
            hi[a] = hi[a].max(r.hi[a]);
        }
    }
    (lo, hi)
}

/// Scale a solid's grid and faces about `c` by per-axis factors (positive).
fn scale_solid(s: &mut FakeSolid, c: [f64; 3], k: [f64; 3]) {
    let m = |a: usize, x: f64| c[a] + k[a] * (x - c[a]);
    for a in 0..3 {
        for g in &mut s.grid[a] {
            *g = m(a, *g);
        }
    }
    let uniform = (k[0] - k[1]).abs() < 1e-12 && (k[1] - k[2]).abs() < 1e-12;
    for f in &mut s.faces {
        if let Some(p) = &mut f.plane {
            p.offset = m(p.axis, p.offset);
        }
        for p in &mut f.facet_planes {
            p.offset = m(p.axis, p.offset);
        }
        if let Some(h) = &mut f.surface {
            *h = match *h {
                SurfaceKind::Cylinder {
                    axis_point,
                    axis_direction,
                    radius,
                } if uniform => SurfaceKind::Cylinder {
                    axis_point: [
                        m(0, axis_point[0]),
                        m(1, axis_point[1]),
                        m(2, axis_point[2]),
                    ],
                    axis_direction,
                    radius: radius * k[0],
                },
                SurfaceKind::Plane { normal } => {
                    let n = [normal[0] / k[0], normal[1] / k[1], normal[2] / k[2]];
                    SurfaceKind::Plane {
                        normal: crate::model::datum::normalize(n),
                    }
                }
                _ => SurfaceKind::Other,
            };
        }
        for r in &mut f.rects {
            for a in 0..3 {
                r.lo[a] = m(a, r.lo[a]);
                r.hi[a] = m(a, r.hi[a]);
            }
        }
    }
}

impl crate::evaluate::GeometryAnalyzer for FakeKernel {
    /// Mass properties exactly from grid occupancy (orthogonal cells), draft from face normals,
    /// and the rest from the grid and its faces ([`super::analysis`]).
    fn analyze(
        &self,
        request: &crate::evaluate::AnalysisRequest,
        bodies: &[&BodyHandle],
    ) -> Result<crate::evaluate::AnalysisResult, KernelError> {
        use crate::evaluate::{AnalysisRequest as R, AnalysisResult, MassProperties};
        match request {
            R::MassProperties { density } => {
                let mut v = 0.0;
                let mut m1 = [0.0; 3];
                // Second moments about the origin: ∫ x_i x_j dV.
                let mut m2 = [[0.0; 3]; 3];
                for (k, h) in bodies.iter().enumerate() {
                    let s = h
                        .downcast::<FakeSolid>()
                        .ok_or(KernelError::ForeignOperand { operand: k as u32 })?;
                    let d = s.dims();
                    for z in 0..d[2] {
                        for y in 0..d[1] {
                            for x in 0..d[0] {
                                if !s.occ[x + d[0] * (y + d[1] * z)] {
                                    continue;
                                }
                                let c = [x, y, z];
                                let lo: Vec<f64> = (0..3).map(|a| s.grid[a][c[a]]).collect();
                                let hi: Vec<f64> = (0..3).map(|a| s.grid[a][c[a] + 1]).collect();
                                let w: Vec<f64> = (0..3).map(|a| hi[a] - lo[a]).collect();
                                let dv = w[0] * w[1] * w[2];
                                v += dv;
                                let mid: Vec<f64> = (0..3).map(|a| (lo[a] + hi[a]) / 2.0).collect();
                                for i in 0..3 {
                                    m1[i] += dv * mid[i];
                                    for j in 0..3 {
                                        m2[i][j] += if i == j {
                                            dv * (mid[i] * mid[i] + w[i] * w[i] / 12.0)
                                        } else {
                                            dv * mid[i] * mid[j]
                                        };
                                    }
                                }
                            }
                        }
                    }
                }
                if v <= 0.0 {
                    return Err(KernelError::Failed {
                        message: "no volume".into(),
                    });
                }
                let c = [m1[0] / v, m1[1] / v, m1[2] / v];
                // Central second moments, then the inertia tensor.
                let mut cm = [[0.0; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        cm[i][j] = m2[i][j] - v * c[i] * c[j];
                    }
                }
                let tr = cm[0][0] + cm[1][1] + cm[2][2];
                let mut inertia = [[0.0; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        inertia[i][j] = density * (if i == j { tr } else { 0.0 } - cm[i][j]);
                    }
                }
                Ok(AnalysisResult::MassProperties(MassProperties {
                    volume: v,
                    mass: v * density,
                    centroid: c,
                    inertia,
                }))
            }
            R::Draft { pull, min } => {
                let first = bodies.first().ok_or(KernelError::Failed {
                    message: "no body".into(),
                })?;
                let s = first
                    .downcast::<FakeSolid>()
                    .ok_or(KernelError::ForeignOperand { operand: 0 })?;
                Ok(AnalysisResult::Draft(crate::evaluate::draft_analysis(
                    &s.topology,
                    *pull,
                    *min,
                )))
            }
            R::SectionProperties { origin, normal } => {
                super::analysis::section(bodies, *origin, *normal)
            }
            R::Thickness { min, max } => super::analysis::thickness(bodies, *min, *max),
            R::Curvature => super::analysis::curvature(bodies),
            R::Zebra => super::analysis::zebra(bodies),
            R::Deviation => super::analysis::deviation(bodies),
            R::Undercut { pull } => super::analysis::undercut(bodies, *pull),
        }
    }
}
