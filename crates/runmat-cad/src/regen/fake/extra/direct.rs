//! Direct edits on the grid test kernel: faces moved, deleted or replaced, and bodies scaled.

use super::*;

impl FakeKernel {
    pub(super) fn move_faces(
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

    pub(super) fn delete_faces(
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

    pub(super) fn replace_faces(
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

    pub(super) fn scale(
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
}
