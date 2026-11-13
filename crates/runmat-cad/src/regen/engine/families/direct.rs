//! Direct edits (move, replace and delete faces, move bodies) and the live relations they keep.

use super::*;

impl<'a, 'b> Run<'a, 'b> {
    pub(super) fn motion(
        &mut self,
        i: usize,
        id: &FeatureId,
        m: &DirectMotion,
    ) -> Result<FaceMotion, Halt> {
        Ok(match m {
            DirectMotion::Offset { distance } => FaceMotion::Offset {
                distance: self.value(id, "motion.distance", distance, Dimension::LENGTH)?,
            },
            DirectMotion::Translate { dx, dy, dz } => FaceMotion::Translate {
                vector: [
                    self.value(id, "motion.dx", dx, Dimension::LENGTH)?,
                    self.value(id, "motion.dy", dy, Dimension::LENGTH)?,
                    self.value(id, "motion.dz", dz, Dimension::LENGTH)?,
                ],
            },
            DirectMotion::Rotate { axis, angle } => {
                let (p, d) = self.axis(i, id, "motion.axis", axis)?;
                FaceMotion::Rotate {
                    axis_point: p,
                    axis_direction: d,
                    angle: self.value(id, "motion.angle", angle, Dimension::ANGLE)?,
                }
            }
        })
    }

    pub(super) fn direct(
        &mut self,
        i: usize,
        id: &FeatureId,
        d: &DirectEdit,
    ) -> Result<Compiled, Halt> {
        let simple = NamingContext::simple(id, Vec::new());
        match d {
            DirectEdit::MoveFaces {
                faces,
                motion,
                keep,
            } => {
                let set = self.list(i, id, "faces", faces, Some(TopoKind::Face))?;
                let operands = Self::bodies_of(&set);
                if operands.len() != 1 {
                    return Err(self.error(
                        id,
                        Some("faces"),
                        DiagCode::RefKindMismatch,
                        "moved faces must be on one body".into(),
                        Vec::new(),
                    ));
                }
                let motion = self.motion(i, id, motion)?;
                self.check_relations(i, id, keep, &set, &motion)?;
                let faces = Self::operand_entities(&set, &operands);
                Ok(op(KernelOp::MoveFaces { faces, motion }, operands, simple))
            }
            DirectEdit::ReplaceFaces { faces, with } => {
                let set = self.list(i, id, "faces", faces, Some(TopoKind::Face))?;
                let target = Self::bodies_of(&set);
                let w = self.face_handle(i, id, "with", with)?;
                if target.len() != 1 || target[0] == w.body as usize {
                    return Err(self.error(
                        id,
                        Some("with"),
                        DiagCode::RefKindMismatch,
                        "replace faces of one body by a face of a surface body".into(),
                        Vec::new(),
                    ));
                }
                let operands = vec![target[0], w.body as usize];
                let faces = Self::operand_entities(&set, &operands);
                let with = OperandEntity {
                    operand: 1,
                    entity: w.index,
                };
                Ok(op(KernelOp::ReplaceFaces { faces, with }, operands, simple))
            }
            DirectEdit::DeleteFaces { faces, heal } => {
                let set = self.list(i, id, "faces", faces, Some(TopoKind::Face))?;
                let operands = Self::bodies_of(&set);
                if operands.len() != 1 {
                    return Err(self.error(
                        id,
                        Some("faces"),
                        DiagCode::RefKindMismatch,
                        "deleted faces must be on one body".into(),
                        Vec::new(),
                    ));
                }
                let faces = Self::operand_entities(&set, &operands);
                Ok(op(
                    KernelOp::DeleteFaces { faces, heal: *heal },
                    operands,
                    simple,
                ))
            }
            DirectEdit::MoveBodies {
                bodies,
                motion,
                copy,
            } => {
                let set = self.list(i, id, "bodies", bodies, Some(TopoKind::Body))?;
                let operands = Self::bodies_of(&set);
                let transform = match self.motion(i, id, motion)? {
                    FaceMotion::Offset { .. } => {
                        return Err(self.error(
                            id,
                            Some("motion"),
                            DiagCode::RegenInvalid,
                            "a body moves by a translation or a rotation".into(),
                            Vec::new(),
                        ))
                    }
                    FaceMotion::Translate { vector } => Transform::translation(vector),
                    FaceMotion::Rotate {
                        axis_point,
                        axis_direction,
                        angle,
                    } => Transform::rotation_about(axis_point, axis_direction, angle),
                };
                if *copy {
                    let key = InstanceKey::Row { key: "copy".into() };
                    Ok(Compiled::Op {
                        op: KernelOp::PatternBodies {
                            instances: vec![Instance {
                                key: key.clone(),
                                transform,
                            }],
                        },
                        operands,
                        naming: simple,
                        import: false,
                        instances: Some([key].into_iter().collect()),
                        tools: None,
                    })
                } else {
                    Ok(op(KernelOp::Transform { transform }, operands, simple))
                }
            }
        }
    }

    /// A live relation fails the edit when the motion cannot keep it.
    pub(super) fn check_relations(
        &mut self,
        i: usize,
        id: &FeatureId,
        keep: &[LiveRelation],
        moved: &BTreeSet<EntityHandle>,
        motion: &FaceMotion,
    ) -> Result<(), Halt> {
        for (k, rel) in keep.iter().enumerate() {
            let param = format!("keep[{}]", k);
            let resolve = |s: &mut Self, r: &Reference| -> Result<(bool, Option<[f64; 3]>), Halt> {
                let set = s.list(i, id, &param, std::slice::from_ref(r), Some(TopoKind::Face))?;
                let any_moved = set.iter().any(|h| moved.contains(h));
                let n = set.iter().next().and_then(|h| s.face_normal(h));
                Ok((any_moved, n))
            };
            let broken = match rel {
                LiveRelation::Fixed { face } => resolve(self, face)?.0,
                LiveRelation::Parallel { a, b }
                | LiveRelation::Perpendicular { a, b }
                | LiveRelation::Coplanar { a, b } => {
                    let (ma, na) = resolve(self, a)?;
                    let (mb, _) = resolve(self, b)?;
                    let one_moves = ma != mb;
                    match motion {
                        FaceMotion::Rotate {
                            axis_direction,
                            angle,
                            ..
                        } => {
                            one_moves
                                && angle.abs() > 1e-12
                                && na
                                    .map(|n| {
                                        dot(normalize(n), normalize(*axis_direction)).abs()
                                            < 1.0 - 1e-9
                                    })
                                    .unwrap_or(true)
                        }
                        FaceMotion::Offset { distance } => {
                            matches!(rel, LiveRelation::Coplanar { .. })
                                && one_moves
                                && distance.abs() > 1e-12
                        }
                        FaceMotion::Translate { vector } => {
                            matches!(rel, LiveRelation::Coplanar { .. })
                                && one_moves
                                && na.map(|n| dot(n, *vector).abs() > 1e-12).unwrap_or(true)
                        }
                    }
                }
                LiveRelation::Concentric { a, b } | LiveRelation::Tangent { a, b } => {
                    let (ma, _) = resolve(self, a)?;
                    let (mb, _) = resolve(self, b)?;
                    ma != mb
                }
                LiveRelation::Symmetric { a, b, .. } => {
                    let (ma, _) = resolve(self, a)?;
                    let (mb, _) = resolve(self, b)?;
                    ma != mb
                }
            };
            if broken {
                return Err(self.error(
                    id,
                    Some(&param),
                    DiagCode::DirectRelationBroken,
                    format!("the edit would break the live relation {}", param),
                    Vec::new(),
                ));
            }
        }
        Ok(())
    }

    // ---------------------------------------------------------------------- surfaces
}
