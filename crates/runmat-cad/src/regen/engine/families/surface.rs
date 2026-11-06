//! Surface features: extruded, revolved, swept, lofted, boundary, fill, planar, ruled and offset
//! surfaces and the operations on surface bodies.

use super::*;

impl<'a, 'b> Run<'a, 'b> {
    pub(super) fn surface_feature(
        &mut self,
        i: usize,
        id: &FeatureId,
        s: &SurfaceFeature,
    ) -> Result<Compiled, Halt> {
        let simple = |p: Vec<FeatureId>| NamingContext::simple(id, p);
        let sheet_bodies =
            |run: &mut Self, param: &str, refs: &[Reference]| -> Result<Vec<usize>, Halt> {
                let set = run.list(i, id, param, refs, Some(TopoKind::Body))?;
                Ok(Self::bodies_of(&set))
            };
        Ok(match s {
            SurfaceFeature::Extrude {
                profile,
                start,
                end,
            } => {
                let (frame, segs) = self.chain(i, id, "profile", profile)?;
                let (a, b) = (
                    self.value(id, "start", start, Dimension::LENGTH)?,
                    self.value(id, "end", end, Dimension::LENGTH)?,
                );
                let shape = ToolShape::prism(
                    CanonicalProfile {
                        frame,
                        loops: vec![segs],
                        merge_collinear: profile.merge_collinear,
                    },
                    a,
                    b,
                );
                op(
                    KernelOp::Surface(SurfaceOp::Swept {
                        tool: LabeledTool::new(shape),
                    }),
                    Vec::new(),
                    simple(vec![profile.sketch.clone()]),
                )
            }
            SurfaceFeature::Revolve {
                profile,
                axis,
                angle,
            } => {
                let (frame, segs) = self.chain(i, id, "profile", profile)?;
                let (p, d) = self.axis(i, id, "axis", axis)?;
                let a = self.value(id, "angle", angle, Dimension::ANGLE)?;
                let shape = ToolShape::Revolve {
                    profile: CanonicalProfile {
                        frame,
                        loops: vec![segs],
                        merge_collinear: false,
                    },
                    axis_point: p,
                    axis_direction: d,
                    angle: a,
                };
                op(
                    KernelOp::Surface(SurfaceOp::Swept {
                        tool: LabeledTool::new(shape),
                    }),
                    Vec::new(),
                    simple(vec![profile.sketch.clone()]),
                )
            }
            SurfaceFeature::Planar { profile } => {
                let p = self.profile(i, id, "profile", profile)?;
                op(
                    KernelOp::Surface(SurfaceOp::Planar {
                        patches: vec![SurfacePatch {
                            profile: p,
                            label: FaceLabel {
                                role: Role::SurfacePatch,
                                tag: KernelTag::None,
                                ordinal: None,
                            },
                        }],
                    }),
                    Vec::new(),
                    simple(vec![profile.sketch.clone()]),
                )
            }
            SurfaceFeature::Offset { faces, distance } => {
                let set = self.list(i, id, "faces", faces, Some(TopoKind::Face))?;
                let operands = Self::bodies_of(&set);
                let faces = Self::operand_entities(&set, &operands);
                let distance = self.value(id, "distance", distance, Dimension::LENGTH)?;
                op(
                    KernelOp::Surface(SurfaceOp::Offset { faces, distance }),
                    operands,
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::Extend { edges, distance } => {
                let set = self.list(i, id, "edges", edges, Some(TopoKind::Edge))?;
                let operands = Self::bodies_of(&set);
                let edges = Self::operand_entities(&set, &operands);
                let distance = self.value(id, "distance", distance, Dimension::LENGTH)?;
                op(
                    KernelOp::Surface(SurfaceOp::Extend { edges, distance }),
                    operands,
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::Trim { bodies, tool, keep } => {
                let operands = sheet_bodies(self, "bodies", bodies)?;
                let frame = self.plane(i, id, "tool", tool)?;
                op(
                    KernelOp::Surface(SurfaceOp::Trim { frame, keep: *keep }),
                    operands,
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::Knit { bodies, solid } => {
                let operands = sheet_bodies(self, "bodies", bodies)?;
                op(
                    KernelOp::Surface(SurfaceOp::Knit { solid: *solid }),
                    operands,
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::Thicken {
                bodies,
                thickness,
                both_sides,
            } => {
                let operands = sheet_bodies(self, "bodies", bodies)?;
                let distance = self.value(id, "thickness", thickness, Dimension::LENGTH)?;
                op(
                    KernelOp::Surface(SurfaceOp::Thicken {
                        distance,
                        both: *both_sides,
                    }),
                    operands,
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::MidSurface { body, pairs } => {
                let b = self.single(i, id, "body", body, Some(TopoKind::Body))?;
                let body = b.entities().iter().next().expect("one").body as usize;
                let mut out = Vec::new();
                for (k, (a, c)) in pairs.iter().enumerate() {
                    let ha = self.face_handle(i, id, &format!("pairs[{}].0", k), a)?;
                    let hc = self.face_handle(i, id, &format!("pairs[{}].1", k), c)?;
                    if ha.body as usize != body || hc.body as usize != body {
                        return Err(self.error(
                            id,
                            Some("pairs"),
                            DiagCode::RefKindMismatch,
                            "mid-surface faces must be on the body".into(),
                            Vec::new(),
                        ));
                    }
                    out.push((
                        OperandEntity {
                            operand: 0,
                            entity: ha.index,
                        },
                        OperandEntity {
                            operand: 0,
                            entity: hc.index,
                        },
                    ));
                }
                op(
                    KernelOp::Surface(SurfaceOp::MidSurface { pairs: out }),
                    vec![body],
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::Sweep { profile, path } => {
                let (frame, segs) = self.chain(i, id, "profile", profile)?;
                let (path_points, path_curves) = self.path(i, id, "path", path)?;
                let shape = ToolShape::Sweep {
                    profile: CanonicalProfile {
                        frame,
                        loops: vec![segs],
                        merge_collinear: false,
                    },
                    path: path_points,
                    path_curves,
                };
                op(
                    KernelOp::Surface(SurfaceOp::Swept {
                        tool: LabeledTool::new(shape),
                    }),
                    Vec::new(),
                    simple(vec![profile.sketch.clone(), path.sketch.clone()]),
                )
            }
            SurfaceFeature::Loft { profiles, guides } => {
                if !guides.is_empty() {
                    return Err(self.not_lowered(id, "lofts along guide curves"));
                }
                let mut sections = Vec::new();
                for (k, p) in profiles.iter().enumerate() {
                    let (frame, segs) = self.chain(i, id, &format!("profiles[{}]", k), p)?;
                    sections.push(CanonicalProfile {
                        frame,
                        loops: vec![segs],
                        merge_collinear: false,
                    });
                }
                op(
                    KernelOp::Surface(SurfaceOp::Swept {
                        tool: LabeledTool::new(ToolShape::Loft { profiles: sections }),
                    }),
                    Vec::new(),
                    simple(profiles.iter().map(|p| p.sketch.clone()).collect()),
                )
            }
            SurfaceFeature::Boundary { first, second } => {
                let mut operands: Vec<usize> = Vec::new();
                let mut lower = |run: &mut Self, param: &str, list: &[crate::model::surface::BoundaryEdge]| -> Result<Vec<(OperandEntity, crate::model::surface::Continuity)>, Halt> {
                    let mut out = Vec::new();
                    for (k, b) in list.iter().enumerate() {
                        let h = run.single(i, id, &format!("{param}[{k}]"), &b.edge, Some(TopoKind::Edge))?;
                        let e = *h.entities().iter().next().expect("one");
                        let body = e.body as usize;
                        let operand = match operands.iter().position(|x| *x == body) {
                            Some(p) => p,
                            None => {
                                operands.push(body);
                                operands.len() - 1
                            }
                        };
                        out.push((OperandEntity { operand: operand as u32, entity: e.index }, b.continuity));
                    }
                    Ok(out)
                };
                let first = lower(self, "first", first)?;
                let second = lower(self, "second", second)?;
                op(
                    KernelOp::Surface(SurfaceOp::Boundary { first, second }),
                    operands,
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::Fill { boundary } => {
                let refs: Vec<Reference> = boundary.iter().map(|b| b.edge.clone()).collect();
                let mut edges = Vec::new();
                let mut operands: Vec<usize> = Vec::new();
                for (k, (r, b)) in refs.iter().zip(boundary).enumerate() {
                    let h =
                        self.single(i, id, &format!("boundary[{}]", k), r, Some(TopoKind::Edge))?;
                    let e = *h.entities().iter().next().expect("one");
                    let body = e.body as usize;
                    let operand = match operands.iter().position(|x| *x == body) {
                        Some(p) => p,
                        None => {
                            operands.push(body);
                            operands.len() - 1
                        }
                    };
                    edges.push((
                        OperandEntity {
                            operand: operand as u32,
                            entity: e.index,
                        },
                        b.continuity,
                    ));
                }
                op(
                    KernelOp::Surface(SurfaceOp::Fill { edges }),
                    operands,
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::Ruled {
                edges,
                distance,
                angle,
            } => {
                let set = self.list(i, id, "edges", edges, Some(TopoKind::Edge))?;
                let operands = Self::bodies_of(&set);
                let edges = Self::operand_entities(&set, &operands);
                let distance = self.value(id, "distance", distance, Dimension::LENGTH)?;
                let angle = self.value(id, "angle", angle, Dimension::ANGLE)?;
                op(
                    KernelOp::Surface(SurfaceOp::Ruled {
                        edges,
                        distance,
                        angle,
                    }),
                    operands,
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::Untrim { .. } => return Err(self.not_lowered(id, "untrim")),
            SurfaceFeature::DeleteHole { edges } => {
                let set = self.list(i, id, "edges", edges, Some(TopoKind::Edge))?;
                let operands = Self::bodies_of(&set);
                let edges = Self::operand_entities(&set, &operands);
                op(
                    KernelOp::Surface(SurfaceOp::DeleteHole { edges }),
                    operands,
                    simple(Vec::new()),
                )
            }
            SurfaceFeature::CutWithSurface {
                body,
                surface,
                keep,
            } => {
                let b = self.single(i, id, "body", body, Some(TopoKind::Body))?;
                let s = self.single(i, id, "surface", surface, Some(TopoKind::Body))?;
                let solid = b.entities().iter().next().expect("one").body as usize;
                let sheet = s.entities().iter().next().expect("one").body as usize;
                if solid == sheet {
                    return Err(self.error(
                        id,
                        Some("surface"),
                        DiagCode::RefKindMismatch,
                        "a body cannot be cut with itself".into(),
                        Vec::new(),
                    ));
                }
                op(
                    KernelOp::Surface(SurfaceOp::CutWithSurface { keep: *keep }),
                    vec![solid, sheet],
                    simple(Vec::new()),
                )
            }
        })
    }

    // ------------------------------------------------------------------- sheet metal
}
