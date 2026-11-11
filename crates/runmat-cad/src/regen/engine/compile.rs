//! Lowering one feature to its kernel operation: operand bodies, the operation, and the
//! pattern-like features.

use super::*;

impl<'a, 'b> Run<'a, 'b> {
    pub(super) fn bodies_of(set: &BTreeSet<EntityHandle>) -> Vec<usize> {
        let mut v: Vec<usize> = set.iter().map(|h| h.body as usize).collect();
        v.sort();
        v.dedup();
        v
    }

    pub(super) fn operand_entities(
        set: &BTreeSet<EntityHandle>,
        operands: &[usize],
    ) -> Vec<OperandEntity> {
        set.iter()
            .map(|h| OperandEntity {
                operand: operands
                    .iter()
                    .position(|b| *b == h.body as usize)
                    .expect("operand listed") as u32,
                entity: h.index,
            })
            .collect()
    }

    pub(super) fn mode_operands(
        &mut self,
        i: usize,
        f: &FeatureId,
        mode: &BodyMode,
    ) -> Result<(Combine, Vec<usize>), Halt> {
        let combine = match mode {
            BodyMode::New => return Ok((Combine::New, Vec::new())),
            BodyMode::Add { .. } => Combine::Add,
            BodyMode::Cut { .. } => Combine::Cut,
            BodyMode::Intersect { .. } => Combine::Intersect,
        };
        let targets = mode.targets();
        if targets.is_empty() {
            return self.implicit_bodies(i, f).map(|b| (combine, b));
        }
        let set = self.list(i, f, "targets", targets, Some(TopoKind::Body))?;
        Ok((combine, Self::bodies_of(&set)))
    }

    /// Every solid body, for a feature with implicit targets (surface bodies, with boundary
    /// edges, are never implicit targets of solid tools). With none present, the feature waits
    /// on (or cascades from) the earlier body feature that did not run; with none at all it
    /// fails.
    pub(super) fn implicit_bodies(&mut self, i: usize, f: &FeatureId) -> Result<Vec<usize>, Halt> {
        let solids: Vec<usize> = (0..self.st.bodies.len())
            .filter(|b| !self.st.bodies[*b].topology.edges.iter().any(|e| e.laminar))
            .collect();
        if !solids.is_empty() {
            return Ok(solids);
        }
        let earlier: Vec<FeatureId> = self.doc.features[..i]
            .iter()
            .filter(|x| x.kind.is_body_feature())
            .filter(|x| {
                !matches!(
                    self.st
                        .table
                        .entries
                        .get(&x.id)
                        .and_then(|e| e.status.clone()),
                    Some(MakerStatus::Ran)
                )
            })
            .map(|x| x.id.clone())
            .collect();
        if let Some(dep) = earlier.first() {
            self.gate_feature(i, f, dep, "targets")?;
        }
        Err(self.error(
            f,
            Some("targets"),
            DiagCode::RefEmpty,
            format!("`{}` has no body to act on", f),
            Vec::new(),
        ))
    }

    pub(super) fn compile(&mut self, i: usize, f: &Feature) -> Result<Compiled, Halt> {
        let id = &f.id;
        // Any parameter reading a failed value makes the whole feature a cascade of it (sketch
        // coordinates are evaluated by the solver, so check here rather than per value).
        for v in f.kind.values() {
            let mut ins = std::collections::BTreeSet::new();
            let mut ls = std::collections::BTreeSet::new();
            v.collect_names(&mut ins, &mut ls);
            if ins
                .iter()
                .map(|n| format!("input.{}", n))
                .chain(ls.iter().map(|n| format!("let.{}", n)))
                .any(|k| self.values.failed.contains_key(&k))
            {
                self.value(id, "values", v, Dimension::DIMENSIONLESS)?;
            }
        }
        if let Some(c) = self.compile_family(i, f)? {
            return Ok(c);
        }
        let simple = |profiles: Vec<FeatureId>| NamingContext::simple(id, profiles);
        match &f.kind {
            FeatureKind::Sketch(s) => {
                let frame = self.plane(i, id, "on", &s.plane)?;
                let watch = Stopwatch::start(self.meter);
                let solved = self.ctx.solver.solve(s, &self.ev, self.values);
                self.meter.record(&UsageEvent::SketchSolve {
                    file: self.doc.path.clone(),
                    feature: id.clone(),
                    entities: s.entities.len() as u32,
                    constraints: s.constraints.len() as u32,
                    dof: solved.as_ref().ok().and_then(|x| x.dof),
                    outcome: match &solved {
                        Err(_) => SolveOutcome::Failed,
                        Ok(x) => match x.status {
                            SketchSolveStatus::WellDefined => SolveOutcome::WellDefined,
                            SketchSolveStatus::UnderDefined => SolveOutcome::UnderDefined,
                            SketchSolveStatus::OverDefined => SolveOutcome::OverDefined,
                            SketchSolveStatus::Explicit => SolveOutcome::Explicit,
                        },
                    },
                    duration_nanos: watch.elapsed(self.meter),
                });
                match solved {
                    Ok(solved) => {
                        if !solved.redundant.is_empty() {
                            self.diags.push(
                                DiagCode::SketchRedundant,
                                Severity::Warning,
                                Site::feature(id, Some("constraints")),
                                format!("constraints {:?} are implied by others", solved.redundant),
                                Cause::Root,
                                Vec::new(),
                            );
                        }
                        Ok(Compiled::Sketch(solved, frame))
                    }
                    Err(e) => {
                        if !e.constraints.is_empty() {
                            self.keep_over_defined_sketch(id, s, frame, &e.constraints);
                        }
                        Err(self.error(
                            id,
                            Some("constraints"),
                            DiagCode::SketchSolveFailed,
                            e.message,
                            Vec::new(),
                        ))
                    }
                }
            }
            FeatureKind::Plane(p) => {
                let frame = match p {
                    PlaneDef::Offset { base, distance } => {
                        let b = self.plane(i, id, "base", base)?;
                        let d = self.value(id, "distance", distance, Dimension::LENGTH)?;
                        b.offset(d)
                    }
                    PlaneDef::Angle { base, axis, angle } => {
                        let b = self.plane(i, id, "base", base)?;
                        let (pt, dir) = self.axis(i, id, "axis", axis)?;
                        let a = self.value(id, "angle", angle, Dimension::ANGLE)?;
                        let rot = |v: [f64; 3]| rotate(v, dir, a);
                        Frame {
                            origin: add(pt, rot(sub(b.origin, pt))),
                            x: rot(b.x),
                            y: rot(b.y),
                            z: rot(b.z),
                        }
                    }
                    _ => unreachable!("handled by compile_family"),
                };
                Ok(Compiled::Datum(DatumValue::Plane { frame }))
            }
            FeatureKind::Axis(a) => {
                let (point, direction) = match a {
                    AxisDef::Through { a, b } => {
                        let pa = self.point(i, id, "a", a)?;
                        let pb = self.point(i, id, "b", b)?;
                        (pa, normalize(sub(pb, pa)))
                    }
                    AxisDef::OfEdge { edge } => {
                        self.axis(i, id, "edge", &AxisRef::Edge { edge: edge.clone() })?
                    }
                    AxisDef::OfFace { face } => {
                        let r = self.single(i, id, "face", face, Some(TopoKind::Face))?;
                        let h = *r.entities().iter().next().expect("one");
                        match self.st.bodies[h.body as usize].topology.faces[h.index as usize]
                            .geom
                            .surface
                        {
                            SurfaceKind::Cylinder {
                                axis_point,
                                axis_direction,
                                ..
                            } => (axis_point, axis_direction),
                            _ => {
                                return Err(self.error(
                                    id,
                                    Some("face"),
                                    DiagCode::RefKindMismatch,
                                    format!("`{}` is not cylindrical", face),
                                    Vec::new(),
                                ))
                            }
                        }
                    }
                    _ => unreachable!("handled by compile_family"),
                };
                Ok(Compiled::Datum(DatumValue::Axis { point, direction }))
            }
            FeatureKind::Point(p) => {
                let at = match p {
                    PointDef::At { at } => self.point(i, id, "at", at)?,
                    PointDef::Offset { base, dx, dy, dz } => {
                        let b = self.point(i, id, "base", base)?;
                        add(
                            b,
                            [
                                self.value(id, "dx", dx, Dimension::LENGTH)?,
                                self.value(id, "dy", dy, Dimension::LENGTH)?,
                                self.value(id, "dz", dz, Dimension::LENGTH)?,
                            ],
                        )
                    }
                    _ => unreachable!("handled by compile_family"),
                };
                Ok(Compiled::Datum(DatumValue::Point { at }))
            }
            FeatureKind::Frame(fr) => {
                let frame = match fr {
                    FrameDef::At { origin, z, x } => {
                        let o = self.point(i, id, "origin", origin)?;
                        let (_, zd) = self.axis(i, id, "z", z)?;
                        let xd = match x {
                            Some(x) => self.axis(i, id, "x", x)?.1,
                            None => {
                                if dot(zd, [1.0, 0.0, 0.0]).abs() < 0.9 {
                                    [1.0, 0.0, 0.0]
                                } else {
                                    [0.0, 1.0, 0.0]
                                }
                            }
                        };
                        let xd = normalize(sub(xd, scale(zd, dot(xd, zd))));
                        Frame {
                            origin: o,
                            x: xd,
                            y: cross(zd, xd),
                            z: zd,
                        }
                    }
                    FrameDef::OnFace { face } => {
                        self.plane(i, id, "face", &PlaneRef::Face { face: face.clone() })?
                    }
                };
                Ok(Compiled::Datum(DatumValue::Frame { frame }))
            }
            FeatureKind::Extrude(e) => self.extrude(i, id, e),
            FeatureKind::Revolve(r) => {
                let (profile, face) = self.profile_source(i, id, "profile", &r.profile)?;
                let (axis_point, axis_direction) = self.axis(i, id, "axis", &r.axis)?;
                let angle = self.value(id, "angle", &r.angle, Dimension::ANGLE)?;
                let (combine, mut operands) = self.mode_operands(i, id, &r.mode)?;
                let mut tool = ToolOp::new(
                    ToolShape::Revolve {
                        profile,
                        axis_point,
                        axis_direction,
                        angle,
                    },
                    combine,
                );
                profile_face(&mut tool, &mut operands, face);
                let reusable = tool.profile_face.is_none();
                Ok(Compiled::Op {
                    op: KernelOp::Tool(tool.clone()),
                    operands,
                    naming: simple(vec![Self::profile_naming(id, &r.profile)]),
                    import: false,
                    instances: None,
                    tools: reusable.then(|| vec![tool]),
                })
            }
            FeatureKind::Sweep(s) => {
                let (profile, face) = self.profile_source(i, id, "profile", &s.profile)?;
                let (path, path_curves) = self.path(i, id, "path", &s.path)?;
                let (combine, mut operands) = self.mode_operands(i, id, &s.mode)?;
                let mut tool = ToolOp::new(
                    ToolShape::Sweep {
                        profile,
                        path,
                        path_curves,
                    },
                    combine,
                );
                profile_face(&mut tool, &mut operands, face);
                Ok(Compiled::Op {
                    op: KernelOp::Tool(tool),
                    operands,
                    naming: simple(vec![Self::profile_naming(id, &s.profile)]),
                    import: false,
                    instances: None,
                    tools: None,
                })
            }
            FeatureKind::Loft(l) => {
                let mut profiles = Vec::new();
                for (k, p) in l.profiles.iter().enumerate() {
                    profiles.push(self.profile(i, id, &format!("profiles[{}]", k), p)?);
                }
                self.list(i, id, "guides", &l.guides, Some(TopoKind::Edge))
                    .map(|_| ())
                    .or_else(|h| if l.guides.is_empty() { Ok(()) } else { Err(h) })?;
                let (combine, operands) = self.mode_operands(i, id, &l.mode)?;
                let tool = ToolOp::new(ToolShape::Loft { profiles }, combine);
                Ok(Compiled::Op {
                    op: KernelOp::Tool(tool),
                    operands,
                    naming: simple(l.profiles.iter().map(|p| p.sketch.clone()).collect()),
                    import: false,
                    instances: None,
                    tools: None,
                })
            }
            FeatureKind::Hole(h) => self.hole(i, id, h),
            FeatureKind::Fillet(fl) => self.fillet(i, id, fl),
            FeatureKind::FaceFillet(fl) => self.face_fillet(i, id, fl),
            FeatureKind::FullRound(fl) => self.full_round(i, id, fl),
            FeatureKind::Chamfer(c) => self.chamfer(i, id, c),
            FeatureKind::Shell(s) => self.shell(i, id, s),
            FeatureKind::Draft(d) => self.draft(i, id, d),
            FeatureKind::Pattern(p) => self.pattern(i, id, p),
            FeatureKind::Mirror(m) => {
                let frame = self.plane(i, id, "plane", &m.plane)?;
                let instances = vec![Instance {
                    key: InstanceKey::Mirror,
                    transform: Transform::reflection(frame.origin, frame.z),
                }];
                self.pattern_like(i, id, &m.seeds, instances, Vec::new())
            }
            FeatureKind::Boolean(b) => {
                let t = self.single(i, id, "target", &b.target, Some(TopoKind::Body))?;
                let target = t.entities().iter().next().expect("one").body as usize;
                let tools = self.list(i, id, "tools", &b.tools, Some(TopoKind::Body))?;
                let mut operands = vec![target];
                for tb in Self::bodies_of(&tools) {
                    if tb != target {
                        operands.push(tb);
                    }
                }
                let combine = match b.kind {
                    BooleanKind::Union => Combine::Add,
                    BooleanKind::Subtract => Combine::Cut,
                    BooleanKind::Intersect => Combine::Intersect,
                };
                Ok(Compiled::Op {
                    op: KernelOp::Boolean {
                        combine,
                        keep_tools: b.keep_tools,
                    },
                    operands,
                    naming: simple(Vec::new()),
                    import: false,
                    instances: None,
                    tools: None,
                })
            }
            FeatureKind::Split(s) => {
                let b = self.single(i, id, "body", &s.body, Some(TopoKind::Body))?;
                let body = b.entities().iter().next().expect("one").body as usize;
                let frame = self.plane(i, id, "tool", &s.tool)?;
                Ok(Compiled::Op {
                    op: KernelOp::Split { frame },
                    operands: vec![body],
                    naming: simple(Vec::new()),
                    import: false,
                    instances: None,
                    tools: None,
                })
            }
            FeatureKind::BodyOp(op) => match op {
                BodyOp::Delete { bodies } => {
                    let set = self.list(i, id, "bodies", bodies, Some(TopoKind::Body))?;
                    Ok(Compiled::Delete(Self::bodies_of(&set)))
                }
                BodyOp::Translate { bodies, dx, dy, dz } => {
                    let set = self.list(i, id, "bodies", bodies, Some(TopoKind::Body))?;
                    let t = [
                        self.value(id, "dx", dx, Dimension::LENGTH)?,
                        self.value(id, "dy", dy, Dimension::LENGTH)?,
                        self.value(id, "dz", dz, Dimension::LENGTH)?,
                    ];
                    Ok(Compiled::Op {
                        op: KernelOp::Transform {
                            transform: Transform::translation(t),
                        },
                        operands: Self::bodies_of(&set),
                        naming: simple(Vec::new()),
                        import: false,
                        instances: None,
                        tools: None,
                    })
                }
            },
            FeatureKind::Direct(_)
            | FeatureKind::Surface(_)
            | FeatureKind::SheetMetal(_)
            | FeatureKind::Weldment(_)
            | FeatureKind::Mold(_)
            | FeatureKind::Thread(_)
            | FeatureKind::Emboss(_)
            | FeatureKind::Rib(_)
            | FeatureKind::Library(_)
            | FeatureKind::Curve(_)
            | FeatureKind::MateConnector(_) => unreachable!("handled by compile_family"),
            FeatureKind::Import(imp) => Ok(Compiled::Op {
                op: KernelOp::Import {
                    source: imp.source.clone(),
                    digest: imp.digest.clone(),
                },
                operands: Vec::new(),
                naming: simple(Vec::new()),
                import: true,
                instances: None,
                tools: None,
            }),
        }
    }

    /// A pattern or mirror of seeds at `instances`. `varied`: per instance, the seed tools
    /// rebuilt with that instance's values (a variable pattern).
    pub(super) fn pattern_like(
        &mut self,
        i: usize,
        id: &FeatureId,
        seeds: &PatternSeeds,
        instances: Vec<Instance>,
        varied: Vec<Vec<ToolOp>>,
    ) -> Result<Compiled, Halt> {
        let keys: BTreeSet<InstanceKey> = instances.iter().map(|x| x.key.clone()).collect();
        match seeds {
            PatternSeeds::Features { features } => {
                let mut tools = Vec::new();
                let mut seed_ids = Vec::new();
                let mut profiles = Vec::new();
                for (k, s) in features.iter().enumerate() {
                    self.gate_feature(i, id, s, &format!("seeds[{}]", k))?;
                    let Some(t) = self.st.tools.get(s).cloned() else {
                        let kind = self
                            .doc
                            .feature(s)
                            .map(|x| x.kind.name())
                            .unwrap_or("feature");
                        return Err(self.error(
                            id,
                            Some(&format!("seeds[{}]", k)),
                            DiagCode::KernelNotSupported,
                            format!(
                    "not supported yet: patterning a {} feature (or one that ends at a chosen \
                     face or body)",
                    kind
                ),
                            Vec::new(),
                        ));
                    };
                    let sketch = match self.doc.feature(s).map(|x| &x.kind) {
                        Some(FeatureKind::Extrude(e)) => vec![Self::profile_naming(s, &e.profile)],
                        Some(FeatureKind::Revolve(r)) => vec![Self::profile_naming(s, &r.profile)],
                        _ => Vec::new(),
                    };
                    for tool in t {
                        tools.push(tool);
                        seed_ids.push(s.clone());
                        profiles.push(sketch.clone());
                    }
                }
                let naming = NamingContext {
                    feature: id.clone(),
                    profiles,
                    seeds: seed_ids,
                    step: None,
                    restore: Vec::new(),
                };
                let operands = self.implicit_bodies(i, id)?;
                Ok(Compiled::Op {
                    op: KernelOp::PatternTools {
                        seeds: tools,
                        instances,
                        varied,
                    },
                    operands,
                    naming,
                    import: false,
                    instances: Some(keys),
                    tools: None,
                })
            }
            PatternSeeds::Bodies { bodies } => {
                let set = self.list(i, id, "bodies", bodies, Some(TopoKind::Body))?;
                Ok(Compiled::Op {
                    op: KernelOp::PatternBodies { instances },
                    operands: Self::bodies_of(&set),
                    naming: NamingContext::simple(id, Vec::new()),
                    import: false,
                    instances: Some(keys),
                    tools: None,
                })
            }
        }
    }
}

/// A face profile's operand: the face's body, among the combined bodies or else appended as a
/// reference the tool reads and returns unchanged.
fn profile_face(tool: &mut ToolOp, operands: &mut Vec<usize>, face: Option<EntityHandle>) {
    let Some(h) = face else {
        return;
    };
    let body = h.body as usize;
    let at = match operands.iter().position(|b| *b == body) {
        Some(k) => k,
        None => {
            operands.push(body);
            tool.references += 1;
            operands.len() - 1
        }
    };
    tool.profile_face = Some(OperandEntity {
        operand: at as u32,
        entity: h.index,
    });
}
