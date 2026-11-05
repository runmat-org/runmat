//! The fake kernel's canonical operations (`CadKernel::execute`) and rigid moves of solids.

use super::*;

impl CadKernel for FakeKernel {
    fn id(&self) -> KernelId {
        Self::id_static()
    }

    fn execute(
        &self,
        op: &KernelOp,
        operands: &[&BodyHandle],
    ) -> Result<KernelOutput, KernelError> {
        if let Some(out) = self.execute_extra(op, operands) {
            return out;
        }
        let solids: Vec<&FakeSolid> = (0..operands.len())
            .map(|i| Self::solid(operands, i))
            .collect::<Result<_, _>>()?;
        let ids: Vec<u32> = (0..operands.len() as u32).collect();
        match op {
            KernelOp::Tool(t) => self.tool_op(t, &solids),
            KernelOp::PatternTools {
                seeds,
                instances,
                varied,
            } => {
                let combine =
                    seeds
                        .first()
                        .map(|s| s.combine)
                        .ok_or_else(|| KernelError::Failed {
                            message: "pattern with no seeds".into(),
                        })?;
                if seeds.iter().any(|s| s.combine != combine) {
                    return Err(KernelError::NotSupported {
                        what: "fake kernel: patterns of mixed tools".into(),
                    });
                }
                if !varied.is_empty() && varied.len() != instances.len() {
                    return Err(KernelError::Failed {
                        message: "a variable pattern needs tools for every instance".into(),
                    });
                }
                if combine == Combine::New {
                    return self.pattern_new_bodies(seeds, instances, varied, &solids);
                }
                let mut tools = Vec::new();
                for (ii, inst) in instances.iter().enumerate() {
                    let these = if varied.is_empty() {
                        seeds
                    } else {
                        &varied[ii]
                    };
                    for (si, seed) in these.iter().enumerate() {
                        for p in &self.tools(seed, &solids, si as u32)? {
                            let mut q = p.transformed(&inst.transform);
                            q.relabel(si as u32, &inst.key);
                            tools.push(q);
                        }
                    }
                }
                let res = run_grid(GridJob {
                    operands: solids.clone(),
                    operand_ids: ids.clone(),
                    tools,
                    mode: Self::mode(combine),
                    split: None,
                    separate_tools: false,
                    body_label: None,
                    modifier: None,
                    merge_collinear: false,
                    carve: Vec::new(),
                })?;
                self.finish(&solids, &ids, res, Vec::new(), Vec::new())
            }
            KernelOp::FullRound {
                first,
                center,
                second,
            } => self.full_round(first, center, second, &solids),
            KernelOp::VertexChamfer {
                vertices,
                distances,
            } => self.vertex_chamfer(vertices, *distances, &solids),
            KernelOp::Fillet { .. } | KernelOp::Chamfer { .. } | KernelOp::FaceBlend { .. } => {
                let (role, radius, curved, g2, edges) = match op {
                    KernelOp::Fillet {
                        edges,
                        radius,
                        variable,
                        profile,
                        setbacks,
                    } => {
                        if !setbacks.is_empty() {
                            return Err(KernelError::NotSupported {
                                what: "fake kernel: setback corner blends".into(),
                            });
                        }
                        let r = variable.iter().map(|v| v.1).fold(*radius, f64::max);
                        (
                            Role::Blend,
                            r,
                            !variable.is_empty() || !profile.is_circular(),
                            matches!(profile, BlendProfile::Curvature),
                            edges.clone(),
                        )
                    }
                    KernelOp::FaceBlend {
                        first,
                        second,
                        radius,
                        profile,
                    } => (
                        Role::Blend,
                        *radius,
                        !profile.is_circular(),
                        matches!(profile, BlendProfile::Curvature),
                        self.face_blend(first, second, *radius, &solids)?,
                    ),
                    KernelOp::Chamfer { edges, size } => {
                        (Role::Chamfer, size.reach(), false, false, edges.clone())
                    }
                    _ => unreachable!("blends only"),
                };
                let edges = &edges;
                let radius = &radius;
                let mut fresh_all: Vec<Vec<Decoration>> = vec![Vec::new(); solids.len()];
                for e in edges {
                    let s = solids[e.operand as usize];
                    let meta = s
                        .edge_meta
                        .get(e.entity as usize)
                        .ok_or(KernelError::Failed {
                            message: "edge out of range".into(),
                        })?;
                    let planes = meta.planes.ok_or_else(|| KernelError::NotSupported {
                        what: "fake kernel: blending an edge of a blend or a curved edge".into(),
                    })?;
                    fresh_all[e.operand as usize].push(Decoration {
                        role,
                        size: *radius,
                        along: meta.along,
                        at: meta.at,
                        lo: meta.lo,
                        hi: meta.hi,
                        planes,
                        pieces: Vec::new(),
                        fresh: Some(gen(role, KernelTag::InputEdge { edge: *e }, None)),
                        curved,
                        g2,
                    });
                }
                // Each operand is rebuilt alone with its old and new decorations.
                let mut bodies = Vec::new();
                let mut history = OpHistory::default();
                for (k, s) in solids.iter().enumerate() {
                    let job = GridJob {
                        operands: vec![s],
                        operand_ids: vec![k as u32],
                        tools: Vec::new(),
                        mode: Mode::Keep,
                        split: None,
                        separate_tools: false,
                        body_label: None,
                        modifier: None,
                        merge_collinear: false,
                        carve: Vec::new(),
                    };
                    let res = run_grid(job)?;
                    let mut out = self.finish(
                        &[s],
                        &[k as u32],
                        res,
                        std::mem::take(&mut fresh_all[k]),
                        Vec::new(),
                    )?;
                    let base = bodies.len() as u32;
                    for r in &mut out.history.faces {
                        r.body += base;
                    }
                    for r in &mut out.history.bodies {
                        r.body += base;
                    }
                    history.faces.extend(out.history.faces);
                    history.bodies.extend(out.history.bodies);
                    history.deleted_faces.extend(out.history.deleted_faces);
                    history.deleted_bodies.extend(out.history.deleted_bodies);
                    bodies.extend(out.bodies);
                }
                Ok(KernelOutput { bodies, history })
            }
            KernelOp::Shell {
                open_faces,
                thickness,
                faces,
                outward,
            } => {
                let per_face: BTreeMap<u32, f64> =
                    faces.iter().map(|(e, t)| (e.entity, *t)).collect();
                if solids.len() != 1 {
                    return Err(KernelError::Failed {
                        message: "a shell works on one body".into(),
                    });
                }
                if *thickness <= 0.0 {
                    return Err(KernelError::Failed {
                        message: "shell thickness must be positive".into(),
                    });
                }
                if !solids[0].decorations.is_empty() {
                    return Err(KernelError::NotSupported {
                        what: "fake kernel: shelling a blended body".into(),
                    });
                }
                let open: BTreeSet<u32> = open_faces.iter().map(|e| e.entity).collect();
                let res = run_grid(GridJob {
                    operands: solids.clone(),
                    operand_ids: ids.clone(),
                    tools: Vec::new(),
                    mode: Mode::Keep,
                    split: None,
                    separate_tools: false,
                    body_label: None,
                    modifier: Some(Modifier::Shell(ShellSpec {
                        open,
                        thickness: *thickness,
                        per_face,
                        outward: *outward,
                    })),
                    merge_collinear: false,
                    carve: Vec::new(),
                })?;
                self.finish(&solids, &ids, res, Vec::new(), Vec::new())
            }
            KernelOp::Draft {
                faces,
                neutral,
                angle,
                parting,
                step: _,
            } => {
                let pull = neutral.z;
                let tan = angle.tan();
                let mut bodies = Vec::new();
                let mut history = OpHistory::default();
                for (k, so) in solids.iter().enumerate() {
                    let mut specs = Vec::new();
                    for e in faces.iter().filter(|e| e.operand as usize == k) {
                        let face = so.faces.get(e.entity as usize).ok_or(KernelError::Failed {
                            message: "draft face out of range".into(),
                        })?;
                        let plane = face.plane.ok_or_else(|| KernelError::NotSupported {
                            what: "fake kernel: drafting a curved face".into(),
                        })?;
                        if pull[plane.axis].abs() > 1e-9 {
                            return Err(KernelError::Failed {
                                message: "a drafted face must be parallel to the pull direction"
                                    .into(),
                            });
                        }
                        let mut lo = [f64::INFINITY; 3];
                        let mut hi = [f64::NEG_INFINITY; 3];
                        for r in &face.rects {
                            for a in 0..3 {
                                lo[a] = lo[a].min(r.lo[a]);
                                hi[a] = hi[a].max(r.hi[a]);
                            }
                        }
                        // A parting-line draft hinges each face about the parting edges it
                        // touches: its hinge height is theirs.
                        let mut hinge_at = neutral.origin;
                        let hinge: Vec<f64> = parting
                            .iter()
                            .filter(|p| p.operand as usize == k)
                            .filter_map(|p| so.topology.edges.get(p.entity as usize))
                            .filter(|ed| ed.faces.contains(&e.entity))
                            .flat_map(|ed| [ed.geom.start, ed.geom.end])
                            .map(|q| (0..3).map(|a| (q[a] - neutral.origin[a]) * pull[a]).sum())
                            .collect();
                        if !parting.is_empty() {
                            if hinge.is_empty() {
                                return Err(KernelError::Failed {
                                    message: "a drafted face touches no parting-line edge".into(),
                                });
                            }
                            let h = hinge.iter().sum::<f64>() / hinge.len() as f64;
                            for a in 0..3 {
                                hinge_at[a] += h * pull[a];
                            }
                        }
                        let mut max_h: f64 = 0.0;
                        for i in 0..8 {
                            let c = [
                                if i & 1 == 0 { lo[0] } else { hi[0] },
                                if i & 2 == 0 { lo[1] } else { hi[1] },
                                if i & 4 == 0 { lo[2] } else { hi[2] },
                            ];
                            let h: f64 = (0..3).map(|a| (c[a] - hinge_at[a]) * pull[a]).sum();
                            max_h = max_h.max(h.abs());
                        }
                        specs.push(DraftSpec {
                            face: e.entity,
                            plane,
                            lo,
                            hi,
                            origin: hinge_at,
                            pull,
                            tan,
                            reach: tan.abs() * max_h + EPS,
                            shift: 0.0,
                        });
                    }
                    let job = GridJob {
                        operands: vec![so],
                        operand_ids: vec![k as u32],
                        tools: Vec::new(),
                        mode: Mode::Keep,
                        split: None,
                        separate_tools: false,
                        body_label: None,
                        modifier: (!specs.is_empty()).then_some(Modifier::Draft(specs)),
                        merge_collinear: false,
                        carve: Vec::new(),
                    };
                    let res = run_grid(job)?;
                    let mut out = self.finish(&[so], &[k as u32], res, Vec::new(), Vec::new())?;
                    let base = bodies.len() as u32;
                    for r in &mut out.history.faces {
                        r.body += base;
                    }
                    for r in &mut out.history.bodies {
                        r.body += base;
                    }
                    history.faces.extend(out.history.faces);
                    history.bodies.extend(out.history.bodies);
                    history.deleted_faces.extend(out.history.deleted_faces);
                    history.deleted_bodies.extend(out.history.deleted_bodies);
                    bodies.extend(out.bodies);
                }
                Ok(KernelOutput { bodies, history })
            }
            KernelOp::PatternBodies { instances } => {
                let keep = run_grid(GridJob {
                    operands: solids.clone(),
                    operand_ids: ids.clone(),
                    tools: Vec::new(),
                    mode: Mode::Keep,
                    split: None,
                    separate_tools: false,
                    body_label: None,
                    modifier: None,
                    merge_collinear: false,
                    carve: Vec::new(),
                })?;
                let mut extra = Vec::new();
                for inst in instances {
                    let t = inst.transform;
                    if t.rotation != Transform::translation([0.0; 3]).rotation {
                        // Any other motion: re-sample each body's image on the grid.
                        for (k, s) in solids.iter().enumerate() {
                            let img = sampled::image(s, k as u32, &t, Some(&inst.key))?;
                            let res = run_grid(GridJob {
                                operands: Vec::new(),
                                operand_ids: Vec::new(),
                                tools: vec![Tool::Sampled(Box::new(img))],
                                mode: Mode::New,
                                split: None,
                                separate_tools: false,
                                body_label: Some(vec![HistorySource::Copy {
                                    from: OperandEntity {
                                        operand: k as u32,
                                        entity: 0,
                                    },
                                    instance: inst.key.clone(),
                                }]),
                                modifier: None,
                                merge_collinear: false,
                                carve: Vec::new(),
                            })?;
                            for ((c, f), b) in res
                                .solids
                                .into_iter()
                                .zip(res.face_sources)
                                .zip(res.body_sources)
                            {
                                extra.push((c, f, b));
                            }
                        }
                        continue;
                    }
                    for (k, s) in solids.iter().enumerate() {
                        let mut c = (*s).clone();
                        shift(&mut c, t.translation);
                        let srcs = (0..s.faces.len())
                            .map(|f| {
                                vec![HistorySource::Copy {
                                    from: OperandEntity {
                                        operand: k as u32,
                                        entity: f as u32,
                                    },
                                    instance: inst.key.clone(),
                                }]
                            })
                            .collect::<Vec<_>>();
                        // Drop blend faces from the grid face list; they are rebuilt.
                        let grid_faces = c.faces.iter().take_while(|f| f.is_grid()).count();
                        c.faces.truncate(grid_faces);
                        let srcs = srcs.into_iter().take(grid_faces).collect();
                        extra.push((
                            c,
                            srcs,
                            vec![HistorySource::Copy {
                                from: OperandEntity {
                                    operand: k as u32,
                                    entity: 0,
                                },
                                instance: inst.key.clone(),
                            }],
                        ));
                    }
                }
                self.finish(&solids, &ids, keep, Vec::new(), extra)
            }
            KernelOp::Boolean {
                combine,
                keep_tools,
            } => {
                if solids.is_empty() {
                    return Err(KernelError::Failed {
                        message: "boolean without a target".into(),
                    });
                }
                let mode = match combine {
                    Combine::Add => Mode::BoolUnion,
                    Combine::Cut => Mode::BoolCut,
                    Combine::Intersect => Mode::BoolIntersect,
                    Combine::New => {
                        return Err(KernelError::Failed {
                            message: "boolean needs a combine".into(),
                        })
                    }
                };
                let res = run_grid(GridJob {
                    operands: solids.clone(),
                    operand_ids: ids.clone(),
                    tools: Vec::new(),
                    mode,
                    split: None,
                    separate_tools: false,
                    body_label: None,
                    modifier: None,
                    merge_collinear: false,
                    carve: Vec::new(),
                })?;
                let mut extra = Vec::new();
                if *keep_tools && mode != Mode::BoolUnion {
                    for k in 1..solids.len() {
                        let alone = run_grid(GridJob {
                            operands: vec![solids[k]],
                            operand_ids: vec![k as u32],
                            tools: Vec::new(),
                            mode: Mode::Keep,
                            split: None,
                            separate_tools: false,
                            body_label: None,
                            modifier: None,
                            merge_collinear: false,
                            carve: Vec::new(),
                        })?;
                        for ((s, f), b) in alone
                            .solids
                            .into_iter()
                            .zip(alone.face_sources)
                            .zip(alone.body_sources)
                        {
                            extra.push((s, f, b));
                        }
                    }
                }
                self.finish(&solids, &ids, res, Vec::new(), extra)
            }
            KernelOp::Split { frame } => {
                let (a, s) = axis_of(frame.z).ok_or_else(|| KernelError::NotSupported {
                    what: "fake kernel: split planes not normal to an axis".into(),
                })?;
                let c = frame.origin[a];
                // Each operand alone, so only bodies the plane crosses get side tags.
                let mut bodies = Vec::new();
                let mut history = OpHistory::default();
                for (k, so) in solids.iter().enumerate() {
                    let crosses = so
                        .bounds()
                        .map(|(lo, hi)| lo[a] < c - EPS && hi[a] > c + EPS)
                        .unwrap_or(false);
                    let job = GridJob {
                        operands: vec![so],
                        operand_ids: vec![k as u32],
                        tools: Vec::new(),
                        mode: if crosses { Mode::Split } else { Mode::Keep },
                        split: crosses.then_some((a, c, s)),
                        separate_tools: false,
                        body_label: None,
                        modifier: None,
                        merge_collinear: false,
                        carve: Vec::new(),
                    };
                    let res = run_grid(job)?;
                    let mut out = self.finish(&[so], &[k as u32], res, Vec::new(), Vec::new())?;
                    let base = bodies.len() as u32;
                    for r in &mut out.history.faces {
                        r.body += base;
                    }
                    for r in &mut out.history.bodies {
                        r.body += base;
                    }
                    history.faces.extend(out.history.faces);
                    history.bodies.extend(out.history.bodies);
                    history.deleted_faces.extend(out.history.deleted_faces);
                    history.deleted_bodies.extend(out.history.deleted_bodies);
                    bodies.extend(out.bodies);
                }
                Ok(KernelOutput { bodies, history })
            }
            KernelOp::Transform { transform } => {
                if transform.rotation != Transform::translation([0.0; 3]).rotation {
                    // Rotations and reflections: each body re-sampled at its new place.
                    let mut bodies = Vec::new();
                    let mut history = OpHistory::default();
                    for (k, so) in solids.iter().enumerate() {
                        let img = sampled::image(so, k as u32, transform, None)?;
                        let res = run_grid(GridJob {
                            operands: Vec::new(),
                            operand_ids: Vec::new(),
                            tools: vec![Tool::Sampled(Box::new(img))],
                            mode: Mode::New,
                            split: None,
                            separate_tools: false,
                            body_label: Some(vec![HistorySource::Input {
                                from: OperandEntity {
                                    operand: k as u32,
                                    entity: 0,
                                },
                            }]),
                            modifier: None,
                            merge_collinear: false,
                            carve: Vec::new(),
                        })?;
                        let mut out = self.finish(&[], &[], res, Vec::new(), Vec::new())?;
                        // Faces and the body continue the operand's.
                        out.history.deleted_faces.clear();
                        out.history.deleted_bodies.clear();
                        let used: BTreeSet<u32> = out
                            .history
                            .faces
                            .iter()
                            .flat_map(|r| r.sources.iter())
                            .filter_map(|s| match s {
                                HistorySource::Input { from } => Some(from.entity),
                                _ => None,
                            })
                            .collect();
                        for f in 0..so.faces.len() as u32 {
                            if !used.contains(&f) {
                                history.deleted_faces.push(OperandEntity {
                                    operand: k as u32,
                                    entity: f,
                                });
                            }
                        }
                        for r in &mut out.history.faces {
                            r.body += bodies.len() as u32;
                            r.evolution = Evolution::Modified;
                        }
                        for r in &mut out.history.bodies {
                            r.body += bodies.len() as u32;
                        }
                        history.faces.extend(out.history.faces);
                        history.bodies.extend(out.history.bodies);
                        bodies.extend(out.bodies);
                    }
                    return Ok(KernelOutput { bodies, history });
                }
                let moved: Vec<FakeSolid> = solids
                    .iter()
                    .map(|s| {
                        let mut c = (*s).clone();
                        shift(&mut c, transform.translation);
                        c
                    })
                    .collect();
                let refs: Vec<&FakeSolid> = moved.iter().collect();
                // Re-grid the moved solids so labels follow the moved faces.
                let res = run_grid(GridJob {
                    operands: refs.clone(),
                    operand_ids: ids.clone(),
                    tools: Vec::new(),
                    mode: Mode::Keep,
                    split: None,
                    separate_tools: false,
                    body_label: None,
                    modifier: None,
                    merge_collinear: false,
                    carve: Vec::new(),
                })?;
                self.finish(&refs, &ids, res, Vec::new(), Vec::new())
            }
            KernelOp::Import { source, .. } => {
                let boxes = self
                    .imports
                    .get(source)
                    .ok_or_else(|| KernelError::Failed {
                        message: format!("cannot read `{}`", source),
                    })?;
                let mut tools = Vec::new();
                let mut body_labels = Vec::new();
                for b in boxes {
                    let lab = |i: usize| {
                        gen(
                            Role::Side,
                            KernelTag::CadName {
                                name: b.face_names[i].clone(),
                            },
                            None,
                        )
                    };
                    // A box is a prism along z of a rectangle in x/y.
                    let pts = vec![
                        [b.min[0], b.min[1]],
                        [b.max[0], b.min[1]],
                        [b.max[0], b.max[1]],
                        [b.min[0], b.max[1]],
                    ];
                    let labels = vec![lab(2), lab(1), lab(3), lab(0)];
                    tools.push(Tool::Exact(Prism {
                        k: 2,
                        u: 0,
                        v: 1,
                        loops: vec![pts],
                        edge_labels: vec![labels],
                        lo: b.min[2],
                        hi: b.max[2],
                        lo_label: lab(4),
                        hi_label: lab(5),
                    }));
                    body_labels.push(gen(
                        Role::Body,
                        KernelTag::CadName {
                            name: b.name.clone(),
                        },
                        None,
                    ));
                }
                let res = run_grid(GridJob {
                    operands: Vec::new(),
                    operand_ids: Vec::new(),
                    tools,
                    mode: Mode::New,
                    split: None,
                    separate_tools: true,
                    body_label: Some(body_labels),
                    modifier: None,
                    merge_collinear: false,
                    carve: Vec::new(),
                })?;
                self.finish(&[], &[], res, Vec::new(), Vec::new())
            }
            // Handled by `execute_extra`.
            KernelOp::Labeled { .. }
            | KernelOp::MoveFaces { .. }
            | KernelOp::DeleteFaces { .. }
            | KernelOp::ReplaceFaces { .. }
            | KernelOp::Scale { .. }
            | KernelOp::Surface(_) => unreachable!("dispatched first"),
        }
    }
}

pub(super) fn shift(s: &mut FakeSolid, t: [f64; 3]) {
    for a in 0..3 {
        for g in &mut s.grid[a] {
            *g += t[a];
        }
    }
    for f in &mut s.faces {
        if let Some(p) = &mut f.plane {
            p.offset += t[p.axis];
        }
        for p in &mut f.facet_planes {
            p.offset += t[p.axis];
        }
        if let Some(h) = &mut f.surface {
            *h = sampled::move_hint(h, &Transform::translation(t));
        }
        if let Some(e) = &mut f.exact {
            *e = Arc::new(e.transformed(&Transform::translation(t)));
        }
        for r in &mut f.rects {
            for a in 0..3 {
                r.lo[a] += t[a];
                r.hi[a] += t[a];
            }
        }
    }
    for d in &mut s.decorations {
        for a in 0..3 {
            d.at[a] += t[a];
        }
        d.lo += t[d.along];
        d.hi += t[d.along];
        for p in &mut d.planes {
            p.offset += t[p.axis];
        }
        for piece in &mut d.pieces {
            piece.0 += t[d.along];
            piece.1 += t[d.along];
        }
    }
}

impl FakeKernel {
    /// A pattern of new-body seeds: one body per instance and seed (as OCCT makes them); the
    /// operands are kept as they are.
    fn pattern_new_bodies(
        &self,
        seeds: &[ToolOp],
        instances: &[Instance],
        varied: &[Vec<ToolOp>],
        solids: &[&FakeSolid],
    ) -> Result<KernelOutput, KernelError> {
        let mut outs = Vec::new();
        for (ii, inst) in instances.iter().enumerate() {
            let these = if varied.is_empty() { seeds } else { &varied[ii] };
            for (si, seed) in these.iter().enumerate() {
                let mut tools = Vec::new();
                for p in &self.tools(seed, solids, si as u32)? {
                    let mut q = p.transformed(&inst.transform);
                    q.relabel(si as u32, &inst.key);
                    tools.push(q);
                }
                let label = HistorySource::Generated {
                    role: Role::Body,
                    tag: KernelTag::None,
                    ordinal: None,
                    seed: Some(si as u32),
                    instance: Some(inst.key.clone()),
                };
                let res = run_grid(GridJob {
                    operands: Vec::new(),
                    operand_ids: Vec::new(),
                    body_label: Some(vec![label; tools.len()]),
                    tools,
                    mode: Mode::New,
                    split: None,
                    separate_tools: false,
                    modifier: None,
                    merge_collinear: seed.shape.merge_collinear(),
                    carve: Vec::new(),
                })?;
                outs.push(self.finish(&[], &[], res, Vec::new(), Vec::new())?);
            }
        }
        for (k, s) in solids.iter().enumerate() {
            let res = run_grid(super::features::keep_job(vec![*s], vec![k as u32]))?;
            let mut out = self.finish(&[*s], &[k as u32], res, Vec::new(), Vec::new())?;
            for r in &mut out.history.faces {
                r.evolution = Evolution::Kept;
            }
            outs.push(out);
        }
        Ok(super::features::concat(outs))
    }
}
