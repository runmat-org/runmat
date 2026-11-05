//! Sheet metal: the rules a part's sheet bodies follow, the sheet-metal features' lowering, and
//! the shared steps that place a feature's result in its body.

use super::*;

impl<'a, 'b> Run<'a, 'b> {
    pub(super) fn sheet_rules(
        &mut self,
        id: &FeatureId,
        r: &SheetMetalRules,
    ) -> Result<SheetRules, Halt> {
        let (mut t, mut radius) = (
            self.value(id, "rules.thickness", &r.thickness, Dimension::LENGTH)?,
            self.value(id, "rules.bend_radius", &r.bend_radius, Dimension::LENGTH)?,
        );
        let mut rule = match &r.rule {
            BendRule::KFactor { k } => BendRuleValue::KFactor {
                k: self.value(id, "rules.k", k, Dimension::DIMENSIONLESS)?,
            },
            BendRule::Table { table } => BendRuleValue::Table {
                table: table.clone(),
            },
            BendRule::Deduction { value } => BendRuleValue::Deduction {
                value: self.value(id, "rules.deduction", value, Dimension::LENGTH)?,
            },
            BendRule::Allowance { value } => BendRuleValue::Allowance {
                value: self.value(id, "rules.allowance", value, Dimension::LENGTH)?,
            },
        };
        if let Some((table, gauge)) = &r.gauge {
            let Some(row) = table.rows.iter().find(|g| &g.gauge == gauge) else {
                return Err(self.error(
                    id,
                    Some("rules.gauge"),
                    DiagCode::RegenInvalid,
                    format!("gauge table `{}` has no gauge `{}`", table.name, gauge),
                    Vec::new(),
                ));
            };
            t = row.thickness;
            radius = row.bend_radius;
            if matches!(r.rule, BendRule::KFactor { .. }) {
                rule = BendRuleValue::KFactor { k: row.k_factor };
            }
        }
        if t <= 0.0 || radius < 0.0 {
            return Err(self.error(
                id,
                Some("rules"),
                DiagCode::RegenInvalid,
                "sheet thickness must be positive and the bend radius non-negative".into(),
                Vec::new(),
            ));
        }
        Ok(SheetRules {
            thickness: t,
            bend_radius: radius,
            rule,
        })
    }

    /// The sheet record of the base flange that made a body.
    pub(in crate::regen::engine) fn sheet_of(
        &self,
        body: usize,
    ) -> Option<(
        FeatureId,
        SheetRules,
        Option<crate::model::sheet_metal::Section>,
        f64,
    )> {
        for n in &self.st.bodies[body].names.body {
            if let Some(FeatureRecord::Sheet {
                rules,
                section,
                depth,
            }) = self.st.records.get(n.maker())
            {
                return Some((n.maker().clone(), rules.clone(), section.clone(), *depth));
            }
        }
        None
    }

    pub(super) fn sheet_metal(
        &mut self,
        i: usize,
        id: &FeatureId,
        s: &SheetMetalFeature,
    ) -> Result<Compiled, Halt> {
        let simple = NamingContext::simple(id, Vec::new());
        match s {
            SheetMetalFeature::BaseFlange {
                profile,
                depth,
                side,
                rules,
            } => {
                let rules = self.sheet_rules(id, rules)?;
                let (t, r) = (rules.thickness, rules.bend_radius);
                let body = None;
                let Some(depth) = depth else {
                    // A closed profile: a flat plate.
                    let p = self.profile(i, id, "profile", profile)?;
                    let mut tool = LabeledTool::new(ToolShape::prism(p.clone(), 0.0, t));
                    for seg in p.loops.iter().flatten() {
                        tool.sides.push((
                            seg.curve.clone(),
                            FaceLabel::key(Role::SheetEdge, seg.curve.0.clone()),
                        ));
                    }
                    tool.start = Some(FaceLabel::key(Role::SheetFace, "plate").with_ordinal(0));
                    tool.end = Some(FaceLabel::key(Role::SheetFace, "plate").with_ordinal(1));
                    tool.body = body;
                    self.st.records.insert(
                        id.clone(),
                        FeatureRecord::Sheet {
                            rules,
                            section: None,
                            depth: 0.0,
                        },
                    );
                    return Ok(op(
                        labeled(vec![tool], Vec::new(), Combine::New, false),
                        Vec::new(),
                        simple,
                    ));
                };
                let depth = self.value(id, "depth", depth, Dimension::LENGTH)?;
                let (frame, segs) = self.chain(i, id, "profile", profile)?;
                if segs.iter().any(|s| !matches!(s.kind, SegmentKind::Line)) {
                    return Err(self.error(
                        id,
                        Some("profile"),
                        DiagCode::SketchOpenProfile,
                        "a sheet metal chain is drawn with lines; bends come from its corners"
                            .into(),
                        Vec::new(),
                    ));
                }
                if r <= 0.0 {
                    return Err(self.error(
                        id,
                        Some("rules.bend_radius"),
                        DiagCode::RegenInvalid,
                        "bent sheets need a positive bend radius".into(),
                        Vec::new(),
                    ));
                }
                let mut pts: Vec<[f64; 2]> = segs.iter().map(|s| s.start).collect();
                pts.push(segs.last().expect("segments").end);
                let curves: Vec<_> = segs.iter().map(|s| s.curve.clone()).collect();
                let sec = section(&pts, &curves, *side, t, r).map_err(|m| {
                    self.error(id, Some("profile"), DiagCode::RegenInvalid, m, Vec::new())
                })?;
                let mut tools = Vec::new();
                let nflats = sec.flats.len();
                for (k, f) in sec.flats.iter().enumerate() {
                    let off =
                        |p: [f64; 2], o: f64| [p[0] + o * f.normal[0], p[1] + o * f.normal[1]];
                    let c = f.curve.0.clone();
                    let corners = vec![
                        (format!("{}.near", c), off(f.start, f.near)),
                        (format!("{}.b", c), off(f.end, f.near)),
                        (format!("{}.far", c), off(f.end, f.far)),
                        (format!("{}.a", c), off(f.start, f.far)),
                    ];
                    let mut tool =
                        LabeledTool::new(ToolShape::prism(polygon(frame, &corners), 0.0, depth));
                    let sid = |s: &str| crate::ids::SketchEntityId::new(format!("{}.{}", c, s));
                    tool.sides.push((
                        sid("near"),
                        FaceLabel::key(Role::SheetFace, c.clone()).with_ordinal(0),
                    ));
                    tool.sides.push((
                        sid("far"),
                        FaceLabel::key(Role::SheetFace, c.clone()).with_ordinal(1),
                    ));
                    // Chain ends are sheet edges; ends meeting a bend lie inside the body.
                    tool.sides.push((
                        sid("a"),
                        FaceLabel::key(Role::SheetEdge, c.clone()).with_ordinal(if k == 0 {
                            0
                        } else {
                            4
                        }),
                    ));
                    tool.sides.push((
                        sid("b"),
                        FaceLabel::key(Role::SheetEdge, c.clone())
                            .with_ordinal(if k + 1 == nflats { 1 } else { 5 }),
                    ));
                    tool.start = Some(FaceLabel::key(Role::SheetEdge, c.clone()).with_ordinal(2));
                    tool.end = Some(FaceLabel::key(Role::SheetEdge, c.clone()).with_ordinal(3));
                    tool.body = body.clone();
                    tools.push(tool);
                }
                for b in &sec.bends {
                    tools.push(bend_tool(frame, b, depth, body.clone()));
                }
                self.st.records.insert(
                    id.clone(),
                    FeatureRecord::Sheet {
                        rules,
                        section: Some(sec),
                        depth,
                    },
                );
                Ok(op(
                    labeled(tools, Vec::new(), Combine::New, false),
                    Vec::new(),
                    simple,
                ))
            }
            SheetMetalFeature::EdgeFlange {
                edges,
                length,
                angle,
                length_reference,
                position,
                radius,
                ..
            } => {
                let l = self.value(id, "length", length, Dimension::LENGTH)?;
                let theta = self.value(id, "angle", angle, Dimension::ANGLE)?;
                let rr = match radius {
                    Some(r) => Some(self.value(id, "radius", r, Dimension::LENGTH)?),
                    None => None,
                };
                self.flange(
                    i,
                    id,
                    edges,
                    l,
                    theta,
                    *length_reference,
                    *position,
                    rr,
                    &simple,
                )
            }
            SheetMetalFeature::Hem {
                edges,
                kind,
                length,
            } => {
                let l = self.value(id, "length", length, Dimension::LENGTH)?;
                let gap = match kind {
                    HemKind::Closed => None,
                    HemKind::Open { gap } => {
                        Some(self.value(id, "kind.gap", gap, Dimension::LENGTH)?)
                    }
                    HemKind::Teardrop { radius, angle } | HemKind::Rolled { radius, angle } => {
                        // A roll of `angle` at inner radius `radius`, then the straight leg
                        // of `length` (a teardrop's leg runs back towards the sheet).
                        let r = self.value(id, "kind.radius", radius, Dimension::LENGTH)?;
                        let theta = self.value(id, "kind.angle", angle, Dimension::ANGLE)?;
                        if r <= 0.0 {
                            return Err(self.error(
                                id,
                                Some("kind.radius"),
                                DiagCode::RegenInvalid,
                                "a rolled or teardrop hem needs a positive radius".into(),
                                Vec::new(),
                            ));
                        }
                        if matches!(kind, HemKind::Teardrop { .. })
                            && theta <= std::f64::consts::PI + 1e-12
                        {
                            return Err(self.error(
                                id,
                                Some("kind.angle"),
                                DiagCode::RegenInvalid,
                                "a teardrop hem turns more than 180°; use an open hem".into(),
                                Vec::new(),
                            ));
                        }
                        return self.flange_with(
                            i,
                            id,
                            edges,
                            l,
                            theta,
                            LengthReference::Tangent,
                            crate::model::sheet_metal::FlangePosition::BendOutside,
                            Some(r),
                            &simple,
                            true,
                        );
                    }
                };
                // A closed hem folds flat on the sheet: an inner radius of a hundredth of the
                // thickness (the fold line); an open hem leaves `gap` between the faces.
                let set = self.list(i, id, "edges", edges, Some(TopoKind::Edge))?;
                let body = set.iter().next().map(|h| h.body as usize).unwrap_or(0);
                let t = self.sheet_of(body).map(|s| s.1.thickness).unwrap_or(0.0);
                let r = gap.map(|g| g / 2.0).unwrap_or(t * 0.01);
                self.flange(
                    i,
                    id,
                    edges,
                    l,
                    std::f64::consts::PI,
                    LengthReference::Tangent,
                    crate::model::sheet_metal::FlangePosition::BendOutside,
                    Some(r),
                    &simple,
                )
            }
            SheetMetalFeature::BreakCorner { edges, kind } => {
                let set = self.list(i, id, "edges", edges, Some(TopoKind::Edge))?;
                let operands = Self::bodies_of(&set);
                let es = Self::operand_entities(&set, &operands);
                Ok(op(
                    match kind {
                        crate::model::sheet_metal::BreakKind::Fillet { radius } => {
                            KernelOp::Fillet {
                                edges: es,
                                radius: self.value(id, "kind.radius", radius, Dimension::LENGTH)?,
                                variable: Vec::new(),
                                profile: BlendProfile::Circular,
                                setbacks: Vec::new(),
                            }
                        }
                        crate::model::sheet_metal::BreakKind::Chamfer { distance } => {
                            KernelOp::Chamfer {
                                edges: es,
                                size: ChamferSize::Equal {
                                    distance: self.value(
                                        id,
                                        "kind.distance",
                                        distance,
                                        Dimension::LENGTH,
                                    )?,
                                },
                            }
                        }
                    },
                    operands,
                    simple,
                ))
            }
            SheetMetalFeature::FlatPattern { fixed_face, .. } => {
                let h = self.face_handle(i, id, "fixed_face", fixed_face)?;
                if let Some((maker, rules, None, _)) = self.sheet_of(h.body as usize) {
                    return self.plate_flat(i, id, &h, &maker, &rules, &simple, true);
                }
                let Some((maker, rules, Some(sec), depth)) = self.sheet_of(h.body as usize) else {
                    return Err(self.error(
                        id,
                        Some("fixed_face"),
                        DiagCode::RefKindMismatch,
                        "the fixed face is not on a bent sheet".into(),
                        Vec::new(),
                    ));
                };
                let names = self.st.entity_names(&h);
                let fixed = names.iter().find_map(|n| match n {
                    EntityName::Generated(g) if g.feature == maker && g.role == Role::SheetFace => {
                        match &g.tag {
                            crate::refs::name::SourceTag::Key { key } => Some(key.clone()),
                            _ => None,
                        }
                    }
                    _ => None,
                });
                let Some(fixed) = fixed else {
                    return Err(self.error(
                        id,
                        Some("fixed_face"),
                        DiagCode::RefKindMismatch,
                        "the fixed face is not a flat of the sheet".into(),
                        Vec::new(),
                    ));
                };
                let dev = crate::model::sheet_metal::develop(&sec, &rules.rule)
                    .map_err(|m| self.error(id, None, DiagCode::RegenInvalid, m, Vec::new()))?;
                let Some(((_, frame), flat)) = self
                    .st
                    .sketches
                    .get(&self.sketch_of(&maker))
                    .cloned()
                    .zip(sec.flats.iter().find(|f| f.curve.0 == fixed).cloned())
                else {
                    return Err(self.error(
                        id,
                        None,
                        DiagCode::RegenInvalid,
                        "the sheet's sketch is gone".into(),
                        Vec::new(),
                    ));
                };
                let xf = dev
                    .regions
                    .iter()
                    .find_map(|r| match r {
                        StripRegion::Flat { curve, start, .. } if curve.0 == fixed => Some(*start),
                        _ => None,
                    })
                    .unwrap_or(0.0);
                let p0 = frame.to_world([
                    flat.start[0] + flat.near * flat.normal[0],
                    flat.start[1] + flat.near * flat.normal[1],
                ]);
                let p1 = frame.to_world([
                    flat.end[0] + flat.near * flat.normal[0],
                    flat.end[1] + flat.near * flat.normal[1],
                ]);
                let xdir = normalize(sub(p1, p0));
                let n3 = normalize(add(
                    scale(frame.x, flat.normal[0]),
                    scale(frame.y, flat.normal[1]),
                ));
                let fr = frame_of(add(p0, scale(xdir, -xf)), xdir, frame.z);
                let t = rules.thickness;
                let thick = if dot(fr.z, n3) >= 0.0 { t } else { -t };
                let mut tools = Vec::new();
                for r in &dev.regions {
                    let (key, start, len, role) = match r {
                        StripRegion::Flat {
                            curve,
                            start,
                            length,
                        } => (curve.0.clone(), *start, *length, Role::SheetFace),
                        StripRegion::Bend { key, start, length } => {
                            (key.clone(), *start, *length, Role::BendFace)
                        }
                    };
                    let corners = vec![
                        (format!("{}.y0", key), [start, 0.0]),
                        (format!("{}.x1", key), [start + len, 0.0]),
                        (format!("{}.y1", key), [start + len, depth]),
                        (format!("{}.x0", key), [start, depth]),
                    ];
                    let mut tool =
                        LabeledTool::new(ToolShape::prism(polygon(fr, &corners), 0.0, thick));
                    let sid = |s: &str| crate::ids::SketchEntityId::new(format!("{}.{}", key, s));
                    for (s, o) in [("y0", 2), ("y1", 3), ("x0", 0), ("x1", 1)] {
                        tool.sides.push((
                            sid(s),
                            FaceLabel::key(Role::SheetEdge, key.clone()).with_ordinal(o),
                        ));
                    }
                    tool.start = Some(FaceLabel::key(role, key.clone()).with_ordinal(0));
                    tool.end = Some(FaceLabel::key(role, key.clone()).with_ordinal(1));
                    tools.push(tool);
                }
                // The layout: the strip in `fr`, its bend lines across it.
                let up_side = dot(fr.z, n3) >= 0.0;
                let corner = |x: f64, y: f64| [x, y];
                let outline = [
                    ("y0", corner(0.0, 0.0), corner(dev.length, 0.0)),
                    ("x1", corner(dev.length, 0.0), corner(dev.length, depth)),
                    ("y1", corner(dev.length, depth), corner(0.0, depth)),
                    ("x0", corner(0.0, depth), corner(0.0, 0.0)),
                ];
                let layout = crate::model::sheet_metal::FlatLayout {
                    frame: fr,
                    depth: thick,
                    loops: vec![outline
                        .iter()
                        .map(|(c, a, b)| line_seg(c, *a, *b))
                        .collect()],
                    bend_lines: dev
                        .bend_lines
                        .iter()
                        .map(|b| crate::model::sheet_metal::FlatBendLine {
                            key: b.key.clone(),
                            feature: maker.clone(),
                            start: [b.position, 0.0],
                            end: [b.position, depth],
                            angle: b.angle,
                            inner_radius: b.inner_radius,
                            allowance: b.allowance,
                            deduction: b.deduction,
                            direction: if (b.direction
                                == crate::model::sheet_metal::BendDirection::Up)
                                == up_side
                            {
                                crate::model::sheet_metal::BendDirection::Up
                            } else {
                                crate::model::sheet_metal::BendDirection::Down
                            },
                        })
                        .collect(),
                };
                self.st.records.insert(
                    id.clone(),
                    FeatureRecord::FlatPattern {
                        layout,
                        development: Some(dev),
                    },
                );
                Ok(op(
                    labeled(tools, Vec::new(), Combine::New, false),
                    Vec::new(),
                    simple,
                ))
            }
            SheetMetalFeature::MiterFlange {
                profile,
                edges,
                gap,
            } => self.miter_flange(i, id, profile, edges, gap, &simple),
            SheetMetalFeature::Jog {
                face,
                line,
                offset,
                angle,
            } => {
                let theta = self.value(id, "angle", angle, Dimension::ANGLE)?;
                let rise = self.value(id, "offset", offset, Dimension::LENGTH)?;
                self.bend_chain(i, id, face, line, theta, None, Some(rise), None, &simple)
            }
            SheetMetalFeature::SketchedBend {
                face,
                line,
                angle,
                radius,
                fixed,
            } => {
                let theta = self.value(id, "angle", angle, Dimension::ANGLE)?;
                let r = match radius {
                    Some(r) => Some(self.value(id, "radius", r, Dimension::LENGTH)?),
                    None => None,
                };
                self.bend_chain(i, id, face, line, theta, r, None, Some(*fixed), &simple)
            }
            SheetMetalFeature::LoftedBend { .. } => Err(self.not_lowered(id, "lofted bends")),
            SheetMetalFeature::ConvertToSheetMetal { .. } => {
                Err(self.not_lowered(id, "convert to sheet metal"))
            }
            SheetMetalFeature::ClosedCorner {
                faces,
                gap,
                overlap,
            } => {
                let gap = self.value(id, "gap", gap, Dimension::LENGTH)?;
                self.closed_corner(i, id, faces, gap, *overlap, &simple)
            }
            SheetMetalFeature::CornerRelief { corners, kind } => {
                self.corner_relief(i, id, corners, kind, &simple)
            }
            SheetMetalFeature::TabAndSlot { .. } => Err(self.not_lowered(id, "tabs and slots")),
            SheetMetalFeature::FormTool {
                tool,
                face,
                at,
                rotation,
            } => self.form_tool(i, id, tool, face, at, rotation, &simple),
            SheetMetalFeature::Unfold { fixed_face, bends } => {
                self.unfold(i, id, fixed_face, bends, &simple)
            }
            SheetMetalFeature::Fold { bends } => self.fold(i, id, bends, &simple),
        }
    }

    /// Refuses a whole-sheet rebuild (unfold, fold) of a body that features besides `makers`
    /// changed: the rebuild comes from the base flange's section and would drop them.
    pub(in crate::regen::engine) fn only_made_by(
        &mut self,
        id: &FeatureId,
        body: usize,
        makers: &[FeatureId],
        what: &str,
    ) -> Result<(), Halt> {
        let changed = self.st.bodies[body]
            .names
            .faces
            .iter()
            .any(|names| !names.iter().any(|n| makers.contains(n.maker())));
        if changed {
            return Err(self.not_lowered(
                id,
                &format!(
                    "{what} of a sheet that features after its base flange changed (flanges, hems, cuts)"
                ),
            ));
        }
        Ok(())
    }

    /// A new-body sheet op turned into a rebuild of `body` in place: the body is carved away
    /// whole and the op's tools take its place.
    pub(in crate::regen::engine) fn in_place(&self, compiled: Compiled, body: usize) -> Compiled {
        if let Compiled::Steps(mut steps) = compiled {
            // The first step rebuilds the body in place; later steps read it where it is.
            let new_body = self.st.bodies.len();
            if let Some(first) = steps.first_mut() {
                let naming = first.naming.clone();
                if let Compiled::Op { op, operands, .. } = self.in_place(
                    Compiled::Op {
                        op: first.op.clone(),
                        operands: first.operands.clone(),
                        naming,
                        import: false,
                        instances: None,
                        tools: None,
                    },
                    body,
                ) {
                    first.op = op;
                    first.operands = operands;
                }
            }
            for s in steps.iter_mut().skip(1) {
                for o in s.operands.iter_mut() {
                    if *o == new_body {
                        *o = body;
                    }
                }
            }
            return Compiled::Steps(steps);
        }
        let Compiled::Op {
            op: KernelOp::Labeled { tools, .. },
            naming,
            ..
        } = compiled
        else {
            return compiled;
        };
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for f in &self.st.bodies[body].topology.faces {
            for k in 0..3 {
                lo[k] = lo[k].min(f.geom.bbox_min[k]);
                hi[k] = hi[k].max(f.geom.bbox_max[k]);
            }
        }
        let margin = 1e-3 + 0.01 * (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max);
        let fr = frame_of([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let corners = vec![
            ("c0".to_string(), [lo[0] - margin, lo[1] - margin]),
            ("c1".to_string(), [hi[0] + margin, lo[1] - margin]),
            ("c2".to_string(), [hi[0] + margin, hi[1] + margin]),
            ("c3".to_string(), [lo[0] - margin, hi[1] + margin]),
        ];
        let carve = LabeledTool::new(ToolShape::prism(
            polygon(fr, &corners),
            lo[2] - margin,
            hi[2] + margin,
        ));
        let tools = tools
            .into_iter()
            .map(|mut t| {
                t.body = None;
                t
            })
            .collect();
        op(
            labeled(tools, vec![carve], Combine::Add, false),
            vec![body],
            naming,
        )
    }

    pub(in crate::regen::engine) fn sketch_of(&self, base: &FeatureId) -> FeatureId {
        match self.doc.feature(base).map(|f| &f.kind) {
            Some(FeatureKind::SheetMetal(SheetMetalFeature::BaseFlange { profile, .. })) => {
                profile.sketch.clone()
            }
            _ => base.clone(),
        }
    }
}
