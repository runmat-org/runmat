//! Flat strips: the bends a strip section keys, flattening and unfolding strips, and the
//! features that rebuild a sheet.

use super::*;

/// A rigid motion of the section plane: `p ↦ R(angle) p + shift`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Motion2 {
    pub(super) angle: f64,
    pub(super) shift: [f64; 2],
}

impl Motion2 {
    const IDENTITY: Motion2 = Motion2 {
        angle: 0.0,
        shift: [0.0, 0.0],
    };
    pub(super) fn turn(&self, v: [f64; 2]) -> [f64; 2] {
        let (c, s) = (self.angle.cos(), self.angle.sin());
        [c * v[0] - s * v[1], s * v[0] + c * v[1]]
    }
    pub(super) fn apply(&self, p: [f64; 2]) -> [f64; 2] {
        let q = self.turn(p);
        [q[0] + self.shift[0], q[1] + self.shift[1]]
    }
    /// The motion that turns direction `d` to `d2` and carries `p` to `p2`.
    pub(super) fn carrying(p: [f64; 2], d: [f64; 2], p2: [f64; 2], d2: [f64; 2]) -> Motion2 {
        let angle = d2[1].atan2(d2[0]) - d[1].atan2(d[0]);
        let m = Motion2 {
            angle,
            shift: [0.0, 0.0],
        };
        let q = m.turn(p);
        Motion2 {
            angle,
            shift: [p2[0] - q[0], p2[1] - q[1]],
        }
    }
}

impl<'a, 'b> Run<'a, 'b> {
    /// The keys of the bends of the bent sheet `maker` that `bends` name: faces any bend
    /// region carries (`BendFace` tagged by the bend key), made by the base flange or by an
    /// unfold or fold of it.
    pub(in crate::regen::engine) fn strip_bend_keys(
        sec: &Section,
        bends: &[Reference],
    ) -> Option<Vec<String>> {
        let mut out = Vec::new();
        for r in bends {
            let Reference::Name {
                name: EntityName::Generated(g),
                ..
            } = r
            else {
                return None;
            };
            let crate::refs::name::SourceTag::Key { key } = &g.tag else {
                return None;
            };
            if g.role != Role::BendFace || !sec.bends.iter().any(|b| &b.key == key) {
                return None;
            }
            if !out.contains(key) {
                out.push(key.clone());
            }
        }
        Some(out)
    }

    /// The bend keys of the bent sheet `maker` the unfolds and folds before feature `i` left
    /// flat, and the flat the latest unfold fixed.
    pub(in crate::regen::engine) fn strip_flattened(
        &self,
        i: usize,
        maker: &FeatureId,
        sec: &Section,
    ) -> (Vec<String>, Option<String>) {
        let all: Vec<String> = sec.bends.iter().map(|b| b.key.clone()).collect();
        let mut flat: Vec<String> = Vec::new();
        let mut fixed = None;
        let mut last_ours = false;
        for f in &self.doc.features[..i] {
            let ran = self.st.records.contains_key(&f.id)
                || self
                    .st
                    .bodies
                    .iter()
                    .any(|b| b.names.faces.iter().flatten().any(|n| n.maker() == &f.id));
            if !ran {
                continue;
            }
            match &f.kind {
                FeatureKind::SheetMetal(SheetMetalFeature::Unfold { fixed_face, bends }) => {
                    let key = match fixed_face {
                        Reference::Name {
                            name: EntityName::Generated(g),
                            ..
                        } if g.role == Role::SheetFace => match &g.tag {
                            crate::refs::name::SourceTag::Key { key }
                                if sec.flats.iter().any(|x| &x.curve.0 == key) =>
                            {
                                Some(key.clone())
                            }
                            _ => None,
                        },
                        _ => None,
                    };
                    let on_sheet =
                        matches!(fixed_face, Reference::Name { name, .. } if name.maker() == maker);
                    last_ours = on_sheet && key.is_some();
                    if !last_ours {
                        continue;
                    }
                    fixed = key;
                    let set = if bends.is_empty() {
                        all.clone()
                    } else {
                        Self::strip_bend_keys(sec, bends).unwrap_or_default()
                    };
                    for k in set {
                        if !flat.contains(&k) {
                            flat.push(k);
                        }
                    }
                }
                FeatureKind::SheetMetal(SheetMetalFeature::Fold { bends }) if last_ours => {
                    if bends.is_empty() {
                        flat.clear();
                    } else {
                        let set = Self::strip_bend_keys(sec, bends).unwrap_or_default();
                        flat.retain(|k| !set.contains(k));
                    }
                }
                _ => {}
            }
        }
        (flat, fixed)
    }

    /// A partial unfold or fold of a bent sheet, rebuilt in place: the bends keyed in `add`
    /// are laid flat and those in `remove` bent again, on top of what the unfolds and folds
    /// before feature `i` left. The flat `fixed` stays put (a fold keeps the flat the latest
    /// unfold fixed).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::regen::engine) fn strip_some(
        &mut self,
        i: usize,
        id: &FeatureId,
        body: usize,
        maker: &FeatureId,
        fixed: Option<String>,
        add: &[String],
        remove: &[String],
        naming: &NamingContext,
    ) -> Result<Compiled, Halt> {
        let Some(FeatureRecord::Sheet {
            rules,
            section: Some(sec),
            depth,
        }) = self.st.records.get(maker).cloned()
        else {
            return Err(self.not_lowered(id, "partial unfolds and folds of this sheet"));
        };
        let rebuilders = self.sheet_rebuilders(i, maker);
        self.only_made_by(id, body, &rebuilders, "a partial unfold or fold")?;
        let (mut flat, last_fixed) = self.strip_flattened(i, maker, &sec);
        let Some(fixed) = fixed.or(last_fixed) else {
            return Err(self.error(
                id,
                Some("fixed_face"),
                DiagCode::RefKindMismatch,
                "the fixed face is not a flat of the sheet".into(),
                Vec::new(),
            ));
        };
        for k in add {
            if !flat.contains(k) {
                flat.push(k.clone());
            }
        }
        flat.retain(|k| !remove.contains(k));
        let (op, naming) =
            self.strip_state(id, maker, &rules, &sec, depth, &fixed, &flat, naming)?;
        Ok(self.in_place(
            Compiled::Op {
                op,
                operands: Vec::new(),
                naming,
                import: false,
                instances: None,
                tools: None,
            },
            body,
        ))
    }

    /// The base flange and the unfolds and folds before feature `i`: what may have made the
    /// faces of a sheet a partial unfold or fold rebuilds.
    pub(in crate::regen::engine) fn sheet_rebuilders(
        &self,
        i: usize,
        maker: &FeatureId,
    ) -> Vec<FeatureId> {
        let mut out = vec![maker.clone()];
        for f in &self.doc.features[..i] {
            if matches!(
                f.kind,
                FeatureKind::SheetMetal(SheetMetalFeature::Unfold { .. })
                    | FeatureKind::SheetMetal(SheetMetalFeature::Fold { .. })
            ) {
                out.push(f.id.clone());
            }
        }
        out
    }

    /// A bent sheet with the bends keyed in `flat` laid out flat and the others bent: the flat
    /// `fixed` stays where the base flange put it, and each flattened bend becomes a straight
    /// strip of its allowance. Faces of the flats and of the bends that stay bent answer to the
    /// base flange's names again; the flattened strips are this feature's own.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::regen::engine) fn strip_state(
        &mut self,
        id: &FeatureId,
        maker: &FeatureId,
        rules: &SheetRules,
        sec: &Section,
        depth: f64,
        fixed: &str,
        flat: &[String],
        naming: &NamingContext,
    ) -> Result<(KernelOp, NamingContext), Halt> {
        let dev = crate::model::sheet_metal::develop(sec, &rules.rule)
            .map_err(|m| self.error(id, None, DiagCode::RegenInvalid, m, Vec::new()))?;
        let Some((_, frame)) = self.st.sketches.get(&self.sketch_of(maker)).cloned() else {
            return Err(self.error(
                id,
                None,
                DiagCode::RegenInvalid,
                "the sheet's sketch is gone".into(),
                Vec::new(),
            ));
        };
        let Some(k0) = sec.flats.iter().position(|f| f.curve.0 == fixed) else {
            return Err(self.not_lowered(id, "partial unfolds of this sheet"));
        };
        let allowance = |key: &str| {
            dev.regions.iter().find_map(|r| match r {
                StripRegion::Bend { key: k, length, .. } if k == key => Some(*length),
                _ => None,
            })
        };
        let dir = |f: &crate::model::sheet_metal::SectionFlat| {
            let l = f.length().max(1e-300);
            [(f.end[0] - f.start[0]) / l, (f.end[1] - f.start[1]) / l]
        };
        // Each flat's motion from its bent place, outward from the fixed flat.
        let n = sec.flats.len();
        let mut motion = vec![Motion2::IDENTITY; n];
        for k in k0 + 1..n {
            let b = &sec.bends[k - 1];
            motion[k] = if flat.contains(&b.key) {
                let prev = &sec.flats[k - 1];
                let d = motion[k - 1].turn(dir(prev));
                let e = motion[k - 1].apply(prev.end);
                let ba = allowance(&b.key).unwrap_or(0.0);
                Motion2::carrying(
                    sec.flats[k].start,
                    dir(&sec.flats[k]),
                    [e[0] + ba * d[0], e[1] + ba * d[1]],
                    d,
                )
            } else {
                motion[k - 1]
            };
        }
        for k in (0..k0).rev() {
            let b = &sec.bends[k];
            motion[k] = if flat.contains(&b.key) {
                let next = &sec.flats[k + 1];
                let d = motion[k + 1].turn(dir(next));
                let s = motion[k + 1].apply(next.start);
                let ba = allowance(&b.key).unwrap_or(0.0);
                Motion2::carrying(
                    sec.flats[k].end,
                    dir(&sec.flats[k]),
                    [s[0] - ba * d[0], s[1] - ba * d[1]],
                    d,
                )
            } else {
                motion[k + 1]
            };
        }
        let mut tools = Vec::new();
        for (k, f) in sec.flats.iter().enumerate() {
            let m = motion[k];
            let off =
                |p: [f64; 2], o: f64| m.apply([p[0] + o * f.normal[0], p[1] + o * f.normal[1]]);
            let c = f.curve.0.clone();
            let corners = vec![
                (format!("{}.near", c), off(f.start, f.near)),
                (format!("{}.b", c), off(f.end, f.near)),
                (format!("{}.far", c), off(f.end, f.far)),
                (format!("{}.a", c), off(f.start, f.far)),
            ];
            let mut tool = LabeledTool::new(ToolShape::prism(polygon(frame, &corners), 0.0, depth));
            let sid = |s: &str| crate::ids::SketchEntityId::new(format!("{}.{}", c, s));
            tool.sides.push((
                sid("near"),
                FaceLabel::key(Role::SheetFace, c.clone()).with_ordinal(0),
            ));
            tool.sides.push((
                sid("far"),
                FaceLabel::key(Role::SheetFace, c.clone()).with_ordinal(1),
            ));
            tool.sides.push((
                sid("a"),
                FaceLabel::key(Role::SheetEdge, c.clone()).with_ordinal(if k == 0 { 0 } else { 4 }),
            ));
            tool.sides.push((
                sid("b"),
                FaceLabel::key(Role::SheetEdge, c.clone()).with_ordinal(if k + 1 == n {
                    1
                } else {
                    5
                }),
            ));
            tool.start = Some(FaceLabel::key(Role::SheetEdge, c.clone()).with_ordinal(2));
            tool.end = Some(FaceLabel::key(Role::SheetEdge, c.clone()).with_ordinal(3));
            tools.push(tool);
        }
        let mut kept: BTreeSet<String> = sec.flats.iter().map(|f| f.curve.0.clone()).collect();
        for (j, b) in sec.bends.iter().enumerate() {
            // A bend joins flats j and j + 1; it moves with the flat nearer the fixed one.
            let m = if j < k0 { motion[j + 1] } else { motion[j] };
            if flat.contains(&b.key) {
                let prev = &sec.flats[j];
                let (d, e) = if j < k0 {
                    let next = &sec.flats[j + 1];
                    let d = motion[j + 1].turn(dir(next));
                    let s = motion[j + 1].apply(next.start);
                    let ba = allowance(&b.key).unwrap_or(0.0);
                    (d, [s[0] - ba * d[0], s[1] - ba * d[1]])
                } else {
                    (motion[j].turn(dir(prev)), motion[j].apply(prev.end))
                };
                let nn = [-d[1], d[0]];
                let side = motion[j].turn(prev.normal);
                let sgn = if nn[0] * side[0] + nn[1] * side[1] >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                let ba = allowance(&b.key).unwrap_or(0.0);
                let at = |x: f64, o: f64| {
                    [
                        e[0] + x * d[0] + sgn * o * nn[0],
                        e[1] + x * d[1] + sgn * o * nn[1],
                    ]
                };
                let key = b.key.clone();
                let corners = vec![
                    (format!("{}.near", key), at(0.0, prev.near)),
                    (format!("{}.x1", key), at(ba, prev.near)),
                    (format!("{}.far", key), at(ba, prev.far)),
                    (format!("{}.x0", key), at(0.0, prev.far)),
                ];
                let mut tool =
                    LabeledTool::new(ToolShape::prism(polygon(frame, &corners), 0.0, depth));
                let sid = |s: &str| crate::ids::SketchEntityId::new(format!("{}.{}", key, s));
                for (s, label) in [
                    (
                        "near",
                        FaceLabel::key(Role::BendFace, key.clone()).with_ordinal(0),
                    ),
                    (
                        "far",
                        FaceLabel::key(Role::BendFace, key.clone()).with_ordinal(1),
                    ),
                    (
                        "x0",
                        FaceLabel::key(Role::SheetEdge, key.clone()).with_ordinal(0),
                    ),
                    (
                        "x1",
                        FaceLabel::key(Role::SheetEdge, key.clone()).with_ordinal(1),
                    ),
                ] {
                    tool.sides.push((sid(s), label));
                }
                tool.start = Some(FaceLabel::key(Role::SheetEdge, key.clone()).with_ordinal(2));
                tool.end = Some(FaceLabel::key(Role::SheetEdge, key).with_ordinal(3));
                tools.push(tool);
            } else {
                let mut moved = b.clone();
                moved.center = m.apply(b.center);
                moved.from += m.angle;
                moved.to += m.angle;
                tools.push(super::families::bend_tool(frame, &moved, depth, None));
                kept.insert(b.key.clone());
            }
        }
        let restore = vec![Restore {
            feature: maker.clone(),
            keys: Some(kept),
        }];
        Ok((
            labeled(tools, Vec::new(), Combine::New, false),
            naming.at_step(None, restore),
        ))
    }
}
