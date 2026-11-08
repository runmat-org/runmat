//! Plates with sketched bends and jogs: which bends a plate carries, unfolding some of them,
//! recording each bend, and the plate's flat pattern.

use super::*;

// ------------------------------------------------------------- plates with bends
//
// A sketched bend or jog keeps the plate's developed outline (its line is the bend centre
// line and the strip it consumes is its allowance), so the plate's flat pattern is the base
// flange's own profile with each bend's centre line on it. Each bend records its lines and
// the half of the plate it moved; a later bend must keep its own moved half clear of every
// earlier one, since it rebuilds its part of the plate from the base flange's outline.

impl<'a, 'b> Run<'a, 'b> {
    /// The bend features that made faces on `body`, besides the base flange `maker`, with
    /// their plate-bend records; `Err(feature)` names the first feature of another kind.
    pub(super) fn plate_bends(
        &self,
        body: usize,
        maker: &FeatureId,
    ) -> Result<Vec<(FeatureId, Vec<FlatBendLine>)>, FeatureId> {
        let (bends, flanges, others) = self.plate_bends_and_others(body, maker);
        if let Some(f) = flanges.first() {
            return Err(f.clone());
        }
        match others.first() {
            None => Ok(bends),
            Some(&f) => Err(self.st.bodies[body].names.faces[f]
                .iter()
                .next()
                .map(|n| n.maker().clone())
                .unwrap_or_else(|| maker.clone())),
        }
    }

    /// The bend features that made faces on `body`, besides the base flange `maker`, with
    /// their plate-bend records; the edge flanges and hems on the plate; and the faces of
    /// `body` other features made.
    #[allow(clippy::type_complexity)]
    pub(super) fn plate_bends_and_others(
        &self,
        body: usize,
        maker: &FeatureId,
    ) -> (
        Vec<(FeatureId, Vec<FlatBendLine>)>,
        Vec<FeatureId>,
        Vec<usize>,
    ) {
        let is_bend = |f: &FeatureId| {
            matches!(
                self.st.records.get(f),
                Some(FeatureRecord::PlateBend { base, .. }) if base == maker
            )
        };
        let is_flange = |f: &FeatureId| {
            matches!(
                self.st.records.get(f),
                Some(FeatureRecord::PlateFlange { base, .. }
                    | FeatureRecord::PlateCorner { base, .. }
                    | FeatureRecord::PlateForm { base }) if base == maker
            )
        };
        // Each face answers to the plate or one of its bends (a fold's faces answer to the
        // names it restored as well as its own), or another feature made it.
        let mut bends: Vec<&FeatureId> = Vec::new();
        let mut others = Vec::new();
        for (index, names) in self.st.bodies[body].names.faces.iter().enumerate() {
            let mut accounted = false;
            for n in names {
                if n.maker() == maker || is_flange(n.maker()) {
                    accounted = true;
                } else if is_bend(n.maker()) {
                    accounted = true;
                    if !bends.contains(&n.maker()) {
                        bends.push(n.maker());
                    }
                }
            }
            if !accounted && !names.is_empty() {
                others.push(index);
            }
        }
        let mut out = Vec::new();
        for f in &self.doc.features {
            if let Some(FeatureRecord::PlateBend { lines, .. }) = self.st.records.get(&f.id) {
                if bends.contains(&&f.id) {
                    out.push((f.id.clone(), lines.clone()));
                }
            }
        }
        (out, self.plate_flanges_on(body, maker), others)
    }

    /// The sketched bends and jogs of plate `maker` made before feature `i`, in order.
    pub(in crate::regen::engine) fn bends_of_plate(
        &self,
        i: usize,
        maker: &FeatureId,
    ) -> Vec<FeatureId> {
        self.doc.features[..i]
            .iter()
            .filter(|f| {
                matches!(
                    self.st.records.get(&f.id),
                    Some(FeatureRecord::PlateBend { base, .. }) if base == maker
                ) && self.st.replays.contains_key(&f.id)
            })
            .map(|f| f.id.clone())
            .collect()
    }

    /// Adds to `flat` every bend that bends a flat a bend in `flat` moved: a bend of a moved
    /// flat cannot stay bent once that flat lies on the plate again.
    pub(in crate::regen::engine) fn close_flat(&self, flat: &mut Vec<FeatureId>) {
        loop {
            let more: Vec<FeatureId> = self
                .st
                .records
                .iter()
                .filter_map(|(f, r)| match r {
                    FeatureRecord::PlateBend {
                        parent: Some(p), ..
                    } if flat.contains(p) && !flat.contains(f) => Some(f.clone()),
                    _ => None,
                })
                .collect();
            if more.is_empty() {
                return;
            }
            flat.extend(more);
        }
    }

    /// `bends` with the bends whose moved flats they bend, recursively: folding a bend of a
    /// moved flat folds that flat's bend too.
    pub(in crate::regen::engine) fn with_parents(&self, bends: Vec<FeatureId>) -> Vec<FeatureId> {
        let mut out = bends;
        let mut k = 0;
        while k < out.len() {
            if let Some(FeatureRecord::PlateBend {
                parent: Some(p), ..
            }) = self.st.records.get(&out[k])
            {
                if !out.contains(p) {
                    out.push(p.clone());
                }
            }
            k += 1;
        }
        out
    }

    /// The bends of plate `maker` that the unfolds and folds before feature `i` left flat.
    pub(in crate::regen::engine) fn flattened_bends(
        &self,
        i: usize,
        maker: &FeatureId,
    ) -> Vec<FeatureId> {
        let all = self.bends_of_plate(i, maker);
        let mut flat: Vec<FeatureId> = Vec::new();
        let ours = |name: &EntityName| {
            let mut m = vec![name.maker().clone()];
            if let EntityName::Generated(g) = name {
                if let Some(step) = &g.step {
                    m.push(FeatureId::new(step));
                }
            }
            m.iter().any(|f| {
                f == maker
                    || matches!(
                        self.st.records.get(f),
                        Some(FeatureRecord::PlateBend { base, .. }) if base == maker
                    )
            })
        };
        // Whether the latest unfold was of this plate: a fold folds that one.
        let mut last_ours = false;
        for f in &self.doc.features[..i] {
            // Only unfolds and folds that ran change the state.
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
                FeatureKind::SheetMetal(SheetMetalFeature::Unfold { bends, fixed_face }) => {
                    last_ours = matches!(fixed_face, Reference::Name { name, .. } if ours(name));
                    if !last_ours {
                        continue;
                    }
                    let set = if bends.is_empty() {
                        all.clone()
                    } else {
                        Self::bend_ids_by_name(&self.st.records, maker, bends)
                    };
                    for b in set {
                        if !flat.contains(&b) {
                            flat.push(b);
                        }
                    }
                    self.close_flat(&mut flat);
                }
                FeatureKind::SheetMetal(SheetMetalFeature::Fold { bends }) => {
                    if !last_ours {
                        continue;
                    }
                    if bends.is_empty() {
                        flat.clear();
                    } else {
                        let set = self.with_parents(Self::bend_ids_by_name(
                            &self.st.records,
                            maker,
                            bends,
                        ));
                        flat.retain(|b| !set.contains(b));
                    }
                }
                _ => {}
            }
        }
        flat
    }

    /// The sketched bends and jogs of `maker` that `bends` name: a face a bend made (or a
    /// step of an unfold or fold made for that bend), by the name written in the reference.
    pub(super) fn bend_ids_by_name(
        records: &BTreeMap<FeatureId, FeatureRecord>,
        maker: &FeatureId,
        bends: &[Reference],
    ) -> Vec<FeatureId> {
        let mut out = Vec::new();
        for r in bends {
            let Reference::Name { name, .. } = r else {
                continue;
            };
            let mut candidates = vec![name.maker().clone()];
            if let EntityName::Generated(g) = name {
                if let Some(step) = &g.step {
                    candidates.push(FeatureId::new(step));
                }
            }
            for c in candidates {
                if matches!(records.get(&c), Some(FeatureRecord::PlateBend { base, .. }) if base == maker)
                    && !out.contains(&c)
                {
                    out.push(c);
                }
            }
        }
        out
    }

    /// The bends `bends` name, each a sketched bend or jog of plate `maker`.
    pub(in crate::regen::engine) fn plate_bend_ids(
        &mut self,
        id: &FeatureId,
        maker: &FeatureId,
        bends: &[Reference],
    ) -> Result<Vec<FeatureId>, Halt> {
        let out = Self::bend_ids_by_name(&self.st.records, maker, bends);
        if out.len() != bends.len() {
            return Err(self.error(
                id,
                Some("bends"),
                DiagCode::RefKindMismatch,
                "each entry of `bends` names a face of one of the plate's sketched bends or jogs"
                    .into(),
                Vec::new(),
            ));
        }
        Ok(out)
    }

    /// Steps replaying the bends of plate `maker` that are not in `flat`, in order, on `body`.
    /// Each step is keyed by its bend and restores that bend's names.
    pub(in crate::regen::engine) fn replay_bends(
        &self,
        maker: &FeatureId,
        flat: &[FeatureId],
        body: usize,
        naming: &NamingContext,
    ) -> Vec<Step> {
        self.bends_of_plate(self.doc.features.len(), maker)
            .into_iter()
            .filter(|b| !flat.contains(b))
            .filter_map(|b| self.replay_step(&b, body, naming))
            .collect()
    }

    /// The plate's edge flanges, hems, corner features and forms that a fold replays, in
    /// feature order.
    pub(in crate::regen::engine) fn plate_features_replayed(&self, maker: &FeatureId) -> Vec<FeatureId> {
        self.doc
            .features
            .iter()
            .filter(|f| {
                self.st.replays.contains_key(&f.id)
                    && matches!(
                        self.st.records.get(&f.id),
                        Some(FeatureRecord::PlateFlange { base, .. }
                            | FeatureRecord::PlateCorner { base, .. }
                            | FeatureRecord::PlateForm { base }) if base == maker
                    )
            })
            .map(|f| f.id.clone())
            .collect()
    }

    /// The bends left bent and every flange, corner feature and form of plate `maker`,
    /// replayed in feature order, so a flange on a flat a bend moved follows that bend.
    pub(in crate::regen::engine) fn replay_plate(
        &self,
        maker: &FeatureId,
        flat: &[FeatureId],
        body: usize,
        naming: &NamingContext,
    ) -> Vec<Step> {
        let bends = self.bends_of_plate(self.doc.features.len(), maker);
        let others = self.plate_features_replayed(maker);
        self.doc
            .features
            .iter()
            .filter(|f| (bends.contains(&f.id) && !flat.contains(&f.id)) || others.contains(&f.id))
            .filter_map(|f| self.replay_step(&f.id, body, naming))
            .collect()
    }

    fn replay_step(&self, b: &FeatureId, body: usize, naming: &NamingContext) -> Option<Step> {
        Some(Step {
            op: self.st.replays.get(b)?.clone(),
            operands: vec![body],
            naming: naming.at_step(Some(b.0.clone()), vec![Restore::all(b)]),
        })
    }

    /// Unfolds the named bends of a plate and keeps the rest: the plate rebuilt flat in place,
    /// then the bends that stay bent replayed. A strip's partial unfold is not lowered.
    pub(in crate::regen::engine) fn unfold_some(
        &mut self,
        i: usize,
        id: &FeatureId,
        fixed: &EntityHandle,
        bends: &[Reference],
        naming: &NamingContext,
    ) -> Result<Compiled, Halt> {
        let body = fixed.body as usize;
        if let Some((maker, _, Some(sec), _)) = self.sheet_of(body) {
            let fixed_key = self.st.entity_names(fixed).iter().find_map(|n| match n {
                EntityName::Generated(g) if g.feature == maker && g.role == Role::SheetFace => {
                    match &g.tag {
                        crate::refs::name::SourceTag::Key { key } => Some(key.clone()),
                        _ => None,
                    }
                }
                _ => None,
            });
            let Some(fixed_key) = fixed_key else {
                return Err(self.error(
                    id,
                    Some("fixed_face"),
                    DiagCode::RefKindMismatch,
                    "the fixed face is not a flat of the sheet".into(),
                    Vec::new(),
                ));
            };
            let Some(keys) = Self::strip_bend_keys(&sec, bends) else {
                return Err(self.error(
                    id,
                    Some("bends"),
                    DiagCode::RefKindMismatch,
                    "each entry of `bends` names a bend face of the sheet".into(),
                    Vec::new(),
                ));
            };
            return self.strip_some(i, id, body, &maker, Some(fixed_key), &keys, &[], naming);
        }
        let Some((maker, rules, None, _)) = self.sheet_of(body).or_else(|| {
            // A folded plate's body may answer to the fold instead of the base flange.
            let base =
                self.st.bodies[body]
                    .names
                    .faces
                    .iter()
                    .flatten()
                    .find_map(|n| match self.st.records.get(n.maker()) {
                        Some(FeatureRecord::Sheet { .. }) => Some(n.maker().clone()),
                        _ => None,
                    })?;
            match self.st.records.get(&base) {
                Some(FeatureRecord::Sheet {
                    rules,
                    section,
                    depth,
                    ..
                }) => Some((base, rules.clone(), section.clone(), *depth)),
                _ => None,
            }
        }) else {
            return Err(self.not_lowered(
                id,
                "unfolding some bends of a sheet made by a bent base flange (unfold all of them: leave `bends` empty)",
            ));
        };
        let chosen = self.plate_bend_ids(id, &maker, bends)?;
        let mut flat = self.flattened_bends(i, &maker);
        for b in chosen {
            if !flat.contains(&b) {
                flat.push(b);
            }
        }
        self.close_flat(&mut flat);
        // The fixed face stays put: it lies on the plate's unmoved part or on a flat a bend
        // that stays bent moved.
        let names = self.st.entity_names(fixed);
        let stays = names.iter().any(|n| {
            n.maker() == &maker
                || matches!(
                    self.st.records.get(n.maker()),
                    Some(FeatureRecord::PlateBend { base, .. }) if base == &maker
                ) && !flat.contains(n.maker())
        });
        if !stays {
            return Err(self.not_lowered(
                id,
                "partial unfolds fixed on a face that moves (fix the unfold on the plate or on a flat of a bend that stays bent)",
            ));
        }
        if self.plate_bends(body, &maker).is_err() {
            return Err(self.not_lowered(
                id,
                "unfolding a plate that features besides its bends changed",
            ));
        }
        let Compiled::Op { op: develop, .. } =
            self.plate_flat(i, id, fixed, &maker, &rules, naming, false)?
        else {
            return Err(self.not_lowered(id, "unfolding this plate"));
        };
        // The record lists only the bends this unfold left flat.
        if let Some(FeatureRecord::FlatPattern { layout, .. }) = self.st.records.get_mut(id) {
            layout.bend_lines.retain(|l| flat.contains(&l.feature));
        }
        let Compiled::Op {
            op: develop,
            operands,
            ..
        } = self.in_place(
            Compiled::Op {
                op: develop,
                operands: Vec::new(),
                naming: naming.clone(),
                import: false,
                instances: None,
                tools: None,
            },
            body,
        )
        else {
            return Err(self.not_lowered(id, "unfolding this plate"));
        };
        let mut steps = vec![Step {
            op: develop,
            operands,
            naming: naming.clone(),
        }];
        steps.extend(self.replay_bends(&maker, &flat, body, naming));
        Ok(Compiled::Steps(steps))
    }

    /// Records a sketched bend or jog of the plate `maker` on `body`: refuses one whose moved
    /// half of the plate overlaps a half an earlier bend moved.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::regen::engine) fn record_plate_bend(
        &mut self,
        id: &FeatureId,
        body: usize,
        maker: &FeatureId,
        plate: &CanonicalProfile,
        lines: Vec<FlatBendLine>,
        moved: HalfPlane,
        motion: Transform,
        flat: CanonicalProfile,
        parent: Option<FeatureId>,
    ) -> Result<(), Halt> {
        // A bend of a moved flat moves part of what its parent (and the parent's parents)
        // moved: only the other bends must keep clear.
        let mut ancestors: Vec<FeatureId> = Vec::new();
        let mut up = parent.clone();
        while let Some(p) = up {
            up = match self.st.records.get(&p) {
                Some(FeatureRecord::PlateBend { parent, .. }) => parent.clone(),
                _ => None,
            };
            ancestors.push(p);
        }
        let mut earlier: Vec<(FeatureId, HalfPlane)> = Vec::new();
        for n in self.st.bodies[body].names.faces.iter().flatten() {
            if let Some(FeatureRecord::PlateBend { base, moved, .. }) =
                self.st.records.get(n.maker())
            {
                if base == maker
                    && !ancestors.contains(n.maker())
                    && !earlier.iter().any(|(f, _)| f == n.maker())
                {
                    earlier.push((n.maker().clone(), *moved));
                }
            }
        }
        let outline: Vec<[f64; 2]> = plate
            .loops
            .first()
            .map(|l| l.iter().map(|s| s.start).collect())
            .unwrap_or_default();
        for (f, other) in earlier {
            if half_planes_overlap_in(&outline, moved, other) {
                return Err(self.not_lowered(
                    id,
                    &format!(
                        "a bend of plate material `{f}` already moved (each sketched bend or jog moves a part of the plate no other bend moved)"
                    ),
                ));
            }
        }
        self.st.records.insert(
            id.clone(),
            FeatureRecord::PlateBend {
                base: maker.clone(),
                lines,
                moved,
                motion,
                flat,
                parent,
            },
        );
        Ok(())
    }

    /// The flat pattern of a plate with sketched bends and jogs: the base flange's profile,
    /// in place, with the bends' centre lines.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::regen::engine) fn plate_flat(
        &mut self,
        i: usize,
        id: &FeatureId,
        fixed: &EntityHandle,
        maker: &FeatureId,
        rules: &SheetRules,
        naming: &NamingContext,
        follow_fixed: bool,
    ) -> Result<Compiled, Halt> {
        let body = fixed.body as usize;
        let names = self.st.entity_names(fixed);
        // Fixed on the plate's unmoved part, the flat lies in place; fixed on a flat a bend
        // moved, the developed plate moves with that bend so the fixed face stays put.
        let motion = if !follow_fixed || names.iter().any(|n| n.maker() == maker) {
            None
        } else {
            let moved_by = names
                .iter()
                .find_map(|n| match self.st.records.get(n.maker()) {
                    Some(FeatureRecord::PlateBend { base, motion, .. }) if base == maker => {
                        Some(*motion)
                    }
                    _ => None,
                });
            match moved_by {
                    Some(m) => Some(m),
                    None => return Err(self.not_lowered(
                        id,
                        "flat patterns fixed on a bend face or a flange (fix the flat on the plate or on a flat a sketched bend moved)",
                    )),
                }
        };
        // Cuts and holes other features made in the flats develop onto the flat pattern.
        let (bends, flanges, others) = self.plate_bends_and_others(body, maker);
        let Some(sketched) = self.plate_profile(i, id, "fixed_face", maker)? else {
            return Err(self.not_lowered(id, "flat patterns of this sheet"));
        };
        // Edge flanges develop as strips spliced into the outline along their edges.
        let plate = self.developed_plate(id, maker, &sketched, &flanges)?;
        let t = rules.thickness;
        let mut tool = LabeledTool::new(ToolShape::prism(plate.clone(), 0.0, t));
        for seg in plate.loops.iter().flatten() {
            tool.sides.push((
                seg.curve.clone(),
                FaceLabel::key(Role::SheetEdge, seg.curve.0.clone()),
            ));
        }
        tool.start = Some(FaceLabel::key(Role::SheetFace, "plate").with_ordinal(0));
        tool.end = Some(FaceLabel::key(Role::SheetFace, "plate").with_ordinal(1));
        let layout = FlatLayout {
            frame: plate.frame,
            depth: t,
            loops: plate.loops.clone(),
            bend_lines: bends
                .iter()
                .flat_map(|(_, l)| l.clone())
                .chain(self.flange_bend_lines(&flanges))
                .collect(),
        };
        self.st.records.insert(
            id.clone(),
            FeatureRecord::FlatPattern {
                layout,
                development: None,
                in_place: false,
            },
        );
        let develop = labeled(vec![tool], Vec::new(), Combine::New, false);
        let (cuts, at) = if others.is_empty() {
            (Vec::new(), self.st.bodies.len())
        } else {
            self.plate_cut_steps(id, body, &bends, &flanges, &plate, t, &others, naming)?
        };
        if motion.is_none() && cuts.is_empty() {
            return Ok(op(develop, Vec::new(), naming.clone()));
        }
        let mut steps = vec![Step {
            op: develop,
            operands: Vec::new(),
            naming: naming.clone(),
        }];
        steps.extend(cuts);
        let Some(motion) = motion else {
            return Ok(Compiled::Steps(steps));
        };
        // Then the move that keeps the fixed flat in place.
        if let Some(FeatureRecord::FlatPattern { layout, .. }) = self.st.records.get_mut(id) {
            layout.frame = Frame {
                origin: motion.apply(layout.frame.origin),
                x: rotate_by(&motion, layout.frame.x),
                y: rotate_by(&motion, layout.frame.y),
                z: rotate_by(&motion, layout.frame.z),
            };
        }
        steps.push(Step {
            op: KernelOp::Transform { transform: motion },
            operands: vec![at],
            naming: naming.at_step(Some("place".into()), Vec::new()),
        });
        Ok(Compiled::Steps(steps))
    }
}
