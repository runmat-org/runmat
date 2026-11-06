//! Patterned and mirrored instances, envelopes, and the planes and transforms patterns use.

use super::*;

/// What a mirror reflects for one entry of its seed list: an authored occurrence, or each seed
/// and instance of a pattern (`(seed, key, the pattern's transform)`, the seed itself keyed
/// `mirror`).
struct MirrorImage {
    seed: String,
    key: InstanceKey,
    left: Transform,
    host: Option<NodeId>,
}

impl<'a, 'b> Resolver<'a, 'b> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compose_instances(
        &mut self,
        node: NodeId,
        doc: &Arc<AssemblyDocument>,
        env: &ValueEnv,
        pending: &[Pending],
        stack: &[FilePath],
        flexible_member: bool,
        present: &mut BTreeSet<PathSegment>,
    ) -> Vec<InstancePlan> {
        let file = doc.path.clone();
        let mut plans = Vec::new();
        // Each active pattern's instances, for the mirrors that repeat it.
        let mut laid: BTreeMap<String, (Vec<(InstanceKey, Transform)>, Option<NodeId>)> =
            BTreeMap::new();
        for p in &doc.patterns {
            let site = Site::with_param(
                Subject::Pattern {
                    name: p.name.clone(),
                },
                file.as_str(),
            );
            if !self.holds(p.when.as_ref(), &file, &site) {
                continue;
            }
            let (transforms, host) = match self.pattern_transforms(node, &p.layout, env) {
                Ok(t) => t,
                Err(msg) => {
                    self.diag(
                        DiagCode::AsmBadPattern,
                        Severity::Error,
                        site,
                        format!("pattern `{}`: {}", p.name, msg),
                        Vec::new(),
                    );
                    continue;
                }
            };
            let kept: Vec<(InstanceKey, Transform)> = transforms
                .into_iter()
                .filter(|(k, _)| !p.skip.contains(k))
                .collect();
            for seed_name in &p.seeds {
                let Some(seed) = self.child(node, &PathSegment::named(seed_name)) else {
                    self.diag(
                        DiagCode::AsmUnknownOccurrence,
                        Severity::Error,
                        site.clone(),
                        format!(
                            "pattern `{}` repeats `{}`, which is not active here",
                            p.name, seed_name
                        ),
                        Vec::new(),
                    );
                    continue;
                };
                let Some(occ) = self.active_occurrence(doc, seed_name) else {
                    continue;
                };
                for (key, t) in &kept {
                    let seg = PathSegment::instance(seed_name, &p.name, key.clone());
                    present.insert(seg.clone());
                    if let Some(inst) = self.compose_occurrence(
                        node,
                        doc,
                        env,
                        &occ,
                        seg,
                        pending,
                        NodeOrigin::Pattern {
                            pattern: p.name.clone(),
                        },
                        stack,
                        flexible_member,
                    ) {
                        plans.push(InstancePlan {
                            owner: node,
                            instance: inst,
                            seed,
                            outer: Transform::IDENTITY,
                            left: *t,
                            right: Transform::IDENTITY,
                            host,
                        });
                    }
                }
            }
            laid.insert(p.name.clone(), (kept, host));
        }
        for m in &doc.mirrors {
            let site = Site::with_param(
                Subject::Pattern {
                    name: m.name.clone(),
                },
                file.as_str(),
            );
            if !self.holds(m.when.as_ref(), &file, &site) {
                continue;
            }
            let world = match self.plane(node, &m.plane, env) {
                Ok((pt, n)) => Transform::reflection(pt, n),
                Err(msg) => {
                    self.diag(
                        DiagCode::AsmBadPattern,
                        Severity::Error,
                        site,
                        format!("mirror `{}`: {}", m.name, msg),
                        Vec::new(),
                    );
                    continue;
                }
            };
            let local = {
                let (pt, n) = match &m.hand {
                    MirrorHand::Symmetric { point, normal }
                    | MirrorHand::Opposite { point, normal, .. } => (point, normal),
                };
                match self.vec3(pt, env) {
                    Ok(p) => Transform::reflection(p, normalize(*n)),
                    Err(msg) => {
                        self.diag(
                            DiagCode::AsmBadPattern,
                            Severity::Error,
                            site.clone(),
                            msg,
                            Vec::new(),
                        );
                        continue;
                    }
                }
            };
            for image in self.mirror_images(doc, m, &laid, &site) {
                let Some(seed) = self.child(node, &PathSegment::named(&image.seed)) else {
                    self.diag(
                        DiagCode::AsmUnknownOccurrence,
                        Severity::Error,
                        site.clone(),
                        format!(
                            "mirror `{}` reflects `{}`, which is not active here",
                            m.name, image.seed
                        ),
                        Vec::new(),
                    );
                    continue;
                };
                let Some(mut occ) = self.active_occurrence(doc, &image.seed) else {
                    continue;
                };
                if let MirrorHand::Opposite {
                    source,
                    configuration,
                    ..
                } = &m.hand
                {
                    occ.source = source.clone();
                    occ.configuration = configuration.clone();
                }
                let seg = PathSegment::instance(&image.seed, &m.name, image.key);
                present.insert(seg.clone());
                if let Some(inst) = self.compose_occurrence(
                    node,
                    doc,
                    env,
                    &occ,
                    seg,
                    pending,
                    NodeOrigin::Mirror {
                        mirror: m.name.clone(),
                    },
                    stack,
                    flexible_member,
                ) {
                    plans.push(InstancePlan {
                        owner: node,
                        instance: inst,
                        seed,
                        outer: world,
                        left: image.left,
                        right: local,
                        host: image.host,
                    });
                }
            }
        }
        plans
    }

    /// The occurrences a mirror reflects. A seed that names a pattern of this assembly brings
    /// the pattern's seeds and every instance it lays out here, so a mirrored pattern needs no
    /// second pattern. An image named twice (a seed listed directly and through a pattern, or
    /// two patterns sharing a seed) is an error.
    fn mirror_images(
        &mut self,
        doc: &AssemblyDocument,
        m: &OccurrenceMirror,
        laid: &BTreeMap<String, (Vec<(InstanceKey, Transform)>, Option<NodeId>)>,
        site: &Site,
    ) -> Vec<MirrorImage> {
        let mut out: Vec<MirrorImage> = Vec::new();
        let mut seen: BTreeMap<(String, InstanceKey), String> = BTreeMap::new();
        let mut push = |this: &mut Self, via: &str, image: MirrorImage| {
            let key = (image.seed.clone(), image.key.clone());
            if let Some(first) = seen.get(&key) {
                this.diag(
                    DiagCode::AsmBadPattern,
                    Severity::Error,
                    site.clone(),
                    format!(
                        "mirror `{}` reflects `{}[{}]` twice, through {} and {}",
                        m.name, image.seed, image.key, first, via
                    ),
                    Vec::new(),
                );
                return;
            }
            seen.insert(key, via.to_string());
            out.push(image);
        };
        for name in &m.seeds {
            let Some(p) = doc.patterns.iter().find(|p| &p.name == name) else {
                let image = MirrorImage {
                    seed: name.clone(),
                    key: InstanceKey::Mirror,
                    left: Transform::IDENTITY,
                    host: None,
                };
                push(self, &format!("`{}`", name), image);
                continue;
            };
            let via = format!("pattern `{}`", p.name);
            for seed in &p.seeds {
                let image = MirrorImage {
                    seed: seed.clone(),
                    key: InstanceKey::Mirror,
                    left: Transform::IDENTITY,
                    host: None,
                };
                push(self, &via, image);
                // An inactive or failed pattern has no instances to reflect; its seeds still do.
                if let Some((kept, host)) = laid.get(&p.name) {
                    for (key, t) in kept {
                        let image = MirrorImage {
                            seed: seed.clone(),
                            key: key.clone(),
                            left: *t,
                            host: *host,
                        };
                        push(self, &via, image);
                    }
                }
            }
        }
        out
    }

    pub(super) fn declare_envelopes(
        &mut self,
        node: NodeId,
        doc: &AssemblyDocument,
        env: &ValueEnv,
    ) {
        for e in &doc.envelopes {
            let site = Site::with_param(
                Subject::Document,
                &format!("{}#envelope.{}", doc.path, e.name),
            );
            let (min, max) = match (self.vec3(&e.min, env), self.vec3(&e.max, env)) {
                (Ok(a), Ok(b)) => (a, b),
                (Err(m), _) | (_, Err(m)) => {
                    self.diag(
                        DiagCode::ValueFailed,
                        Severity::Error,
                        site,
                        format!("envelope `{}`: {}", e.name, m),
                        Vec::new(),
                    );
                    continue;
                }
            };
            let occupant = match &e.owner {
                None => None,
                Some(o) => {
                    let path = RelPath(o.split('/').map(PathSegment::named).collect());
                    match self.walk(node, &path) {
                        Ok(n) => Some(n),
                        Err(_) => {
                            self.diag(
                                DiagCode::AsmUnknownOccurrence,
                                Severity::Warning,
                                site,
                                format!(
                                    "envelope `{}` is owned by `{}`, which is not active here",
                                    e.name, o
                                ),
                                Vec::new(),
                            );
                            None
                        }
                    }
                }
            };
            let local = crate::assembly::source::Aabb::new(min, max);
            self.envelopes.push(ResolvedEnvelope {
                declared_by: node,
                name: e.name.clone(),
                occupant,
                keep_out: e.keep_out,
                local,
                world: local,
            });
        }
    }

    pub(super) fn active_occurrence(
        &mut self,
        doc: &AssemblyDocument,
        name: &str,
    ) -> Option<Occurrence> {
        let file = doc.path.clone();
        let alts: Vec<Occurrence> = doc.alternatives(name).cloned().collect();
        alts.into_iter().find(|o| {
            let site = Site::with_param(
                Subject::Occurrence {
                    path: name.to_owned(),
                },
                file.as_str(),
            );
            self.holds(o.when.as_ref(), &file, &site)
        })
    }

    pub(super) fn plane(
        &mut self,
        node: NodeId,
        p: &PlaneSpec,
        env: &ValueEnv,
    ) -> Result<([f64; 3], [f64; 3]), String> {
        match p {
            PlaneSpec::Explicit { point, normal } => {
                Ok((self.vec3(point, env)?, normalize(*normal)))
            }
            PlaneSpec::Frame { frame } => {
                let t = self.frame_ref(node, frame, env)?;
                let f = t.to_frame();
                Ok((f.origin, f.z))
            }
        }
    }

    /// Instance transforms in this assembly's coordinates, or, for a followed pattern, in the
    /// host occurrence's definition coordinates (conjugated by the host's placement when the
    /// plan is applied, after placement and the solve).
    #[allow(clippy::type_complexity)]
    pub(super) fn pattern_transforms(
        &mut self,
        node: NodeId,
        layout: &OccurrenceLayout,
        env: &ValueEnv,
    ) -> Result<(Vec<(InstanceKey, Transform)>, Option<NodeId>), String> {
        let mut out = Vec::new();
        match layout {
            OccurrenceLayout::Linear {
                direction,
                spacing,
                count,
            } => {
                let d = normalize(*direction);
                let s = self.length(spacing, env)?;
                let n = self.count(count, env)?;
                for k in 1..n {
                    let t = k as f64 * s;
                    out.push((
                        InstanceKey::Linear { step: k },
                        Transform::translation([d[0] * t, d[1] * t, d[2] * t]),
                    ));
                }
            }
            OccurrenceLayout::Grid {
                first,
                first_spacing,
                first_count,
                second,
                second_spacing,
                second_count,
            } => {
                let (d1, d2) = (normalize(*first), normalize(*second));
                let (s1, s2) = (
                    self.length(first_spacing, env)?,
                    self.length(second_spacing, env)?,
                );
                let (n1, n2) = (
                    self.count(first_count, env)?,
                    self.count(second_count, env)?,
                );
                for i in 0..n1 {
                    for j in 0..n2 {
                        if i == 0 && j == 0 {
                            continue;
                        }
                        let (a, b) = (i as f64 * s1, j as f64 * s2);
                        out.push((
                            InstanceKey::Grid {
                                first: i,
                                second: j,
                            },
                            Transform::translation([
                                d1[0] * a + d2[0] * b,
                                d1[1] * a + d2[1] * b,
                                d1[2] * a + d2[2] * b,
                            ]),
                        ));
                    }
                }
            }
            OccurrenceLayout::Circular {
                axis_point,
                axis,
                count,
                angle,
            } => {
                let p = self.vec3(axis_point, env)?;
                let n = self.count(count, env)?;
                let a = self.angle(angle, env)?;
                let full = (a.abs() - std::f64::consts::TAU).abs() < 1e-9;
                let step = if full || n == 1 {
                    a / n as f64
                } else {
                    a / (n - 1) as f64
                };
                for k in 1..n {
                    out.push((
                        InstanceKey::Circular { step: k },
                        Transform::rotation_about(p, *axis, k as f64 * step),
                    ));
                }
            }
            OccurrenceLayout::Table { rows } => {
                let mut seen = BTreeSet::new();
                for r in rows {
                    if !seen.insert(r.key.clone()) {
                        return Err(format!("row key `{}` appears twice", r.key));
                    }
                    out.push((
                        InstanceKey::Row { key: r.key.clone() },
                        self.offset(&r.offset, env)?,
                    ));
                }
            }
            OccurrenceLayout::FollowPattern {
                occurrence,
                publish,
            } => {
                let host = self.walk(node, occurrence).map_err(|i| {
                    format!("no occurrence `{}`", RelPath(occurrence.0[..=i].to_vec()))
                })?;
                let n = self.nodes[host as usize].clone();
                let s = self
                    .configured_summary(&n.definition, n.revision.as_ref(), &n.configuration)
                    .map_err(|e| e.to_string())?;
                let Some(PublishedGeometry::Pattern { frames }) = s.publishes.get(publish) else {
                    return Err(format!(
                        "`{}` publishes no pattern `{}`",
                        n.definition, publish
                    ));
                };
                let mut frames = frames.clone();
                frames.sort_by(|a, b| a.0.cmp(&b.0));
                let Some((_, f0)) = frames.first() else {
                    return Ok((out, None));
                };
                // The host's frame is not final until placement; the transform is applied in
                // host coordinates, so it is conjugated when the plan is applied. Here the host
                // chain is taken as placed (patterns follow after placement and the solve).
                let f0 = Transform::from_frame(f0);
                for (k, f) in frames.iter().skip(1) {
                    out.push((k.clone(), Transform::from_frame(f).compose(&f0.inverse())));
                }
                return Ok((out, Some(host)));
            }
        }
        Ok((out, None))
    }
}
