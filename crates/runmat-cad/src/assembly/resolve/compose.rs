//! Composing an assembly and each of its occurrences into the tree.

use super::*;

impl<'a, 'b> Resolver<'a, 'b> {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::assembly) fn compose_assembly(
        &mut self,
        node: NodeId,
        doc: Arc<AssemblyDocument>,
        env: ValueEnv,
        pending: Vec<Pending>,
        stack: Vec<FilePath>,
        flexible_member: bool,
        suppress: &BTreeSet<String>,
    ) {
        self.stats.assemblies += 1;
        let file = doc.path.clone();
        let depth = self.nodes[node as usize].depth;
        let names = doc.child_names();
        if names.len() > 500 {
            self.diag(
                DiagCode::AsmTooManyChildren,
                Severity::Warning,
                Site::with_param(Subject::Document, file.as_str()),
                format!(
                    "`{}` has {} direct children; split it by zone or partition (about 500 at most)",
                    file,
                    names.len()
                ),
                Vec::new(),
            );
        }
        self.asm.insert(
            node,
            AsmInfo {
                doc: doc.clone(),
                env: env.clone(),
            },
        );

        // Segments that exist in this configuration, suppressed or not (override targets).
        let mut present: BTreeSet<PathSegment> = BTreeSet::new();
        // 1–3: authored children.
        for name in names {
            let mut active = Vec::new();
            for (i, o) in doc.occurrences.iter().enumerate() {
                if o.name != name {
                    continue;
                }
                let site = Site::with_param(
                    Subject::Occurrence {
                        path: name.to_owned(),
                    },
                    file.as_str(),
                );
                if self.holds(o.when.as_ref(), &file, &site) {
                    active.push(i);
                }
            }
            if active.len() > 1 {
                let witness = self
                    .point
                    .as_ref()
                    .map(|p| self.space.describe_point(p))
                    .unwrap_or_default();
                self.diag(
                    DiagCode::AsmDuplicateName,
                    Severity::Error,
                    Site::with_param(
                        Subject::Occurrence {
                            path: name.to_owned(),
                        },
                        file.as_str(),
                    ),
                    format!(
                        "{} alternatives of `{}` in `{}` are active at once; the first is used",
                        active.len(),
                        name,
                        file
                    ),
                    vec![Repair::ExcludeConfiguration {
                        assembly: file.clone(),
                        occurrence: name.to_owned(),
                        witness,
                    }],
                );
            }
            let Some(&first) = active.first() else {
                if doc.alternatives(name).any(|o| o.required) {
                    self.diag(
                        DiagCode::AsmIncompleteVariant,
                        Severity::Error,
                        Site::with_param(
                            Subject::Occurrence {
                                path: name.to_owned(),
                            },
                            file.as_str(),
                        ),
                        format!(
                            "no alternative of the required occurrence `{}` is active",
                            name
                        ),
                        Vec::new(),
                    );
                }
                continue;
            };
            let o = &doc.occurrences[first];
            present.insert(PathSegment::named(&o.name));
            if o.suppressed || suppress.contains(&o.name) {
                self.stats.suppressed += 1;
                let p = self.display_path(node);
                self.suppressed.push(if p.is_empty() {
                    o.name.clone()
                } else {
                    format!("{}/{}", p, o.name)
                });
                continue;
            }
            self.compose_occurrence(
                node,
                &doc,
                &env,
                o,
                PathSegment::named(&o.name),
                &pending,
                NodeOrigin::Authored,
                &stack,
                flexible_member,
            );
        }

        // 6 (structure): pattern and mirror instances.
        let plans = self.compose_instances(
            node,
            &doc,
            &env,
            &pending,
            &stack,
            flexible_member,
            &mut present,
        );

        // Pending overrides whose first segment names no child here.
        for p in &pending {
            let Some(first) = p.remaining.first() else {
                continue;
            };
            if self.child(node, first).is_none() && !present.contains(first) {
                let candidates: Vec<String> = self.nodes[node as usize]
                    .children
                    .iter()
                    .map(|c| self.nodes[*c as usize].segment.to_string())
                    .filter(|s| s.contains(&first.name) || first.name.contains(s.as_str()))
                    .collect();
                let path = self.display_path(node);
                self.diag(
                    DiagCode::AsmOverrideTargetMissing,
                    Severity::Error,
                    Site::with_param(Subject::Override { index: p.index }, p.file.as_str()),
                    format!(
                        "override {} in `{}` ({}) targets `{}`, which `{}` does not contain in this configuration",
                        p.index,
                        p.file,
                        p.kind.slot(),
                        RelPath(p.remaining.clone()),
                        if path.is_empty() { doc.name.clone() } else { path }
                    ),
                    if candidates.is_empty() {
                        vec![Repair::RemoveOverride {
                            assembly: p.file.clone(),
                            index: p.index,
                        }]
                    } else {
                        vec![
                            Repair::RetargetOverride {
                                assembly: p.file.clone(),
                                index: p.index,
                                candidates,
                            },
                            Repair::RemoveOverride {
                                assembly: p.file.clone(),
                                index: p.index,
                            },
                        ]
                    },
                );
            }
        }
        let _ = depth;

        // 4: placements of authored children.
        self.place_children(node, &doc, &env);
        // 5: mates (a flexible member's mates, instances and fasteners join its parent's solve).
        if !self.req.solve_mates {
            self.apply_instances(&plans);
        } else if flexible_member {
            self.deferred.insert(node, plans);
            if self.req.expand_fasteners && !doc.fasteners.is_empty() {
                self.deferred_fasteners.insert(node);
            }
            self.declare_envelopes(node, &doc, &env);
            return;
        } else {
            self.solve_scope(node, plans);
        }
        // 6: fasteners.
        if self.req.expand_fasteners {
            self.expand_fasteners(node, &doc, &env);
        }
        self.declare_envelopes(node, &doc, &env);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn compose_occurrence(
        &mut self,
        parent: NodeId,
        doc: &Arc<AssemblyDocument>,
        env: &ValueEnv,
        o: &Occurrence,
        segment: PathSegment,
        pending: &[Pending],
        origin: NodeOrigin,
        stack: &[FilePath],
        parent_flexible: bool,
    ) -> Option<NodeId> {
        let file = doc.path.clone();
        let pdepth = self.nodes[parent as usize].depth;
        // Overrides that land here: outer (lower depth) wins per slot.
        let mut here: BTreeMap<String, &Pending> = BTreeMap::new();
        let mut shadowed: Vec<(&Pending, &Pending)> = Vec::new();
        for p in pending {
            if p.remaining.len() == 1 && p.remaining[0] == segment {
                let slot = p.kind.slot();
                match here.get(&slot) {
                    Some(q) if q.depth <= p.depth => shadowed.push((p, q)),
                    Some(q) => {
                        shadowed.push((q, p));
                        here.insert(slot, p);
                    }
                    None => {
                        here.insert(slot, p);
                    }
                }
            }
        }
        let child_path = {
            let p = self.display_path(parent);
            if p.is_empty() {
                segment.to_string()
            } else {
                format!("{}/{}", p, segment)
            }
        };
        for (lost, won) in shadowed {
            self.diag(
                DiagCode::AsmOverrideShadowed,
                Severity::Info,
                Site::with_param(Subject::Override { index: lost.index }, lost.file.as_str()),
                format!(
                    "override {} in `{}` sets `{}` of `{}`, but `{}` (outer) sets it too and wins",
                    lost.index,
                    lost.file,
                    lost.kind.slot(),
                    child_path,
                    won.file
                ),
                Vec::new(),
            );
        }
        if let Some(p) = here.get("suppress") {
            if let OverrideKind::Suppress { suppressed: true } = p.kind {
                self.stats.suppressed += 1;
                self.suppressed.push(child_path);
                return None;
            }
        }

        // Effective source and configuration.
        let mut selection = o.configuration.clone();
        if let Some(p) = here.get("configuration") {
            if let OverrideKind::Configuration { select } = &p.kind {
                selection = select.clone();
            }
        }
        let mut extra_inputs: BTreeMap<InputName, ValueExpr> = BTreeMap::new();
        for (slot, p) in &here {
            if let (true, OverrideKind::Input { name, value }) =
                (slot.starts_with("input."), &p.kind)
            {
                extra_inputs.insert(name.clone(), value.clone());
            }
        }
        if !extra_inputs.is_empty() {
            selection = match selection {
                ConfigurationSelection::Default => ConfigurationSelection::Inputs {
                    base: None,
                    values: extra_inputs,
                },
                ConfigurationSelection::Named { name } => ConfigurationSelection::Inputs {
                    base: Some(name),
                    values: extra_inputs,
                },
                ConfigurationSelection::Inputs { base, mut values } => {
                    values.extend(extra_inputs);
                    ConfigurationSelection::Inputs { base, values }
                }
                other => other,
            };
        }
        let source = match here.get("replace").map(|p| &p.kind) {
            Some(OverrideKind::Replace { source }) => source,
            _ => &o.source,
        };
        let revision = match &source.context {
            XrefContext::Pinned { revision } => Some(revision.clone()),
            XrefContext::Live => None,
        };
        let def = source.definition.clone();
        let site = Site::with_param(
            Subject::Occurrence {
                path: child_path.clone(),
            },
            file.as_str(),
        );

        // Live across partitions is not allowed (decision 11).
        if revision.is_none() {
            let mine = self.ctx.source.partition(&file);
            let theirs = self.ctx.source.partition(&def.file);
            if mine.is_some() && theirs.is_some() && mine != theirs {
                let rel = self.ctx.source.latest_release(&def.file);
                self.diag(
                    DiagCode::AsmLiveAcrossPartition,
                    Severity::Error,
                    site.clone(),
                    format!(
                        "`{}` uses `{}` live, but it is in another partition; pin it to a released revision",
                        child_path, def.file
                    ),
                    rel.map(|r| {
                        vec![Repair::PinOccurrence {
                            assembly: file.clone(),
                            occurrence: o.name.clone(),
                            revision: r,
                        }]
                    })
                    .unwrap_or_default(),
                );
            }
        }

        let summary = self.definition(&def, revision.as_ref());
        let (config, part_number, failed, def_material) = match &summary {
            Ok(s) => {
                let s = s.clone();
                let key = match self.eval_selection(&selection, env, &s, &file, &site) {
                    Ok(k) => k,
                    Err((code, msg)) => {
                        self.diag(code, Severity::Error, site.clone(), msg, Vec::new());
                        ConfigKey::default()
                    }
                };
                let part_number = s.part_number_in(&key);
                (key, part_number, false, s.material.clone())
            }
            Err(e) => {
                self.diag(
                    DiagCode::AsmUnknownDefinition,
                    Severity::Error,
                    site.clone(),
                    format!("`{}`: {}", child_path, e),
                    Vec::new(),
                );
                (ConfigKey::default(), None, true, None)
            }
        };

        // Representation: override, own, or the parent's.
        let parent_rep = self.nodes[parent as usize].representation;
        let parent_by_ancestor = self.nodes[parent as usize].represented_by_ancestor
            || matches!(
                parent_rep,
                Representation::Envelope | Representation::Interface
            ) && self.nodes[parent as usize].parent.is_some();
        let mut rep = o.representation.unwrap_or(parent_rep);
        if let Some(p) = here.get("representation") {
            if let OverrideKind::Representation { representation } = p.kind {
                rep = representation;
            }
        }
        if let Ok(s) = &summary {
            if !s.provides(rep) {
                self.diag(
                    DiagCode::AsmRepresentationUnavailable,
                    Severity::Warning,
                    site.clone(),
                    format!(
                        "`{}` declares no `{:?}` representation; `full` is used",
                        def, rep
                    ),
                    Vec::new(),
                );
                rep = Representation::Full;
            }
        }

        let mut placement_fixed = o.fixed;
        let mut flexible = o.flexible;
        if let Some(p) = here.get("flexible") {
            if let OverrideKind::Flexible { flexible: f } = p.kind {
                flexible = f;
            }
        }
        if let Some(p) = here.get("position") {
            if !(o.fixed || parent_flexible) {
                self.diag(
                    DiagCode::AsmOverrideNotAllowed,
                    Severity::Error,
                    Site::with_param(Subject::Override { index: p.index }, p.file.as_str()),
                    format!(
                        "`{}` is positioned by mates in a rigid sub-assembly; a position override cannot move it (make the sub-assembly flexible in this context)",
                        child_path
                    ),
                    Vec::new(),
                );
            } else {
                placement_fixed = true;
            }
        }
        let mut material = o.material.clone().or(def_material);
        if let Some(p) = here.get("material") {
            if let OverrideKind::Material { material: m } = &p.kind {
                material = Some(m.clone());
            }
        }
        let mut properties = o.properties.clone();
        for (slot, p) in &here {
            if let (true, OverrideKind::Property { property }) =
                (slot.starts_with("property."), &p.kind)
            {
                properties.retain(|q| q.name != property.name);
                properties.push(property.clone());
            }
        }

        // Identity.
        let (local_id, locked) = match &origin {
            NodeOrigin::Authored | NodeOrigin::Root => {
                self.local_uuid(&file, "occurrence", &o.name)
            }
            NodeOrigin::Pattern { pattern } => {
                let (pid, pl) = self.local_uuid(&file, "pattern", pattern);
                let (sid, sl) = self.local_uuid(&file, "occurrence", &o.name);
                let key = segment
                    .instance
                    .as_ref()
                    .map(|i| i.key.to_string())
                    .unwrap_or_default();
                (
                    hash16(&[
                        b"runmat-cad/occurrence/v1/instance",
                        &pid.0.to_be_bytes(),
                        &sid.0.to_be_bytes(),
                        key.as_bytes(),
                    ]),
                    pl && sl,
                )
            }
            NodeOrigin::Mirror { mirror } => {
                let (mid, ml) = self.local_uuid(&file, "mirror", mirror);
                let (sid, sl) = self.local_uuid(&file, "occurrence", &o.name);
                (
                    hash16(&[
                        b"runmat-cad/occurrence/v1/mirror",
                        &mid.0.to_be_bytes(),
                        &sid.0.to_be_bytes(),
                    ]),
                    ml && sl,
                )
            }
        };
        let parent_uuid = self.nodes[parent as usize].uuid;
        let uuid = hash16(&[
            b"runmat-cad/occurrence/v1/node",
            &parent_uuid.0.to_be_bytes(),
            &local_id.0.to_be_bytes(),
        ]);
        let id = self.nodes.len() as NodeId;
        let proto = self.prototype(&def, revision.as_ref(), &config, rep);
        let attached = match here.get("position").map(|p| &p.kind) {
            Some(OverrideKind::Position { placement }) if placement_fixed => {
                matches!(placement, Placement::Attach { .. })
            }
            _ => matches!(o.placement, Placement::Attach { .. }),
        };
        self.nodes.push(OccNode {
            parent: Some(parent),
            segment: segment.clone(),
            depth: pdepth + 1,
            uuid,
            locked,
            definition: def.clone(),
            revision: revision.clone(),
            configuration: config.clone(),
            authored_in: file.clone(),
            origin,
            local: Transform::IDENTITY,
            world: Transform::IDENTITY,
            fixed: placement_fixed,
            attached,
            flexible: flexible && def.kind == DefinitionKind::Assembly,
            pose: PoseStatus::Placed,
            dof: None,
            representation: rep,
            represented_by_ancestor: parent_by_ancestor,
            material,
            properties,
            part_number,
            prototype: proto,
            children: Vec::new(),
            expanded: false,
            failed,
        });
        self.nodes[parent as usize].children.push(id);
        // Position override: replaces the placement (applied in place_children through this map).
        if let Some(p) = here.get("position") {
            if let (OverrideKind::Position { placement }, true) = (&p.kind, placement_fixed) {
                self.position_overrides.insert(id, placement.clone());
            }
        }

        // Descend into sub-assemblies.
        if def.kind == DefinitionKind::Assembly && !failed {
            let depth_ok = self.req.max_depth.map(|m| pdepth + 1 < m).unwrap_or(true);
            if !depth_ok || self.req.collapse.contains(&def.file) {
                self.stats.collapsed += 1;
                return Some(id);
            }
            if stack.contains(&def.file) {
                let mut cycle: Vec<String> = stack.iter().map(|f| f.0.clone()).collect();
                cycle.push(def.file.0.clone());
                self.diag(
                    DiagCode::AsmCycle,
                    Severity::Error,
                    site,
                    format!("assemblies use each other: {}", cycle.join(" → ")),
                    Vec::new(),
                );
                self.nodes[id as usize].failed = true;
                return Some(id);
            }
            match self.ctx.source.assembly(&def.file, revision.as_ref()) {
                Ok(sub) => {
                    let mut child_pending: Vec<Pending> = pending
                        .iter()
                        .filter(|p| p.remaining.len() > 1 && p.remaining[0] == segment)
                        .map(|p| Pending {
                            remaining: p.remaining[1..].to_vec(),
                            ..p.clone()
                        })
                        .collect();
                    for (i, ov) in o.overrides.iter().enumerate() {
                        let osite = Site::with_param(Subject::Override { index: i }, file.as_str());
                        if !self.holds(ov.when.as_ref(), &file, &osite) {
                            continue;
                        }
                        if ov.target.is_empty() {
                            continue;
                        }
                        child_pending.push(Pending {
                            remaining: ov.target.0.clone(),
                            kind: ov.set.clone(),
                            file: file.clone(),
                            index: i,
                            depth: pdepth,
                        });
                    }
                    let (overrides, suppress) = self.assembly_overrides(&sub, &config);
                    let env = self.env_for(&sub, &overrides);
                    let mut st = stack.to_vec();
                    st.push(def.file.clone());
                    self.nodes[id as usize].expanded = true;
                    let flex = self.nodes[id as usize].flexible;
                    self.compose_assembly(id, sub, env, child_pending, st, flex, &suppress);
                }
                Err(e) => {
                    self.diag(
                        DiagCode::AsmUnknownDefinition,
                        Severity::Error,
                        site,
                        format!("`{}`: {}", child_path, e),
                        Vec::new(),
                    );
                    self.nodes[id as usize].failed = true;
                }
            }
        } else if !o.overrides.is_empty() {
            for (i, ov) in o.overrides.iter().enumerate() {
                if ov.target.is_empty() {
                    continue;
                }
                self.diag(
                    DiagCode::AsmOverrideTargetMissing,
                    Severity::Error,
                    Site::with_param(Subject::Override { index: i }, file.as_str()),
                    format!(
                        "override {} of `{}` targets `{}`, but a {:?} has no occurrences",
                        i, o.name, ov.target, def.kind
                    ),
                    vec![Repair::RemoveOverride {
                        assembly: file.clone(),
                        index: i,
                    }],
                );
            }
        }
        Some(id)
    }
}
