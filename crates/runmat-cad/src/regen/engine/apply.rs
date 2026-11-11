//! Running a compiled feature: kernel execution, naming its outputs, and committing the result
//! and the part's items.

use super::*;

impl<'a, 'b> Run<'a, 'b> {
    pub(super) fn apply(
        &mut self,
        _i: usize,
        f: &Feature,
        compiled: Compiled,
        key: &CacheKey,
        cache: &mut RegenCache,
    ) -> Result<FeatureStatus, Halt> {
        let id = &f.id;
        match compiled {
            Compiled::Record(r) => {
                self.st.records.insert(id.clone(), r);
                Ok(FeatureStatus::Regenerated)
            }
            Compiled::Datum(d) => {
                self.st.datums.insert(id.clone(), d);
                Ok(FeatureStatus::Regenerated)
            }
            Compiled::Sketch(s, frame) => {
                self.st.sketches.insert(id.clone(), (s, frame));
                Ok(FeatureStatus::Regenerated)
            }
            Compiled::Delete(bodies) => {
                for &b in bodies.iter().rev() {
                    let body = self.st.bodies.remove(b);
                    for n in body.names.body {
                        self.st
                            .burials
                            .entry((TopoKind::Body, n))
                            .or_insert_with(|| id.clone());
                    }
                    for n in body.names.faces.into_iter().flatten() {
                        self.st
                            .burials
                            .entry((TopoKind::Face, n))
                            .or_insert_with(|| id.clone());
                    }
                }
                Ok(FeatureStatus::Regenerated)
            }
            Compiled::Op {
                op,
                operands,
                naming,
                import,
                instances,
                tools,
            } => {
                let record = self.pending_record.take();
                let dressup = self.dressup_of(&op, &operands);
                let bodies = std::mem::take(&mut self.st.bodies);
                let ran = self.execute(
                    id,
                    f.frozen.as_ref(),
                    &op,
                    &operands,
                    &naming,
                    import,
                    key,
                    cache,
                    &bodies,
                );
                self.st.bodies = bodies;
                let (status, output, named) = ran?;
                let buried = Self::commit(&mut self.st.bodies, &operands, output, named);
                for (kind, n) in buried {
                    self.st
                        .burials
                        .entry((kind, n))
                        .or_insert_with(|| id.clone());
                }
                if let Some(keys) = instances {
                    self.st.table.instances.insert(id.clone(), keys);
                }
                if let Some(t) = tools {
                    self.st.tools.insert(id.clone(), t);
                }
                if let Some(d) = dressup {
                    self.st.dressups.insert(id.clone(), d);
                }
                if let Some(r) = record {
                    self.st.records.insert(id.clone(), r);
                }
                Ok(status)
            }
            Compiled::Steps(steps) => {
                let record = self.pending_record.take();
                let mut working = self.st.bodies.clone();
                let mut buried: Vec<(TopoKind, EntityName)> = Vec::new();
                let mut statuses = Vec::new();
                for (k, step) in steps.iter().enumerate() {
                    let ran = self.execute(
                        id,
                        f.frozen.as_ref().map(|fk| fk.step(k)).as_ref(),
                        &step.op,
                        &step.operands,
                        &step.naming,
                        false,
                        &key.step(k),
                        cache,
                        &working,
                    );
                    let (status, output, named) = ran?;
                    statuses.push(status);
                    buried.extend(Self::commit(&mut working, &step.operands, output, named));
                }
                // Names alive after the last step were not removed, whatever an earlier step
                // did; names only an intermediate step made were never seen.
                let alive: BTreeSet<&EntityName> = working
                    .iter()
                    .flat_map(|b| b.names.body.iter().chain(b.names.faces.iter().flatten()))
                    .collect();
                let mut dead: Vec<(TopoKind, EntityName)> = Vec::new();
                for (kind, n) in buried {
                    if !alive.contains(&n) && n.maker() != id && !dead.contains(&(kind, n.clone()))
                    {
                        dead.push((kind, n));
                    }
                }
                // A name a step restored is alive again: it is no longer buried.
                let restored: Vec<(TopoKind, EntityName)> = self
                    .st
                    .burials
                    .keys()
                    .filter(|(_, n)| alive.contains(n))
                    .cloned()
                    .collect();
                for k in restored {
                    self.st.burials.remove(&k);
                }
                for (kind, n) in dead {
                    self.st
                        .burials
                        .entry((kind, n))
                        .or_insert_with(|| id.clone());
                }
                self.st.bodies = working;
                if let Some(r) = record {
                    self.st.records.insert(id.clone(), r);
                }
                Ok(if statuses.contains(&FeatureStatus::Regenerated) {
                    FeatureStatus::Regenerated
                } else if statuses.contains(&FeatureStatus::Frozen) {
                    FeatureStatus::Frozen
                } else {
                    FeatureStatus::CacheHit
                })
            }
        }
    }

    /// Runs one kernel operation on `bodies` (from the cache when it holds the key, or the
    /// frozen result) and names its outputs.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute(
        &mut self,
        id: &FeatureId,
        frozen: Option<&CacheKey>,
        op: &KernelOp,
        operands: &[usize],
        naming: &NamingContext,
        import: bool,
        key: &CacheKey,
        cache: &mut RegenCache,
        bodies: &[LiveBody],
    ) -> Result<(FeatureStatus, KernelOutput, NamingOutput), Halt> {
        let mut status = FeatureStatus::Regenerated;
        let mut output = None;
        if let Some(frozen) = frozen {
            match cache.entries.get(frozen) {
                Some(o) => {
                    if frozen != key {
                        self.diags.push(
                            DiagCode::RegenFrozenStale,
                            Severity::Info,
                            Site::feature(id, None),
                            format!("`{}` is frozen; upstream changes since the freeze are not applied to it", id),
                            Cause::Root,
                            Vec::new(),
                        );
                    }
                    output = Some(o.clone());
                    status = FeatureStatus::Frozen;
                }
                None => {
                    self.diags.push(
                        DiagCode::RegenFrozenMissing,
                        Severity::Warning,
                        Site::feature(id, None),
                        format!(
                            "`{}` is frozen but its cached result is gone; regenerated instead",
                            id
                        ),
                        Cause::Root,
                        Vec::new(),
                    );
                }
            }
        }
        if output.is_none() {
            if let Some(o) = cache.entries.get(key) {
                cache.hits += 1;
                status = FeatureStatus::CacheHit;
                output = Some(o.clone());
            }
        }
        let output = match output {
            Some(o) => o,
            None => {
                let handles: Vec<&BodyHandle> =
                    operands.iter().map(|b| &bodies[*b].handle).collect();
                match self.ctx.kernel.execute(op, &handles) {
                    Ok(o) => {
                        cache.misses += 1;
                        cache.entries.insert(key.clone(), o.clone());
                        o
                    }
                    Err(e) => {
                        let code = match &e {
                            KernelError::NotSupported { .. } => DiagCode::KernelNotSupported,
                            KernelError::HistoryIncomplete { .. } => {
                                DiagCode::KernelHistoryIncomplete
                            }
                            _ => DiagCode::KernelFailed,
                        };
                        return Err(self.error(id, None, code, e.to_string(), Vec::new()));
                    }
                }
            }
        };
        let named = {
            let views: Vec<OperandView<'_>> = operands
                .iter()
                .map(|b| OperandView {
                    names: &bodies[*b].names,
                    topology: &bodies[*b].topology,
                })
                .collect();
            let outs: Vec<&Topology> = output.bodies.iter().map(|b| &b.topology).collect();
            name_outputs(naming, &views, &outs, &output.history)
        };
        let mut named = match named {
            Ok(n) => n,
            Err(e) => {
                return Err(self.error(
                    id,
                    None,
                    DiagCode::KernelHistoryIncomplete,
                    e.to_string(),
                    Vec::new(),
                ))
            }
        };
        if import {
            cad_names_for_import(id, &mut named);
        }
        if let Err(c) = named.intern_into(&mut self.st.names) {
            return Err(self.error(
                id,
                None,
                DiagCode::RefTokenCollision,
                c.to_string(),
                Vec::new(),
            ));
        }
        for tie in &named.slot_ties {
            let Some(first) = tie.names.first() else {
                continue;
            };
            let body_ref = Reference::body(first.clone());
            let candidates = slot_tie_candidates(&body_ref, tie, &named.bodies);
            self.diags.push(
                DiagCode::RefBodySlotTie,
                Severity::Warning,
                Site::feature(id, None),
                format!(
                    "`{}` split `{}` into {} pieces equally similar to it by lineage; \
                     references to it now select all of them",
                    id,
                    Shown(&body_ref, Some(&self.st.names)),
                    tie.bodies.len()
                ),
                Cause::Root,
                vec![Repair::Disambiguate {
                    site: Site::feature(id, None),
                    candidates,
                }],
            );
        }
        Ok((status, output, named))
    }

    /// Replaces the operands in `bodies` by the outputs (at the first operand's place) and
    /// returns the names the operation buried.
    pub(super) fn commit(
        bodies: &mut Vec<LiveBody>,
        operands: &[usize],
        output: KernelOutput,
        named: NamingOutput,
    ) -> Vec<(TopoKind, EntityName)> {
        let insert_at = operands.first().copied().unwrap_or(bodies.len());
        let mut gone: Vec<usize> = operands.to_vec();
        gone.sort_unstable();
        gone.dedup();
        for &b in gone.iter().rev() {
            bodies.remove(b);
        }
        let insert_at = insert_at.min(bodies.len());
        for (k, (kb, names)) in output.bodies.into_iter().zip(named.bodies).enumerate() {
            bodies.insert(
                insert_at + k,
                LiveBody {
                    handle: kb.handle,
                    topology: kb.topology,
                    names,
                },
            );
        }
        named.buried
    }

    #[allow(clippy::type_complexity)]
    pub(super) fn items(
        &mut self,
    ) -> (
        BTreeMap<PartName, ItemOutcome>,
        BTreeMap<RegionName, ItemOutcome>,
        Vec<ItemOutcome>,
        PublishTable,
    ) {
        let doc = self.doc;
        let mut parts = BTreeMap::new();
        for p in &doc.parts {
            self.records.clear();
            let o =
                self.resolver(None)
                    .resolve(&p.body, Some(TopoKind::Body), Cardinality::ExactlyOne);
            let site = Site {
                subject: Subject::Part {
                    part: p.name.clone(),
                },
                param: Some("body".into()),
            };
            let r = self.take(site, &p.body, o);
            let entities = match &r {
                Ok(res) => res
                    .entities()
                    .iter()
                    .map(|h| self.st.entity_names(h))
                    .collect(),
                Err(_) => Vec::new(),
            };
            parts.insert(
                p.name.clone(),
                ItemOutcome {
                    references: std::mem::take(&mut self.records),
                    entities,
                    ok: r.is_ok(),
                },
            );
        }
        let mut regions = BTreeMap::new();
        for rg in &doc.regions {
            self.records.clear();
            let (each, _) = self.resolver(None).resolve_list(&rg.select, Some(rg.kind));
            let mut set = BTreeSet::new();
            let mut foreign: Vec<Vec<EntityName>> = Vec::new();
            let mut ok = true;
            for (k, (r, o)) in rg.select.iter().zip(each).enumerate() {
                // An optional region (`expect = any`) selects nothing from a face its feature
                // doesn't make at these inputs: a full-turn revolve has no start or end face.
                if rg.expect == Cardinality::Any
                    && matches!(o, ResolveOutcome::Failed(RefProblem::NotGenerated { .. }))
                {
                    continue;
                }
                let site = Site {
                    subject: Subject::Region {
                        region: rg.name.clone(),
                    },
                    param: Some(format!("select[{}]", k)),
                };
                match self.take(site, r, o) {
                    Ok(res) => {
                        if let Resolved::Foreign { target } = &res.target {
                            foreign.extend(target.entities.iter().map(|e| e.names.clone()));
                        }
                        set.extend(res.entities());
                    }
                    Err(_) => ok = false,
                }
            }
            if ok {
                let n = set.len() + foreign.len();
                let bad = match rg.expect {
                    Cardinality::ExactlyOne => n != 1,
                    Cardinality::AtLeastOne => n == 0,
                    Cardinality::Any => false,
                };
                if bad {
                    ok = false;
                    let candidates = if n > 1 && rg.select.len() == 1 {
                        self.resolver(None).disambiguate(&rg.select[0], &set)
                    } else {
                        Vec::new()
                    };
                    let site = Site {
                        subject: Subject::Region {
                            region: rg.name.clone(),
                        },
                        param: None,
                    };
                    self.diags.push(
                        DiagCode::RegionExpectation,
                        Severity::Error,
                        site.clone(),
                        format!(
                            "region `{}` expects {:?} but selects {}",
                            rg.name, rg.expect, n
                        ),
                        Cause::Root,
                        vec![Repair::Disambiguate { site, candidates }],
                    );
                }
            }
            let mut entities: Vec<Vec<EntityName>> =
                set.iter().map(|h| self.st.entity_names(h)).collect();
            entities.extend(foreign);
            regions.insert(
                rg.name.clone(),
                ItemOutcome {
                    references: std::mem::take(&mut self.records),
                    entities,
                    ok,
                },
            );
        }
        let mut materials = Vec::new();
        for (k, m) in doc.materials.iter().enumerate() {
            self.records.clear();
            let o =
                self.resolver(None)
                    .resolve(&m.body, Some(TopoKind::Body), Cardinality::AtLeastOne);
            let site = Site {
                subject: Subject::Material { index: k },
                param: Some("body".into()),
            };
            let r = self.take(site, &m.body, o);
            let entities = match &r {
                Ok(res) => res
                    .entities()
                    .iter()
                    .map(|h| self.st.entity_names(h))
                    .collect(),
                Err(_) => Vec::new(),
            };
            materials.push(ItemOutcome {
                references: std::mem::take(&mut self.records),
                entities,
                ok: r.is_ok(),
            });
        }
        let mut table = PublishTable::new();
        for p in &doc.publishes {
            self.records.clear();
            let site = Site {
                subject: Subject::Publish {
                    publish: p.name.clone(),
                },
                param: None,
            };
            let target = ForeignTarget {
                file: doc.path.clone(),
                revision: None,
                name: p.name.clone(),
                entities: Vec::new(),
                datum: None,
                value: None,
            };
            let result = match &p.target {
                PublishTarget::Entities { kind, references } => {
                    let mut entities = Vec::new();
                    let mut ok = true;
                    for r in references {
                        let o =
                            self.resolver(None)
                                .resolve(r, Some(*kind), Cardinality::AtLeastOne);
                        match self.take(site.clone(), r, o) {
                            Ok(res) => {
                                for h in res.entities() {
                                    entities.push(ForeignEntity {
                                        kind: h.kind,
                                        names: self.st.entity_names(&h),
                                    });
                                }
                            }
                            Err(_) => ok = false,
                        }
                    }
                    if ok {
                        Ok(ForeignTarget { entities, ..target })
                    } else {
                        Err(())
                    }
                }
                PublishTarget::Datum { datum } => match datum {
                    DatumRef::Feature { feature } => match self.st.datums.get(feature) {
                        Some(d) => Ok(ForeignTarget {
                            datum: Some(*d),
                            ..target
                        }),
                        None => Err(()),
                    },
                    DatumRef::OriginPlane { plane } => Ok(ForeignTarget {
                        datum: Some(DatumValue::Plane {
                            frame: plane.frame(),
                        }),
                        ..target
                    }),
                    _ => Ok(ForeignTarget {
                        datum: Some(DatumValue::Frame {
                            frame: Frame::WORLD,
                        }),
                        ..target
                    }),
                },
                PublishTarget::Region { region } => match regions.get(region) {
                    Some(o) if o.ok => Ok(ForeignTarget {
                        entities: o
                            .entities
                            .iter()
                            .map(|n| ForeignEntity {
                                kind: doc
                                    .regions
                                    .iter()
                                    .find(|r| &r.name == region)
                                    .map(|r| r.kind)
                                    .unwrap_or(TopoKind::Face),
                                names: n.clone(),
                            })
                            .collect(),
                        ..target
                    }),
                    _ => Err(()),
                },
                PublishTarget::Part { part } => match parts.get(part) {
                    Some(o) if o.ok => Ok(ForeignTarget {
                        entities: o
                            .entities
                            .iter()
                            .map(|n| ForeignEntity {
                                kind: TopoKind::Body,
                                names: n.clone(),
                            })
                            .collect(),
                        ..target
                    }),
                    _ => Err(()),
                },
                PublishTarget::Value { input } => match self.values.inputs.get(input) {
                    Some(q) => Ok(ForeignTarget {
                        value: Some(*q),
                        ..target
                    }),
                    None => Err(()),
                },
                PublishTarget::Let { name } => match self.values.lets.get(name) {
                    Some(q) => Ok(ForeignTarget {
                        value: Some(*q),
                        ..target
                    }),
                    None => Err(()),
                },
            };
            table.insert(
                p.name.clone(),
                result.map_err(|_| XrefProblem::PublisherBroken {
                    file: doc.path.clone(),
                    name: p.name.clone(),
                }),
            );
        }
        // Lock ids are looked up, never minted, during regeneration.
        if let Some(lock) = self.ctx.lock {
            for p in &doc.parts {
                let _ = lock.lookup(&ItemPath::part(&doc.path, &p.name));
            }
        }
        (parts, regions, materials, table)
    }
}
