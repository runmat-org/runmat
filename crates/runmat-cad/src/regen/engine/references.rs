//! Feature status and reference resolution during a run: halts and their root causes, the
//! resolver, consumed and noted references, and the gate on a feature's inputs.

use super::*;

impl<'a, 'b> Run<'a, 'b> {
    pub(super) fn set_status(&mut self, f: &FeatureId, s: MakerStatus) {
        if let Some(e) = self.st.table.entries.get_mut(f) {
            e.status = Some(s);
        }
    }

    pub(super) fn halt(&mut self, f: &FeatureId, h: Halt) -> FeatureStatus {
        match h {
            Halt::Wait(reason) => {
                let root = match &reason.cause {
                    WaitCause::MakerWaiting { root } => root.clone(),
                    _ => reason.maker.clone(),
                };
                self.set_status(f, MakerStatus::Waiting { root });
                FeatureStatus::Waiting { reason }
            }
            Halt::Fail(d) => {
                self.failure_diag.insert(format!("feature.{}", f), d);
                self.set_status(f, MakerStatus::Failed);
                FeatureStatus::Failed { diagnostic: d }
            }
            Halt::Block(root, d) => {
                self.failure_diag.insert(format!("feature.{}", f), d);
                self.set_status(f, MakerStatus::Blocked { root: root.clone() });
                FeatureStatus::Blocked {
                    root,
                    diagnostic: d,
                }
            }
        }
    }

    pub(super) fn root_of(&self, root: &FeatureId) -> u32 {
        // A cascade points at the root diagnostic, following cascades back.
        let id = self
            .failure_diag
            .get(&format!("feature.{}", root))
            .copied()
            .unwrap_or(0);
        match self.diags.diagnostics.get(id as usize).map(|d| d.cause) {
            Some(Cause::Cascade { root }) => root,
            _ => id,
        }
    }

    pub(super) fn error(
        &mut self,
        f: &FeatureId,
        param: Option<&str>,
        code: DiagCode,
        message: String,
        repairs: Vec<Repair>,
    ) -> Halt {
        Halt::Fail(self.diags.push(
            code,
            Severity::Error,
            Site::feature(f, param),
            message,
            Cause::Root,
            repairs,
        ))
    }

    pub(super) fn resolver(&self, consumer: Option<usize>) -> Resolver<'_> {
        Resolver {
            file: &self.doc.path,
            bodies: self.st.views(),
            features: &self.st.table,
            consumer,
            burials: &self.st.burials,
            publishes: &self.doc.publishes,
            lock: self.ctx.lock,
            catalog: self.ctx.catalog,
            policy: self.ctx.policy,
            names: Some(&self.st.names),
        }
    }

    /// Record an outcome, emit its diagnostics, and turn it into a resolution or a halt.
    pub(super) fn take(
        &mut self,
        site: Site,
        reference: &Reference,
        outcome: ResolveOutcome,
    ) -> Result<Resolution, Halt> {
        self.stats.record(&outcome);
        self.records.push(ReferenceRecord {
            param: site.param.clone().unwrap_or_default(),
            reference: reference.clone(),
            outcome: outcome.clone(),
            entities: Vec::new(),
        });
        match outcome {
            ResolveOutcome::Resolved(r) => {
                for n in &r.notes {
                    self.note(&site, reference, n);
                }
                let names: Vec<Vec<EntityName>> = r
                    .entities()
                    .iter()
                    .map(|h| self.st.entity_names(h))
                    .collect();
                let names = match &r.target {
                    Resolved::Foreign { target } => {
                        target.entities.iter().map(|e| e.names.clone()).collect()
                    }
                    _ => names,
                };
                if let Some(rec) = self.records.last_mut() {
                    rec.entities = names.clone();
                }
                self.ref_ids
                    .push((site.param.clone().unwrap_or_default(), names));
                Ok(r)
            }
            ResolveOutcome::Waiting(w) => {
                let (msg, repairs) = describe_wait_in(reference, &w, Some(&self.st.names));
                self.diags.push(
                    DiagCode::RefWaiting,
                    Severity::Info,
                    site,
                    msg,
                    Cause::Root,
                    repairs,
                );
                Err(Halt::Wait(w))
            }
            ResolveOutcome::Failed(RefProblem::Blocked { maker, root }) => {
                let root_diag = self.root_of(&root);
                let (code, msg, repairs) = describe_ref_problem_in(
                    &site,
                    reference,
                    &RefProblem::Blocked {
                        maker,
                        root: root.clone(),
                    },
                    Some(&self.st.names),
                );
                let d = self.diags.push(
                    code,
                    Severity::Error,
                    site,
                    msg,
                    Cause::Cascade { root: root_diag },
                    repairs,
                );
                Err(Halt::Block(root, d))
            }
            ResolveOutcome::Failed(p) => {
                let (code, msg, repairs) =
                    describe_ref_problem_in(&site, reference, &p, Some(&self.st.names));
                Err(Halt::Fail(self.diags.push(
                    code,
                    Severity::Error,
                    site,
                    msg,
                    Cause::Root,
                    repairs,
                )))
            }
        }
    }

    pub(super) fn note(&mut self, site: &Site, reference: &Reference, n: &ResolveNote) {
        match n {
            ResolveNote::ViaAlias {
                from,
                to,
                rewritten,
            } => {
                self.diags.push(
                    DiagCode::RefViaAlias,
                    Severity::Info,
                    site.clone(),
                    format!(
                        "`{}` was renamed to `{}`; resolved through the lock alias",
                        from, to
                    ),
                    Cause::Root,
                    vec![Repair::RewriteReference {
                        site: site.clone(),
                        reference: rewritten.clone(),
                    }],
                );
            }
            ResolveNote::FallbackUsed { maker } => {
                self.diags.push(
                    DiagCode::RefFallbackUsed,
                    Severity::Warning,
                    site.clone(),
                    format!(
                        "`{}` resolved by geometry among `{}`'s entities, not by name; check it",
                        reference, maker
                    ),
                    Cause::Root,
                    vec![Repair::RePick {
                        site: site.clone(),
                        candidates: Vec::new(),
                    }],
                );
            }
            ResolveNote::PinBehind {
                file,
                pinned,
                latest,
            } => {
                self.diags.push(
                    DiagCode::XrefPinBehind,
                    Severity::Info,
                    site.clone(),
                    format!("`{}` is pinned to {}; {} is released", file, pinned, latest),
                    Cause::Root,
                    vec![Repair::UpdatePin {
                        site: site.clone(),
                        file: file.clone(),
                        from: pinned.clone(),
                        to: latest.clone(),
                    }],
                );
            }
            ResolveNote::Dormant { waiting } => {
                let makers: Vec<String> = waiting.iter().map(|w| w.maker.0.clone()).collect();
                self.diags.push(
                    DiagCode::RefDormant,
                    Severity::Info,
                    site.clone(),
                    format!(
                        "{} member(s) of `{}` are waiting on {}; the rest apply",
                        waiting.len(),
                        site.key(),
                        makers.join(", ")
                    ),
                    Cause::Root,
                    makers
                        .into_iter()
                        .map(|m| Repair::Unsuppress {
                            feature: FeatureId::new(m),
                        })
                        .collect(),
                );
            }
            ResolveNote::XrefRenamed {
                from_file,
                to_file,
                from_name,
                to_name,
                rewritten,
            } => {
                let what = match (from_file == to_file, from_name == to_name) {
                    (false, false) => format!(
                        "`{}` → `{}` and `{}` → `{}`",
                        from_file, to_file, from_name, to_name
                    ),
                    (false, true) => format!("`{}` → `{}`", from_file, to_file),
                    _ => format!("`{}` → `{}` in `{}`", from_name, to_name, to_file),
                };
                self.diags.push(
                    DiagCode::XrefViaAlias,
                    Severity::Info,
                    site.clone(),
                    format!("renamed ({}); resolved through the lock alias", what),
                    Cause::Root,
                    vec![Repair::RewriteReference {
                        site: site.clone(),
                        reference: rewritten.clone(),
                    }],
                );
            }
            ResolveNote::Merged { .. } | ResolveNote::Pieces { .. } => {}
        }
    }

    pub(super) fn single(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        r: &Reference,
        kind: Option<TopoKind>,
    ) -> Result<Resolution, Halt> {
        let o = self
            .resolver(Some(i))
            .resolve(r, kind, Cardinality::ExactlyOne);
        self.take(Site::feature(f, Some(param)), r, o)
    }

    pub(super) fn list(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        refs: &[Reference],
        kind: Option<TopoKind>,
    ) -> Result<BTreeSet<EntityHandle>, Halt> {
        let (each, combined) = self.resolver(Some(i)).resolve_list(refs, kind);
        // Record and diagnose per member (waiting members are dormant, not halts).
        let mut set = BTreeSet::new();
        let mut first_halt = None;
        let mut waiting = Vec::new();
        for (k, (r, o)) in refs.iter().zip(each).enumerate() {
            let site = Site::feature(f, Some(&format!("{}[{}]", param, k)));
            match o {
                ResolveOutcome::Waiting(w) => {
                    self.stats.record(&ResolveOutcome::Waiting(w.clone()));
                    self.records.push(ReferenceRecord {
                        param: site.param.clone().unwrap_or_default(),
                        reference: r.clone(),
                        outcome: ResolveOutcome::Waiting(w.clone()),
                        entities: Vec::new(),
                    });
                    waiting.push(w);
                }
                other => match self.take(site, r, other) {
                    Ok(res) => set.extend(res.entities()),
                    Err(h) => {
                        if first_halt.is_none() {
                            first_halt = Some(h);
                        }
                    }
                },
            }
        }
        if let Some(h) = first_halt {
            return Err(h);
        }
        if let ResolveOutcome::Waiting(w) = combined {
            let (msg, repairs) = describe_wait_in(&refs[0], &w, Some(&self.st.names));
            self.diags.push(
                DiagCode::RefWaiting,
                Severity::Info,
                Site::feature(f, Some(param)),
                msg,
                Cause::Root,
                repairs,
            );
            return Err(Halt::Wait(w));
        }
        if !waiting.is_empty() {
            self.note(
                &Site::feature(f, Some(param)),
                &refs[0],
                &ResolveNote::Dormant { waiting },
            );
        }
        Ok(set)
    }

    /// Gate a feature named directly (a profile sketch, a datum, a seed).
    pub(super) fn gate_feature(
        &mut self,
        i: usize,
        f: &FeatureId,
        dep: &FeatureId,
        param: &str,
    ) -> Result<(), Halt> {
        let entry = self.st.table.entries.get(dep).cloned();
        let site = Site::feature(f, Some(param));
        let Some(entry) = entry else {
            return Err(self.error(
                f,
                Some(param),
                DiagCode::RefMakerMissing,
                format!("`{}` is not in the file", dep),
                Vec::new(),
            ));
        };
        if entry.index >= i {
            return Err(self.error(
                f,
                Some(param),
                DiagCode::RefForward,
                format!("`{}` comes later in the history", dep),
                vec![Repair::Reorder {
                    feature: f.clone(),
                    after: dep.clone(),
                }],
            ));
        }
        let wait = |cause| WaitReason {
            maker: dep.clone(),
            cause,
        };
        let w = match entry.status {
            Some(MakerStatus::Ran) => return Ok(()),
            Some(MakerStatus::Suppressed) => wait(WaitCause::Suppressed),
            Some(MakerStatus::SuppressedInConfiguration) => {
                wait(WaitCause::SuppressedInConfiguration)
            }
            Some(MakerStatus::RolledBack) => wait(WaitCause::RolledBack),
            Some(MakerStatus::Waiting { root }) => wait(WaitCause::MakerWaiting { root }),
            Some(MakerStatus::Failed) | Some(MakerStatus::Blocked { .. }) | None => {
                let root = match entry.status {
                    Some(MakerStatus::Blocked { root }) => root,
                    _ => dep.clone(),
                };
                let root_diag = self.root_of(&root);
                let d = self.diags.push(
                    DiagCode::RegenCascade,
                    Severity::Error,
                    site,
                    format!("`{}` needs `{}`, which did not regenerate", f, dep),
                    Cause::Cascade { root: root_diag },
                    Vec::new(),
                );
                return Err(Halt::Block(root, d));
            }
        };
        let r = Reference::feature_body(dep.as_str());
        let (msg, repairs) = describe_wait_in(&r, &w, Some(&self.st.names));
        self.diags.push(
            DiagCode::RefWaiting,
            Severity::Info,
            site,
            msg,
            Cause::Root,
            repairs,
        );
        Err(Halt::Wait(w))
    }
}
