use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver, TryRecvError};

use runmat_execution::{placement::PreparationPaybackEvidence, Digest};
use runmat_hir::FunctionId;
use runmat_jit::{
    entry::{EntryKey, EntryPublications, EntryRegistry, PublishedEntry},
    invalidation::{
        DependencyKey, DependencyTracker, PreparedDependencyObservation, ReadyDependencyObservation,
    },
    tiering::{CompilationMode, RepresentationProfile, TierDecision, TierSiteId, TieringSession},
};

use crate::ExecutableUnit;
#[cfg(test)]
mod diagnostic_tests;
mod diagnostics;
#[cfg(test)]
mod preparation_tests;

/// Core-owned publication boundary for generic-native products in one session.
/// JIT owns cells and dependency matching; Core supplies authoritative program,
/// catalog, and session-function revisions.
pub(crate) struct GenericNativeCache {
    entries: EntryRegistry,
    dependencies: DependencyTracker,
    project_scope_revision: String,
    tiering: TieringSession,
    /// Immutable within one cache generation. Reconfiguration replaces the
    /// complete cache, so old entries/pending results cannot cross policy.
    optimization: runmat_jit::CompilerOptimization,
    pending: BTreeMap<CompilationKey, PendingCompilation>,
    failed_compilations: BTreeSet<CompilationKey>,
    #[cfg(test)]
    compilations: usize,
}

pub(crate) struct PreparedProjectRevision {
    baseline_namespace: String,
    observation: PreparedDependencyObservation,
}

pub(crate) struct ReadyProjectRevision<'a> {
    observation: ReadyDependencyObservation<'a>,
    project_scope_revision: &'a mut String,
    failed_compilations: &'a mut BTreeSet<CompilationKey>,
    entries: &'a mut EntryRegistry,
    namespace: String,
}

impl PreparedProjectRevision {
    pub(crate) fn admit_commit(
        self,
        cache: &mut GenericNativeCache,
    ) -> Result<ReadyProjectRevision<'_>, runmat_runtime::RuntimeError> {
        if cache.project_scope_revision != self.baseline_namespace {
            return Err(super::error::stage(
                "NativeDependency",
                "project scope changed after preparation",
            ));
        }
        let namespace = self.observation.revision().to_owned();
        let observation = self
            .observation
            .admit_commit(&mut cache.dependencies)
            .map_err(|error| super::error::stage("NativeDependency", error))?;
        Ok(ReadyProjectRevision {
            observation,
            project_scope_revision: &mut cache.project_scope_revision,
            failed_compilations: &mut cache.failed_compilations,
            entries: &mut cache.entries,
            namespace,
        })
    }
}

impl ReadyProjectRevision<'_> {
    pub(crate) fn commit(self) {
        if self.namespace == *self.project_scope_revision {
            self.observation.commit();
            return;
        }
        let (_, current) = self.observation.commit_with_snapshot();
        *self.project_scope_revision = self.namespace;
        self.failed_compilations.clear();
        self.entries.invalidate(&current);
    }
}

struct PendingCompilation {
    dependencies: runmat_jit::invalidation::DependencySnapshot,
    receiver: Receiver<Result<super::compile::BackgroundCompiledGenericUnit, String>>,
}

pub(crate) struct ScheduledInvocation {
    pub(crate) published: Option<PublishedEntry>,
    pub(crate) ready_at_start: EntryPublications,
}

struct PreparedInvocation {
    dependency_keys: BTreeSet<DependencyKey>,
    current: runmat_jit::invalidation::DependencySnapshot,
    key: EntryKey,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CompilationKey {
    entry: EntryKey,
    tier: CompilationTier,
}

struct SpecializedCompilationRequest<'a> {
    unit: &'a ExecutableUnit,
    preferred_function: Option<&'a str>,
    profile: &'a RepresentationProfile,
    key: &'a EntryKey,
    current: &'a runmat_jit::invalidation::DependencySnapshot,
    dependency_keys: &'a BTreeSet<DependencyKey>,
    compilation: &'a CompilationKey,
    liveness_admission: runmat_native_codegen::NativeLivenessAdmission,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum CompilationTier {
    Generic,
    Specialized(Digest),
}

impl Default for GenericNativeCache {
    fn default() -> Self {
        let mut dependencies = DependencyTracker::default();
        dependencies
            .observe(
                DependencyKey::SessionCatalog,
                session_catalog_revision(&runmat_vm::FunctionRegistry::default())
                    .expect("the canonical empty session catalog is serializable"),
            )
            .expect("the canonical empty session catalog revision is valid");
        Self {
            entries: EntryRegistry::default(),
            dependencies,
            project_scope_revision: "loose".to_string(),
            tiering: TieringSession::default(),
            optimization: runmat_jit::CompilerOptimization::TierDefault,
            pending: BTreeMap::new(),
            failed_compilations: BTreeSet::new(),
            #[cfg(test)]
            compilations: 0,
        }
    }
}

impl GenericNativeCache {
    pub(crate) fn with_compilation_policy(
        tiering: runmat_jit::tiering::TieringConfig,
        optimization: runmat_jit::CompilerOptimization,
    ) -> Result<Self, &'static str> {
        Ok(Self {
            tiering: TieringSession::new(tiering)?,
            optimization,
            ..Self::default()
        })
    }

    pub(crate) fn compilation_policy(
        &self,
    ) -> (
        runmat_jit::tiering::TieringConfig,
        runmat_jit::CompilerOptimization,
    ) {
        (self.tiering.config(), self.optimization)
    }

    pub(crate) fn reconfigure_compilation_policy(
        &mut self,
        tiering: runmat_jit::tiering::TieringConfig,
        optimization: runmat_jit::CompilerOptimization,
    ) -> Result<(), &'static str> {
        if self.compilation_policy() == (tiering, optimization) {
            return Ok(());
        }
        let mut replacement = Self::with_compilation_policy(tiering, optimization)?;
        // Keep the installed project/catalog authority, not stale compiled
        // products. Active PublishedEntry owners retain their Rc executors.
        std::mem::swap(&mut replacement.dependencies, &mut self.dependencies);
        replacement.project_scope_revision = std::mem::take(&mut self.project_scope_revision);
        *self = replacement;
        Ok(())
    }
    pub(crate) fn has_retained_entries(&self) -> bool {
        self.entries.retained_cell_count() != 0
    }

    /// Snapshot products that existed before this invocation's tiering decision
    /// can compile synchronously. Only those products have zero ready compile
    /// charge and are eligible for a ready-route placement comparison.
    pub(crate) fn ready_publications(
        &self,
        unit: &ExecutableUnit,
        preferred_function: Option<&str>,
        profile: Option<&RepresentationProfile>,
    ) -> Result<EntryPublications, runmat_runtime::RuntimeError> {
        let Some(key) = ready_entry_key(unit, preferred_function) else {
            return Ok(EntryPublications::default());
        };
        Ok(self
            .entries
            .publications_borrowed(key.as_str(), profile.map(|profile| profile.digest)))
    }

    /// Resolve both entry-owned alternatives from the pre-preparation snapshot.
    /// A fresh specialization must not hide an existing generic publication,
    /// and a profileless invocation must never acquire specialized authority.
    pub(crate) fn ready_routes(
        &mut self,
        unit: &ExecutableUnit,
        preferred_function: Option<&str>,
        profile: Option<Digest>,
        initial: EntryPublications,
    ) -> Result<[Option<PublishedEntry>; 2], runmat_runtime::RuntimeError> {
        let Some(key) = ready_entry_key(unit, preferred_function) else {
            return Ok([None, None]);
        };
        Ok([
            self.entries
                .resolve_current_borrowed(key.as_str(), &self.dependencies, None),
            profile.and_then(|profile| {
                self.entries.resolve_current_borrowed(
                    key.as_str(),
                    &self.dependencies,
                    Some(profile),
                )
            }),
        ]
        .map(|entry| entry.filter(|entry| initial.contains(entry))))
    }

    /// Recheck the product selected by whole-work placement immediately before
    /// arguments enter either executor. A placement receipt never pins code or
    /// its source dependencies by itself.
    pub(crate) fn ready_entry_is_current(
        &mut self,
        unit: &ExecutableUnit,
        preferred_function: Option<&str>,
        selected: &PublishedEntry,
    ) -> Result<bool, runmat_runtime::RuntimeError> {
        let Some(key) = ready_entry_key(unit, preferred_function) else {
            return Ok(false);
        };
        let ready = match selected.tier {
            runmat_jit::entry::PublishedTier::Generic => {
                self.entries
                    .resolve_current_borrowed(key.as_str(), &self.dependencies, None)
            }
            runmat_jit::entry::PublishedTier::Specialized { profile } => self
                .entries
                .resolve_current_borrowed(key.as_str(), &self.dependencies, Some(profile)),
        };
        Ok(ready.is_some_and(|ready| {
            ready.publication == selected.publication
                && ready.tier == selected.tier
                && ready.entrypoint == selected.entrypoint
                && ready.dependencies == selected.dependencies
                && Rc::ptr_eq(&ready.executor, &selected.executor)
        }))
    }

    #[cfg(test)]
    pub(crate) fn with_tiering_config(config: runmat_jit::tiering::TieringConfig) -> Self {
        Self {
            tiering: TieringSession::new(config).expect("test tiering config must be valid"),
            ..Self::default()
        }
    }

    pub(crate) fn resolve_or_compile(
        &mut self,
        unit: &ExecutableUnit,
        preferred_function: Option<&str>,
        liveness_admission: runmat_native_codegen::NativeLivenessAdmission,
    ) -> Result<PublishedEntry, runmat_runtime::RuntimeError> {
        let dependency_keys = self.observe_unit_dependencies(unit, preferred_function)?;
        let current = self.dependencies.snapshot_all();
        let key = entry_key(unit, preferred_function)?;
        if let Some(entry) = self.entries.resolve(&key, &current) {
            return Ok(entry);
        }

        let compiled = super::compile::compile_with_policy(
            unit,
            preferred_function,
            self.optimization,
            liveness_admission,
        )?;
        #[cfg(test)]
        {
            self.compilations += 1;
        }
        let snapshot = self.dependencies.snapshot(dependency_keys.iter());
        self.entries
            .publish(
                key,
                compiled.executor,
                compiled.entrypoint,
                snapshot,
                compiled.compilation,
            )
            .map_err(|error| super::error::stage("NativePublication", error))
    }

    pub(crate) fn representation_profile(
        &self,
        arguments: &[runmat_value::Value],
    ) -> Option<RepresentationProfile> {
        self.representation_profile_from_values(arguments.iter())
    }

    pub(crate) fn representation_profile_from_values<'a>(
        &self,
        arguments: impl IntoIterator<Item = &'a runmat_value::Value>,
    ) -> Option<RepresentationProfile> {
        let facts = arguments
            .into_iter()
            .map(runmat_runtime::value_fact::value_fact)
            .collect();
        RepresentationProfile::from_facts(facts, self.tiering.config().max_profile_bytes).ok()
    }

    #[cfg(test)]
    pub(crate) fn resolve_or_schedule(
        &mut self,
        unit: &ExecutableUnit,
        preferred_function: Option<&str>,
        invocation_profile: Option<&RepresentationProfile>,
    ) -> Result<Option<PublishedEntry>, runmat_runtime::RuntimeError> {
        self.resolve_or_schedule_with_evidence(
            unit,
            preferred_function,
            invocation_profile,
            None,
            false,
            runmat_native_codegen::NativeLivenessAdmission::AuthorizedUnbounded,
        )
        .map(|scheduled| scheduled.published)
    }

    pub(crate) fn resolve_or_schedule_with_placement(
        &mut self,
        invocation: super::placement::ReadyInvocation<'_>,
        profile: Option<&RepresentationProfile>,
        runtime: &runmat_runtime::context::RuntimeContext,
        capture_ready: bool,
    ) -> Result<ScheduledInvocation, runmat_runtime::RuntimeError> {
        self.resolve_or_schedule_with_evidence(
            invocation.unit,
            invocation.preferred_function,
            profile,
            Some((runtime, invocation)),
            capture_ready,
            super::compile::capture_liveness_admission(runtime)?,
        )
    }

    fn resolve_or_schedule_with_evidence(
        &mut self,
        unit: &ExecutableUnit,
        preferred_function: Option<&str>,
        invocation_profile: Option<&RepresentationProfile>,
        placement: Option<(
            &runmat_runtime::context::RuntimeContext,
            super::placement::ReadyInvocation<'_>,
        )>,
        capture_ready: bool,
        liveness_admission: runmat_native_codegen::NativeLivenessAdmission,
    ) -> Result<ScheduledInvocation, runmat_runtime::RuntimeError> {
        let dependency_keys = self.observe_unit_dependencies(unit, preferred_function)?;
        let current = self.dependencies.snapshot_all();
        let key = entry_key(unit, preferred_function)?;
        // The mandatory preparation above owns the one canonical fingerprint.
        // Fresh equivalent units have no cached digest before this point. Take
        // the bounded ready snapshot now, before publishing completed work or
        // compiling any product for this invocation.
        let ready_at_start = if capture_ready {
            self.ready_publications(unit, preferred_function, invocation_profile)?
        } else {
            EntryPublications::default()
        };
        let published = self.resolve_prepared_invocation(
            unit,
            preferred_function,
            invocation_profile,
            placement,
            PreparedInvocation {
                dependency_keys,
                current,
                key,
            },
            liveness_admission,
        )?;
        Ok(ScheduledInvocation {
            published,
            ready_at_start,
        })
    }

    fn resolve_prepared_invocation(
        &mut self,
        unit: &ExecutableUnit,
        preferred_function: Option<&str>,
        invocation_profile: Option<&RepresentationProfile>,
        placement: Option<(
            &runmat_runtime::context::RuntimeContext,
            super::placement::ReadyInvocation<'_>,
        )>,
        prepared: PreparedInvocation,
        liveness_admission: runmat_native_codegen::NativeLivenessAdmission,
    ) -> Result<Option<PublishedEntry>, runmat_runtime::RuntimeError> {
        let PreparedInvocation {
            dependency_keys,
            current,
            key,
        } = prepared;
        self.publish_completed(&current)?;
        let site = tier_site(unit, preferred_function)?;
        let availability = self
            .entries
            .availability(&key, &current, self.pending.len());
        let decision = if invocation_profile.is_some() {
            self.tiering.decide(&site, &availability)
        } else {
            self.tiering.decide_generic(&site, &availability)
        };
        diagnostics::decision(
            &site,
            &decision,
            &availability,
            invocation_profile.is_some(),
            &self.tiering,
        );
        match (decision, invocation_profile) {
            (TierDecision::Interpret, _) => Ok(None),
            (TierDecision::ExecuteGeneric, _) => Ok(self.entries.resolve(&key, &current)),
            (TierDecision::ExecuteSpecialized { profile }, Some(invocation_profile))
                if profile == invocation_profile.digest =>
            {
                Ok(self
                    .entries
                    .resolve_specialized(&key, profile, &current)
                    .or_else(|| self.entries.resolve(&key, &current)))
            }
            (TierDecision::ExecuteSpecialized { .. }, _) => {
                Ok(self.entries.resolve(&key, &current))
            }
            (TierDecision::CompileGeneric { mode }, _) => {
                let compilation = CompilationKey {
                    entry: key.clone(),
                    tier: CompilationTier::Generic,
                };
                if self.failed_compilations.contains(&compilation) {
                    diagnostics::rejected(&compilation, "previous_compilation_failure", None);
                    return Ok(None);
                }
                match mode {
                    CompilationMode::DeterministicSynchronous => {
                        let compiled = match super::compile::compile_with_policy(
                            unit,
                            preferred_function,
                            self.optimization,
                            liveness_admission,
                        ) {
                            Ok(compiled) => compiled,
                            Err(error) => {
                                log::warn!("adaptive native compilation failed: {error}");
                                self.failed_compilations.insert(compilation);
                                return Ok(None);
                            }
                        };
                        if !self
                            .code_budget_admits(compiled.executor.retained_code_bytes(), &current)
                        {
                            diagnostics::rejected(
                                &compilation,
                                "code_budget",
                                Some(compiled.executor.retained_code_bytes()),
                            );
                            self.failed_compilations.insert(compilation);
                            return Ok(None);
                        }
                        #[cfg(test)]
                        {
                            self.compilations += 1;
                        }
                        let snapshot = self.dependencies.snapshot(dependency_keys.iter());
                        self.entries
                            .publish(
                                key,
                                compiled.executor,
                                compiled.entrypoint,
                                snapshot,
                                compiled.compilation,
                            )
                            .map(Some)
                            .map_err(|error| super::error::stage("NativePublication", error))
                    }
                    CompilationMode::Background => {
                        if let Err(error) = self.schedule_background(
                            unit,
                            preferred_function,
                            compilation.clone(),
                            None,
                            self.dependencies.snapshot(dependency_keys.iter()),
                            liveness_admission,
                        ) {
                            log::warn!("adaptive native compilation was not scheduled: {error}");
                            self.failed_compilations.insert(compilation);
                        }
                        Ok(None)
                    }
                }
            }
            (TierDecision::CompileSpecialized { profile, mode }, Some(invocation_profile))
                if profile == invocation_profile.digest =>
            {
                if let Some(((runtime, invocation), quote)) =
                    placement.zip(self.entries.preparation_quote(&key, profile, &current))
                {
                    if matches!(super::placement::repreparation_evidence(runtime, invocation, &quote)?,
                        PreparationPaybackEvidence::NonImproving { confidence, .. }
                            if confidence >= runmat_execution::EstimateConfidence::Medium)
                    {
                        // This is a fresh statistics query, not a permanent
                        // failed-compilation mark. Stale/changed evidence
                        // returns to the existing bounded exploration gate.
                        return Ok(self.entries.resolve(&key, &current));
                    }
                }
                let compilation = CompilationKey {
                    entry: key.clone(),
                    tier: CompilationTier::Specialized(profile),
                };
                if self.failed_compilations.contains(&compilation) {
                    return Ok(self.entries.resolve(&key, &current));
                }
                match mode {
                    CompilationMode::DeterministicSynchronous => {
                        self.compile_specialized_now(SpecializedCompilationRequest {
                            unit,
                            preferred_function,
                            profile: invocation_profile,
                            key: &key,
                            current: &current,
                            dependency_keys: &dependency_keys,
                            compilation: &compilation,
                            liveness_admission,
                        })
                    }
                    CompilationMode::Background => {
                        if let Err(error) = self.schedule_background(
                            unit,
                            preferred_function,
                            compilation.clone(),
                            Some(invocation_profile.clone()),
                            self.dependencies.snapshot(dependency_keys.iter()),
                            liveness_admission,
                        ) {
                            log::warn!(
                                "adaptive specialized compilation was not scheduled: {error}"
                            );
                            self.failed_compilations.insert(compilation);
                        }
                        Ok(self.entries.resolve(&key, &current))
                    }
                }
            }
            (TierDecision::CompileSpecialized { .. }, _) => {
                Ok(self.entries.resolve(&key, &current))
            }
        }
    }

    pub(crate) fn observe_invocation(
        &self,
        unit: &ExecutableUnit,
        preferred_function: Option<&str>,
        profile: Option<&RepresentationProfile>,
        elapsed_ns: u64,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        let site = tier_site(unit, preferred_function)?;
        match profile {
            Some(profile) => self.tiering.observe_invocation(site, profile, elapsed_ns),
            None => self.tiering.observe_unprofiled_invocation(site, elapsed_ns),
        }
        .map_err(|error| super::error::stage("NativeTierFeedback", error))
    }

    pub(crate) fn observe_loop_backedges(
        &self,
        unit: &ExecutableUnit,
        preferred_function: Option<&str>,
        backedges: &BTreeMap<runmat_types::ProgramPointId, u64>,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        if backedges.is_empty() {
            return Ok(());
        }
        let function_site = tier_site(unit, preferred_function)?;
        for (header, count) in backedges {
            self.tiering
                .observe_backedge(function_site.clone(), *count)
                .and_then(|_| {
                    self.tiering.observe_backedge(
                        TierSiteId {
                            entry: function_site.entry.clone(),
                            function: header.function,
                            loop_header: Some(*header),
                        },
                        *count,
                    )
                })
                .map_err(|error| super::error::stage("NativeTierFeedback", error))?;
        }
        Ok(())
    }

    pub(crate) fn resolve_or_schedule_osr(
        &mut self,
        unit: &ExecutableUnit,
        preferred_function: Option<&str>,
        profile: Option<&RepresentationProfile>,
        liveness_admission: runmat_native_codegen::NativeLivenessAdmission,
    ) -> Result<Option<(PublishedEntry, runmat_types::ProgramPointId)>, runmat_runtime::RuntimeError>
    {
        let Some(profile) = profile else {
            return Ok(None);
        };
        let dependency_keys = self.observe_unit_dependencies(unit, preferred_function)?;
        let current = self.dependencies.snapshot_all();
        self.publish_completed(&current)?;
        let key = entry_key(unit, preferred_function)?;
        let function_site = tier_site(unit, preferred_function)?;
        let Some(loop_site) = self.tiering.hottest_osr_site(&function_site) else {
            return Ok(None);
        };
        let point = loop_site
            .loop_header
            .expect("OSR policy must return an exact loop site");
        if let Some(published) = self
            .entries
            .resolve_specialized(&key, profile.digest, &current)
        {
            return Ok(Some((published, point)));
        }

        let compilation = CompilationKey {
            entry: key.clone(),
            tier: CompilationTier::Specialized(profile.digest),
        };
        if self.failed_compilations.contains(&compilation)
            || self.pending.contains_key(&compilation)
            || self.pending.len() >= self.tiering.config().max_pending_compilations
        {
            return Ok(None);
        }
        match self.tiering.config().compilation_mode() {
            CompilationMode::DeterministicSynchronous => {
                let published = self.compile_specialized_now(SpecializedCompilationRequest {
                    unit,
                    preferred_function,
                    profile,
                    key: &key,
                    current: &current,
                    dependency_keys: &dependency_keys,
                    compilation: &compilation,
                    liveness_admission,
                })?;
                Ok(published.map(|published| (published, point)))
            }
            CompilationMode::Background => {
                if let Err(error) = self.schedule_background(
                    unit,
                    preferred_function,
                    compilation.clone(),
                    Some(profile.clone()),
                    self.dependencies.snapshot(dependency_keys.iter()),
                    liveness_admission,
                ) {
                    log::warn!("adaptive OSR compilation was not scheduled: {error}");
                    self.failed_compilations.insert(compilation);
                }
                Ok(None)
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn tiering_snapshot(&self) -> runmat_jit::tiering::TierFeedbackSnapshot {
        self.tiering.snapshot()
    }

    fn schedule_background(
        &mut self,
        unit: &ExecutableUnit,
        preferred_function: Option<&str>,
        key: CompilationKey,
        specialization: Option<RepresentationProfile>,
        dependencies: runmat_jit::invalidation::DependencySnapshot,
        liveness_admission: runmat_native_codegen::NativeLivenessAdmission,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        if self.pending.contains_key(&key) {
            return Ok(());
        }
        let prepared = super::compile::prepare_with_policy(
            unit,
            preferred_function,
            specialization,
            self.optimization,
            liveness_admission,
        )?;
        diagnostics::scheduled(&key, &dependencies);
        let (sender, receiver) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("runmat-jit-compile".into())
            .spawn(move || {
                let result =
                    super::compile::compile_prepared(prepared).map_err(|error| error.to_string());
                let _ = sender.send(result);
            })
            .map_err(|error| super::error::stage("NativeBackgroundCompile", error))?;
        self.pending.insert(
            key,
            PendingCompilation {
                dependencies,
                receiver,
            },
        );
        Ok(())
    }

    fn compile_specialized_now(
        &mut self,
        request: SpecializedCompilationRequest<'_>,
    ) -> Result<Option<PublishedEntry>, runmat_runtime::RuntimeError> {
        let compiled = super::compile::compile_prepared(super::compile::prepare_with_policy(
            request.unit,
            request.preferred_function,
            Some(request.profile.clone()),
            self.optimization,
            request.liveness_admission,
        )?)?;
        if self
            .entries
            .make_room_for_specialized(
                request.key,
                request.current,
                compiled.executor.retained_code_bytes(),
                self.tiering.config().max_versions_per_entry,
                self.tiering.config().max_code_bytes,
            )
            .is_err()
        {
            self.failed_compilations.insert(request.compilation.clone());
            return Ok(None);
        }
        #[cfg(test)]
        {
            self.compilations += 1;
        }
        let snapshot = self.dependencies.snapshot(request.dependency_keys.iter());
        self.entries
            .publish_specialized(
                request.key.clone(),
                request.profile.digest,
                Rc::new(compiled.executor),
                compiled.entrypoint,
                snapshot,
                compiled.compilation,
            )
            .map(Some)
            .map_err(|error| super::error::stage("NativePublication", error))
    }

    fn publish_completed(
        &mut self,
        current: &runmat_jit::invalidation::DependencySnapshot,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        let completed = self
            .pending
            .iter()
            .filter_map(|(key, pending)| match pending.receiver.try_recv() {
                Ok(result) => Some((key.clone(), Some(result))),
                Err(TryRecvError::Disconnected) => Some((key.clone(), None)),
                Err(TryRecvError::Empty) => None,
            })
            .collect::<Vec<_>>();
        for (key, result) in completed {
            let pending = self
                .pending
                .remove(&key)
                .expect("completed native compilation must remain pending");
            let Some(compiled) = diagnostics::completed(&key, result) else {
                self.failed_compilations.insert(key);
                continue;
            };
            if !pending.dependencies.is_satisfied_by(current) {
                diagnostics::stale(&key, &pending.dependencies, current);
                continue;
            }
            let admitted = match key.tier {
                CompilationTier::Generic => {
                    self.code_budget_admits(compiled.executor.retained_code_bytes(), current)
                }
                CompilationTier::Specialized(_) => self
                    .entries
                    .make_room_for_specialized(
                        &key.entry,
                        current,
                        compiled.executor.retained_code_bytes(),
                        self.tiering.config().max_versions_per_entry,
                        self.tiering.config().max_code_bytes,
                    )
                    .is_ok(),
            };
            if !admitted {
                diagnostics::rejected(
                    &key,
                    "code_or_version_budget",
                    Some(compiled.executor.retained_code_bytes()),
                );
                self.failed_compilations.insert(key);
                continue;
            }
            #[cfg(test)]
            {
                self.compilations += 1;
            }
            diagnostics::publishing(&key, compiled.executor.retained_code_bytes());
            let published = match key.tier {
                CompilationTier::Generic => self.entries.publish(
                    key.entry,
                    Rc::new(compiled.executor),
                    compiled.entrypoint,
                    pending.dependencies,
                    compiled.compilation,
                ),
                CompilationTier::Specialized(profile) => self.entries.publish_specialized(
                    key.entry,
                    profile,
                    Rc::new(compiled.executor),
                    compiled.entrypoint,
                    pending.dependencies,
                    compiled.compilation,
                ),
            };
            published.map_err(|error| super::error::stage("NativePublication", error))?;
        }
        Ok(())
    }

    fn code_budget_admits(
        &self,
        additional_bytes: u64,
        current: &runmat_jit::invalidation::DependencySnapshot,
    ) -> bool {
        self.entries
            .retained_code_bytes(current)
            .checked_add(additional_bytes)
            .is_some_and(|total| total <= self.tiering.config().max_code_bytes)
    }

    pub(crate) fn publish_session_registry(
        &mut self,
        previous: &runmat_vm::FunctionRegistry,
        current: &runmat_vm::FunctionRegistry,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        let mut names = previous
            .names
            .keys()
            .chain(current.names.keys())
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        let mut changed = false;
        for name in names {
            let before = previous.resolve_name(name);
            let after = current.resolve_name(name);
            let before_revision = function_revision(previous, before)?;
            let after_revision = function_revision(current, after)?;
            if before_revision == after_revision {
                continue;
            }
            changed = true;
            diagnostics::dependency_changed(&DependencyKey::SessionFunction(name.clone()));
            self.observe(DependencyKey::SessionFunction(name.clone()), after_revision)?;
        }
        if changed {
            diagnostics::dependency_changed(&DependencyKey::SessionCatalog);
            self.failed_compilations.clear();
            self.observe(
                DependencyKey::SessionCatalog,
                session_catalog_revision(current)?,
            )?;
            let current = self.dependencies.snapshot_all();
            self.entries.invalidate(&current);
        }
        Ok(())
    }

    pub(crate) fn prepare_project_revision(
        &self,
        revision: Option<&runmat_package::ProjectRevision>,
    ) -> Result<PreparedProjectRevision, runmat_runtime::RuntimeError> {
        let namespace = revision
            .map(runmat_package::ProjectRevision::cache_namespace)
            .unwrap_or_else(|| "loose".to_string());
        let observation = self
            .dependencies
            .prepare_observation(DependencyKey::Program("session-project".into()), namespace)
            .map_err(|error| super::error::stage("NativeDependency", error))?;
        Ok(PreparedProjectRevision {
            baseline_namespace: self.project_scope_revision.clone(),
            observation,
        })
    }

    fn observe_unit_dependencies(
        &mut self,
        unit: &ExecutableUnit,
        preferred_function: Option<&str>,
    ) -> Result<BTreeSet<DependencyKey>, runmat_runtime::RuntimeError> {
        let revision = &unit.revision().program_revision;
        let program_key = DependencyKey::Program(revision.graph_digest().to_string());
        let project_scope_key = DependencyKey::Program("session-project".into());
        let catalog_key = DependencyKey::Catalog("builtins".to_string());
        self.observe(program_key.clone(), revision.canonical_identity())?;
        self.observe(
            project_scope_key.clone(),
            self.project_scope_revision.clone(),
        )?;
        self.observe(
            catalog_key.clone(),
            revision.catalog_fingerprint().to_string(),
        )?;
        let mut keys = BTreeSet::from([program_key, project_scope_key, catalog_key]);

        let roots = BTreeSet::from([unit.native_entrypoint_function(preferred_function)?]);
        let selected = runmat_mir::analysis::selected_function_closure(unit.mir(), &roots);
        let mut referenced = super::dependencies::analyze(unit.mir(), &selected);
        if !referenced.super_callers.is_empty() {
            let registry = unit.bytecode().function_registry();
            for caller in &referenced.super_callers {
                for target in registry
                    .retained_super_constructor_targets_for(*caller)
                    .map_err(|error| super::error::stage("NativeSuperConstructorOwner", error))?
                {
                    referenced.functions.insert(target.function);
                }
            }
        }
        for function in referenced.functions {
            let mut names = unit
                .functions()
                .names
                .iter()
                .filter_map(|(name, candidate)| (*candidate == function).then_some(name))
                .collect::<Vec<_>>();
            names.sort();
            for name in names {
                let key = DependencyKey::SessionFunction(name.clone());
                if self.dependencies.generation(&key).is_some() {
                    keys.insert(key);
                }
            }
        }
        if referenced.dynamic_catalog {
            keys.insert(DependencyKey::SessionCatalog);
        }
        Ok(keys)
    }

    fn observe(
        &mut self,
        key: DependencyKey,
        revision: String,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        self.dependencies
            .observe(key, revision)
            .map(|_| ())
            .map_err(|error| super::error::stage("NativeDependency", error))
    }

    #[cfg(test)]
    pub(crate) fn published_entry_count(&self) -> usize {
        self.entries.published_entry_count()
    }

    #[cfg(test)]
    pub(crate) fn compilation_count(&self) -> usize {
        self.compilations
    }

    #[cfg(test)]
    pub(crate) fn specialized_version_count(&self) -> usize {
        self.entries
            .specialized_version_count(&self.dependencies.snapshot_all())
    }

    #[cfg(test)]
    pub(crate) fn optimized_region_plan_count(
        &self,
        unit: &ExecutableUnit,
        preferred_function: Option<&str>,
        profile: &RepresentationProfile,
    ) -> usize {
        super::compile::prepare_with_policy(
            unit,
            preferred_function,
            Some(profile.clone()),
            self.optimization,
            runmat_native_codegen::NativeLivenessAdmission::AuthorizedUnbounded,
        )
        .and_then(super::compile::compile_prepared)
        .map(|compiled| compiled.executor.optimized_region_count())
        .unwrap_or(0)
    }
}

fn tier_site(
    unit: &ExecutableUnit,
    preferred_function: Option<&str>,
) -> Result<TierSiteId, runmat_runtime::RuntimeError> {
    let function = unit.native_entrypoint_function(preferred_function)?;
    let function = unit
        .analysis()
        .function_for_local(function)
        .ok_or_else(|| {
            super::error::stage(
                "NativeTierIdentity",
                "entry has no authenticated portable identity",
            )
        })?
        .function;
    Ok(TierSiteId {
        entry: authored_entry_key(unit, preferred_function).0,
        function,
        loop_header: None,
    })
}

fn authored_entry_key(unit: &ExecutableUnit, preferred_function: Option<&str>) -> EntryKey {
    EntryKey(format!(
        "{}|{}|{}",
        unit.revision().program_revision.canonical_identity(),
        unit.revision().source_digest,
        preferred_function.unwrap_or("<entrypoint>")
    ))
}

fn entry_key(
    unit: &ExecutableUnit,
    preferred_function: Option<&str>,
) -> Result<EntryKey, runmat_runtime::RuntimeError> {
    let digest = unit
        .native_compilation_digest()
        .map_err(|error| super::error::stage("NativeCompilationIdentity", error))?;
    // Hotness belongs to authored code; published machine code additionally
    // depends on its exact semantic ABI and analysis inputs.
    Ok(EntryKey(format!(
        "{}|{}",
        authored_entry_key(unit, preferred_function).0,
        digest
    )))
}

/// A ready lookup cannot create a fresh compilation fingerprint or allocate
/// the registry key. The fixed request envelope declines oversized names.
struct ReadyEntryKey {
    bytes: [u8; 1024],
    length: usize,
}
impl std::fmt::Write for ReadyEntryKey {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        let end = self.length.checked_add(text.len()).ok_or(std::fmt::Error)?;
        let target = self
            .bytes
            .get_mut(self.length..end)
            .ok_or(std::fmt::Error)?;
        target.copy_from_slice(text.as_bytes());
        self.length = end;
        Ok(())
    }
}
impl ReadyEntryKey {
    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.length]).expect("formatted UTF8 key")
    }
}
fn ready_entry_key(
    unit: &ExecutableUnit,
    preferred_function: Option<&str>,
) -> Option<ReadyEntryKey> {
    use std::fmt::Write;
    let digest = unit.cached_native_compilation_digest()?;
    let revision = unit.revision().program_revision.identity_digest().ok()?;
    let mut key = ReadyEntryKey {
        bytes: [0; 1024],
        length: 0,
    };
    write!(
        &mut key,
        "{}|{}|{}|{}",
        revision,
        unit.revision().source_digest,
        preferred_function.unwrap_or("<entrypoint>"),
        digest
    )
    .ok()?;
    Some(key)
}

fn function_revision(
    registry: &runmat_vm::FunctionRegistry,
    function: Option<FunctionId>,
) -> Result<String, runmat_runtime::RuntimeError> {
    let Some(function) = function.and_then(|function| registry.get(function)) else {
        return Ok("removed".to_string());
    };
    crate::session::semantic_definition::fingerprint(function)
        .map_err(|error| super::error::stage("NativeDependencyRevision", error))
}

fn session_catalog_revision(
    registry: &runmat_vm::FunctionRegistry,
) -> Result<String, runmat_runtime::RuntimeError> {
    let mut names = registry.names.iter().collect::<Vec<_>>();
    names.sort_by(|left, right| left.0.cmp(right.0));
    let payload = names
        .into_iter()
        .map(|(name, function)| {
            function_revision(registry, Some(*function))
                .map(|revision| format!("{name}={revision}"))
        })
        .collect::<Result<Vec<_>, _>>()?
        .join("\0");
    Ok(format!(
        "catalog:{}",
        runmat_execution::Digest::sha256(payload.as_bytes())
    ))
}

#[cfg(test)]
mod revision_tests {
    use super::*;

    fn registry_with_order(reverse: bool) -> runmat_vm::FunctionRegistry {
        let function_id = FunctionId(7);
        let mut function = runmat_vm::FunctionBytecode {
            function: function_id,
            display_name: "revisionTarget".into(),
            instructions: vec![
                runmat_vm::Instr::LoadConst(1.0),
                runmat_vm::Instr::ReturnValue,
            ],
            ..runmat_vm::FunctionBytecode::default()
        };
        let names = if reverse {
            [(2, "second"), (1, "first")]
        } else {
            [(1, "first"), (2, "second")]
        };
        for (slot, name) in names {
            function.var_names.insert(slot, name.into());
            function.initially_unassigned_slots.insert(slot);
        }
        runmat_vm::FunctionRegistry::new(std::collections::HashMap::from([(function_id, function)]))
    }

    #[test]
    fn function_revisions_are_canonical_and_body_sensitive() {
        let first = registry_with_order(false);
        let mut reordered = registry_with_order(true);
        assert_eq!(
            session_catalog_revision(&first).unwrap(),
            session_catalog_revision(&reordered).unwrap()
        );

        reordered
            .functions
            .get_mut(&FunctionId(7))
            .unwrap()
            .instructions[0] = runmat_vm::Instr::LoadConst(2.0);
        assert_ne!(
            session_catalog_revision(&first).unwrap(),
            session_catalog_revision(&reordered).unwrap()
        );

        let before_owner = session_catalog_revision(&first).unwrap();
        let function = reordered.functions.get_mut(&FunctionId(7)).unwrap();
        function.instructions[0] = runmat_vm::Instr::LoadConst(1.0);
        function.class_method_owner = Some(runmat_types::ClassMethodOwner {
            declaring_class: runmat_types::ClassIdentity::new("RevisionOwner").unwrap(),
            method: runmat_types::MethodName::from("revisionTarget"),
            is_static: false,
            role: runmat_types::ClassMethodRole::Ordinary,
        });
        assert_ne!(
            before_owner,
            session_catalog_revision(&reordered).unwrap(),
            "declaration owner identity participates in the executable function revision"
        );
    }
}
