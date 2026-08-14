use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver, TryRecvError};

use runmat_execution::Digest;
use runmat_hir::FunctionId;
use runmat_jit::{
    entry::{EntryKey, EntryRegistry, PublishedEntry},
    invalidation::{DependencyKey, DependencyTracker},
    tiering::{CompilationMode, RepresentationProfile, TierDecision, TierSiteId, TieringSession},
};

use crate::ExecutableUnit;

/// Core-owned publication boundary for generic-native products in one session.
/// JIT owns cells and dependency matching; Core supplies authoritative program,
/// catalog, and session-function revisions.
pub(crate) struct GenericNativeCache {
    entries: EntryRegistry,
    dependencies: DependencyTracker,
    project_scope_revision: String,
    tiering: TieringSession,
    pending: BTreeMap<CompilationKey, PendingCompilation>,
    failed_compilations: BTreeSet<CompilationKey>,
    #[cfg(test)]
    compilations: usize,
}

struct PendingCompilation {
    dependencies: runmat_jit::invalidation::DependencySnapshot,
    receiver: Receiver<Result<super::compile::BackgroundCompiledGenericUnit, String>>,
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
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum CompilationTier {
    Generic,
    Specialized(Digest),
}

impl Default for GenericNativeCache {
    fn default() -> Self {
        Self {
            entries: EntryRegistry::default(),
            dependencies: DependencyTracker::default(),
            project_scope_revision: "loose".to_string(),
            tiering: TieringSession::default(),
            pending: BTreeMap::new(),
            failed_compilations: BTreeSet::new(),
            #[cfg(test)]
            compilations: 0,
        }
    }
}

impl GenericNativeCache {
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
    ) -> Result<PublishedEntry, runmat_runtime::RuntimeError> {
        let dependency_keys = self.observe_unit_dependencies(unit)?;
        let current = self.dependencies.snapshot_all();
        let key = entry_key(unit, preferred_function);
        if let Some(entry) = self.entries.resolve(&key, &current) {
            return Ok(entry);
        }

        let compiled = super::compile::compile(unit, preferred_function)?;
        #[cfg(test)]
        {
            self.compilations += 1;
        }
        let snapshot = self.dependencies.snapshot(dependency_keys.iter());
        self.entries
            .publish(key, compiled.executor, compiled.entrypoint, snapshot)
            .map_err(|error| super::error::stage("NativePublication", error))
    }

    pub(crate) fn representation_profile(
        &self,
        arguments: &[runmat_value::Value],
    ) -> Option<RepresentationProfile> {
        let facts = arguments
            .iter()
            .map(runmat_runtime::value_fact::value_fact)
            .collect();
        RepresentationProfile::from_facts(facts, self.tiering.config().max_profile_bytes).ok()
    }

    pub(crate) fn resolve_or_schedule(
        &mut self,
        unit: &ExecutableUnit,
        preferred_function: Option<&str>,
        invocation_profile: &RepresentationProfile,
    ) -> Result<Option<PublishedEntry>, runmat_runtime::RuntimeError> {
        let dependency_keys = self.observe_unit_dependencies(unit)?;
        let current = self.dependencies.snapshot_all();
        let key = entry_key(unit, preferred_function);
        self.publish_completed(&current)?;
        let site = tier_site(unit, preferred_function, &key)?;
        let availability = self
            .entries
            .availability(&key, &current, self.pending.len());
        match self.tiering.decide(&site, &availability) {
            TierDecision::Interpret => Ok(None),
            TierDecision::ExecuteGeneric => Ok(self.entries.resolve(&key, &current)),
            TierDecision::ExecuteSpecialized { profile }
                if profile == invocation_profile.digest =>
            {
                Ok(self
                    .entries
                    .resolve_specialized(&key, profile, &current)
                    .or_else(|| self.entries.resolve(&key, &current)))
            }
            TierDecision::ExecuteSpecialized { .. } => Ok(self.entries.resolve(&key, &current)),
            TierDecision::CompileGeneric { mode } => {
                let compilation = CompilationKey {
                    entry: key.clone(),
                    tier: CompilationTier::Generic,
                };
                if self.failed_compilations.contains(&compilation) {
                    return Ok(None);
                }
                match mode {
                    CompilationMode::DeterministicSynchronous => {
                        let compiled = match super::compile::compile(unit, preferred_function) {
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
                            self.failed_compilations.insert(compilation);
                            return Ok(None);
                        }
                        #[cfg(test)]
                        {
                            self.compilations += 1;
                        }
                        let snapshot = self.dependencies.snapshot(dependency_keys.iter());
                        self.entries
                            .publish(key, compiled.executor, compiled.entrypoint, snapshot)
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
                        ) {
                            log::warn!("adaptive native compilation was not scheduled: {error}");
                            self.failed_compilations.insert(compilation);
                        }
                        Ok(None)
                    }
                }
            }
            TierDecision::CompileSpecialized { profile, mode }
                if profile == invocation_profile.digest =>
            {
                let compilation = CompilationKey {
                    entry: key.clone(),
                    tier: CompilationTier::Specialized(profile),
                };
                if self.failed_compilations.contains(&compilation) {
                    return Ok(self.entries.resolve(&key, &current));
                }
                let compile = |this: &mut Self| {
                    this.compile_specialized_now(SpecializedCompilationRequest {
                        unit,
                        preferred_function,
                        profile: invocation_profile,
                        key: &key,
                        current: &current,
                        dependency_keys: &dependency_keys,
                        compilation: &compilation,
                    })
                };
                match mode {
                    CompilationMode::DeterministicSynchronous => compile(self),
                    CompilationMode::Background => {
                        if let Err(error) = self.schedule_background(
                            unit,
                            preferred_function,
                            compilation.clone(),
                            Some(invocation_profile.clone()),
                            self.dependencies.snapshot(dependency_keys.iter()),
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
            TierDecision::CompileSpecialized { .. } => Ok(self.entries.resolve(&key, &current)),
        }
    }

    pub(crate) fn observe_invocation(
        &self,
        unit: &ExecutableUnit,
        preferred_function: Option<&str>,
        profile: &RepresentationProfile,
        elapsed_ns: u64,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        let key = entry_key(unit, preferred_function);
        let site = tier_site(unit, preferred_function, &key)?;
        self.tiering
            .observe_invocation(site, profile, elapsed_ns)
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
        let key = entry_key(unit, preferred_function);
        let function_site = tier_site(unit, preferred_function, &key)?;
        for (header, count) in backedges {
            self.tiering
                .observe_backedge(function_site.clone(), *count)
                .and_then(|_| {
                    self.tiering.observe_backedge(
                        TierSiteId {
                            entry: key.0.clone(),
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
        profile: &RepresentationProfile,
    ) -> Result<Option<(PublishedEntry, runmat_types::ProgramPointId)>, runmat_runtime::RuntimeError>
    {
        let dependency_keys = self.observe_unit_dependencies(unit)?;
        let current = self.dependencies.snapshot_all();
        self.publish_completed(&current)?;
        let key = entry_key(unit, preferred_function);
        let function_site = tier_site(unit, preferred_function, &key)?;
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
    ) -> Result<(), runmat_runtime::RuntimeError> {
        if self.pending.contains_key(&key) {
            return Ok(());
        }
        let prepared = super::compile::prepare(unit, preferred_function, specialization)?;
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
        let compiled = super::compile::compile_prepared(super::compile::prepare(
            request.unit,
            request.preferred_function,
            Some(request.profile.clone()),
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
            let Some(Ok(compiled)) = result else {
                self.failed_compilations.insert(key);
                continue;
            };
            if !pending.dependencies.is_satisfied_by(current) {
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
                self.failed_compilations.insert(key);
                continue;
            }
            #[cfg(test)]
            {
                self.compilations += 1;
            }
            let published = match key.tier {
                CompilationTier::Generic => self.entries.publish(
                    key.entry,
                    Rc::new(compiled.executor),
                    compiled.entrypoint,
                    pending.dependencies,
                ),
                CompilationTier::Specialized(profile) => self.entries.publish_specialized(
                    key.entry,
                    profile,
                    Rc::new(compiled.executor),
                    compiled.entrypoint,
                    pending.dependencies,
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
            if before == after {
                continue;
            }
            changed = true;
            self.observe(
                DependencyKey::SessionFunction(name.clone()),
                function_revision(after),
            )?;
        }
        if changed {
            self.failed_compilations.clear();
            self.observe(
                DependencyKey::SessionCatalog,
                session_catalog_revision(current),
            )?;
            let current = self.dependencies.snapshot_all();
            self.entries.invalidate(&current);
        }
        Ok(())
    }

    pub(crate) fn publish_project_revision(
        &mut self,
        revision: Option<&runmat_package::ProjectRevision>,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        let revision = revision
            .map(runmat_package::ProjectRevision::cache_namespace)
            .unwrap_or_else(|| "loose".to_string());
        if revision == self.project_scope_revision {
            return Ok(());
        }
        self.project_scope_revision = revision.clone();
        self.failed_compilations.clear();
        self.observe(DependencyKey::Program("session-project".into()), revision)?;
        let current = self.dependencies.snapshot_all();
        self.entries.invalidate(&current);
        Ok(())
    }

    fn observe_unit_dependencies(
        &mut self,
        unit: &ExecutableUnit,
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

        let referenced = super::dependencies::analyze(unit.mir());
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
                self.observe(key.clone(), function_revision(Some(function)))?;
                keys.insert(key);
            }
        }
        if referenced.dynamic_catalog {
            self.observe(
                DependencyKey::SessionCatalog,
                session_catalog_revision(unit.functions()),
            )?;
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
}

fn tier_site(
    unit: &ExecutableUnit,
    preferred_function: Option<&str>,
    key: &EntryKey,
) -> Result<TierSiteId, runmat_runtime::RuntimeError> {
    let function = preferred_function
        .and_then(|name| unit.native_function_id(name))
        .or_else(|| unit.mir().entrypoints.first().copied())
        .ok_or_else(|| super::error::stage("NativeTierIdentity", "entry function is missing"))?;
    let function = u32::try_from(function.0)
        .map(runmat_types::ProgramFunctionId)
        .map_err(|_| super::error::stage("NativeTierIdentity", "function identity exceeds u32"))?;
    Ok(TierSiteId {
        entry: key.0.clone(),
        function,
        loop_header: None,
    })
}

fn entry_key(unit: &ExecutableUnit, preferred_function: Option<&str>) -> EntryKey {
    EntryKey(format!(
        "{}|{}|{}",
        unit.revision().program_revision.canonical_identity(),
        unit.revision().source_digest,
        preferred_function.unwrap_or("<entrypoint>")
    ))
}

fn function_revision(function: Option<FunctionId>) -> String {
    function
        .map(|function| format!("function:{}", function.0))
        .unwrap_or_else(|| "removed".to_string())
}

fn session_catalog_revision(registry: &runmat_vm::FunctionRegistry) -> String {
    let mut names = registry.names.iter().collect::<Vec<_>>();
    names.sort_by(|left, right| left.0.cmp(right.0));
    names
        .into_iter()
        .map(|(name, function)| format!("{}={}", name, function.0))
        .collect::<Vec<_>>()
        .join("\0")
}
