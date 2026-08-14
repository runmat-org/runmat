use std::collections::BTreeSet;

use runmat_hir::FunctionId;
use runmat_jit::{
    entry::{EntryKey, EntryRegistry, PublishedEntry},
    invalidation::{DependencyKey, DependencyTracker},
};

use crate::ExecutableUnit;

/// Core-owned publication boundary for generic-native products in one session.
/// JIT owns cells and dependency matching; Core supplies authoritative program,
/// catalog, and session-function revisions.
pub(crate) struct GenericNativeCache {
    entries: EntryRegistry,
    dependencies: DependencyTracker,
    project_scope_revision: String,
    #[cfg(test)]
    compilations: usize,
}

impl Default for GenericNativeCache {
    fn default() -> Self {
        Self {
            entries: EntryRegistry::default(),
            dependencies: DependencyTracker::default(),
            project_scope_revision: "loose".to_string(),
            #[cfg(test)]
            compilations: 0,
        }
    }
}

impl GenericNativeCache {
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
