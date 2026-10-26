//! Mutable, fully decorated compilation product before immutable publication.

use std::collections::HashMap;

use runmat_hir::FunctionId;

use super::*;

/// Region, parallel, and coverage decoration runs exactly once here. A caller
/// may still prove and rebind its function identities before sealing the unit.
pub(crate) struct ExecutableUnitDraft {
    source: ExecutableSource,
    revision: ExecutableRevision,
    source_map: ExecutableSourceMap,
    coverage: CoveragePlan,
    mir: runmat_mir::MirAssembly,
    analysis: runmat_mir::analysis::AnalysisStore,
    bytecode: runmat_vm::Bytecode,
}

impl ExecutableUnitDraft {
    #[tracing::instrument(
        name = "runtime.executable.prepare_draft", skip_all,
        fields(instructions = bytecode.instructions.len(), functions = bytecode.function_registry.functions.len())
    )]
    pub(crate) fn prepare(
        source: ExecutableSource,
        revision: ExecutableRevision,
        source_map: ExecutableSourceMap,
        mir: runmat_mir::MirAssembly,
        mut analysis: runmat_mir::analysis::AnalysisStore,
        mut bytecode: runmat_vm::Bytecode,
    ) -> Result<Self, String> {
        source_map.validate_content_revisions()?;
        mir.validate_class_sources(|source| {
            usize::try_from(source.0)
                .ok()
                .and_then(|ordinal| source_map.entries().get(ordinal))
                .is_some_and(|entry| entry.source_id == source)
        })?;
        analysis.validate_class_sources(&mir)?;
        let source_count = source_map.entries().len();
        let outside = |source: runmat_vm::BytecodeSourceRef| {
            source
                .program()
                .and_then(|ordinal| usize::try_from(ordinal.0).ok())
                .is_none_or(|ordinal| ordinal >= source_count)
        };
        if bytecode.source_id.is_some_and(outside)
            || bytecode
                .function_registry
                .functions
                .values()
                .any(|body| body.source_id.is_some_and(outside))
            || bytecode
                .bound_functions
                .values()
                .any(|body| body.source_id.is_some_and(outside))
            || mir
                .functions
                .values()
                .any(|metadata| metadata.source.0 as usize >= source_count)
        {
            return Err("executable references a source ordinal outside its source map".into());
        }
        let current_program = revision
            .program_revision
            .identity_digest()
            .map_err(|error| error.to_string())?;
        let compiled_functions = compiled_function_owners(current_program, &mir, &bytecode)?;
        let analysis_owners = compiled_functions.analysis_owners(&analysis, &bytecode)?;
        analysis = analysis.project_product_functions(&analysis_owners)?;
        let region_contracts = analysis
            .regions
            .iter()
            .map(|region| region.contract.clone())
            .collect::<Vec<_>>();
        bytecode.install_regions_for_product(
            &region_contracts,
            &analysis_owners,
            &compiled_functions.registered_bodies,
        )?;
        bytecode.install_parallel_regions_for_product(
            &mir,
            &analysis.parallel,
            &analysis_owners,
            &compiled_functions.registered_bodies,
        )?;
        let coverage = CoveragePlan::instrument_for_product(
            &source,
            &revision,
            &source_map,
            &mut bytecode,
            &compiled_functions.registered_bodies,
        );
        if bytecode.layout.is_none() {
            return Err("canonical bytecode is missing its VM assembly layout".to_string());
        }
        Ok(Self {
            source,
            revision,
            source_map,
            coverage,
            mir,
            analysis,
            bytecode,
        })
    }

    #[tracing::instrument(name = "runtime.executable.seal", skip_all)]
    pub(crate) fn seal(self) -> ExecutableUnit {
        let layout = self
            .bytecode
            .layout
            .clone()
            .expect("prepared draft retains a VM assembly layout");
        let functions = Rc::new(self.bytecode.function_registry.clone());
        ExecutableUnit {
            source: Rc::new(self.source),
            revision: self.revision,
            source_map: Rc::new(self.source_map),
            coverage: Rc::new(self.coverage),
            mir: Rc::new(self.mir),
            analysis: Arc::new(self.analysis),
            bytecode: Rc::new(self.bytecode),
            layout: Rc::new(layout),
            functions,
            portable_selection: None,
            direct_code_cache: Default::default(),
            original_dependencies: Default::default(),
            imported_contract: None,
            #[cfg(not(target_arch = "wasm32"))]
            native_compilation_digest: Rc::new(std::cell::OnceCell::new()),
        }
    }

    pub(crate) fn bytecode(&self) -> &runmat_vm::Bytecode {
        &self.bytecode
    }

    /// Use the same typed MIR reachability rule as the sealed unit before
    /// foreign preparation, while this decorated draft is still mutable.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn inferred_interop_manifest(&self) -> runmat_types::InteropManifest {
        let sources = self
            .source_map
            .entries()
            .iter()
            .map(|entry| (entry.source_id, entry.relative_path.clone()))
            .collect();
        let bindings = self
            .bytecode
            .layout
            .as_ref()
            .expect("prepared draft retains a VM assembly layout")
            .storage_bindings
            .iter()
            .map(|(binding, metadata)| (*binding, metadata.name.clone()))
            .collect();
        let names = runmat_mir::analysis::ReachabilityNames { sources, bindings };
        let (report, _) =
            runmat_mir::analysis::analyze_reachability_with_functions(&self.mir, &names);
        ExecutableUnit::inferred_interop_manifest_for_report(&report)
    }

    /// Rebind a proved current group on clones so a failed VM/MIR/layout
    /// transform cannot partially change the draft or its derived analysis.
    pub(crate) fn rebind_current_definitions(
        &mut self,
        published: &runmat_vm::FunctionRegistry,
        remap: &HashMap<FunctionId, FunctionId>,
    ) -> Result<(), String> {
        if remap.is_empty() {
            return Ok(());
        }
        let mut mir = self.mir.clone();
        let mut bytecode = self.bytecode.clone();
        runmat_mir::remap_closed_function_ids(&mut mir, remap)?;
        runmat_vm::remap_decorated_session_definitions(&mut bytecode, published, remap)?;
        let analysis = self.analysis.rebind_closed_functions(remap)?;
        let current_program = self
            .revision
            .program_revision
            .identity_digest()
            .map_err(|error| error.to_string())?;
        let owners = compiled_function_owners(current_program, &mir, &bytecode)?;
        let analysis =
            analysis.project_product_functions(&owners.analysis_owners(&analysis, &bytecode)?)?;
        self.mir = mir;
        self.bytecode = bytecode;
        self.analysis = analysis;
        Ok(())
    }
}
