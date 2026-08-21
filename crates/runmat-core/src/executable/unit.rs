use std::rc::Rc;

use super::{CoveragePlan, ExecutableRevision, ExecutableSource, ExecutableSourceMap};

/// Immutable canonical compilation product reusable across invocations.
#[derive(Clone, Debug)]
pub struct ExecutableUnit {
    source: ExecutableSource,
    revision: ExecutableRevision,
    source_map: ExecutableSourceMap,
    coverage: CoveragePlan,
    mir: Rc<runmat_mir::MirAssembly>,
    analysis: Rc<runmat_mir::analysis::AnalysisStore>,
    bytecode: Rc<runmat_vm::Bytecode>,
    layout: Rc<runmat_vm::VmAssemblyLayout>,
    functions: Rc<runmat_vm::FunctionRegistry>,
}

impl ExecutableUnit {
    pub(crate) fn new(
        source: ExecutableSource,
        revision: ExecutableRevision,
        source_map: ExecutableSourceMap,
        mir: runmat_mir::MirAssembly,
        analysis: runmat_mir::analysis::AnalysisStore,
        mut bytecode: runmat_vm::Bytecode,
    ) -> Result<Self, String> {
        let region_contracts = analysis
            .regions
            .iter()
            .map(|region| region.contract.clone())
            .collect::<Vec<_>>();
        bytecode.install_regions(&region_contracts)?;
        let coverage = CoveragePlan::instrument(&source, &revision, &source_map, &mut bytecode);
        let layout = bytecode
            .layout
            .clone()
            .ok_or_else(|| "canonical bytecode is missing its VM assembly layout".to_string())?;
        let functions = Rc::new(bytecode.function_registry.clone());
        Ok(Self {
            source,
            revision,
            source_map,
            coverage,
            mir: Rc::new(mir),
            analysis: Rc::new(analysis),
            bytecode: Rc::new(bytecode),
            layout: Rc::new(layout),
            functions,
        })
    }

    pub fn source(&self) -> &ExecutableSource {
        &self.source
    }

    pub fn revision(&self) -> &ExecutableRevision {
        &self.revision
    }

    pub fn source_map(&self) -> &ExecutableSourceMap {
        &self.source_map
    }

    pub fn coverage_plan(&self) -> &CoveragePlan {
        &self.coverage
    }

    /// Exact immutable semantic facts produced for this executable revision.
    pub fn analysis(&self) -> &runmat_mir::analysis::AnalysisStore {
        &self.analysis
    }

    /// Canonical analyzed MIR retained for native lowering and artifact production.
    pub fn mir(&self) -> &runmat_mir::MirAssembly {
        &self.mir
    }

    /// Deterministic whole-program retention facts for link planning and
    /// explanations. Human-readable source and binding names are supplied by
    /// this immutable executable rather than reconstructed by downstream
    /// artifact producers.
    pub fn reachability_report(&self) -> runmat_mir::analysis::ReachabilityReport {
        let sources = self
            .source_map
            .entries()
            .iter()
            .filter_map(|entry| {
                u32::try_from(entry.source_id).ok().map(|source_id| {
                    (
                        runmat_types::ProgramSourceId(source_id),
                        entry.relative_path.clone(),
                    )
                })
            })
            .collect();
        let names = runmat_mir::analysis::ReachabilityNames {
            sources,
            bindings: self.binding_names(),
        };
        runmat_mir::analysis::analyze_reachability(&self.mir, &names)
    }

    /// Exact VM layout shared by interpreter state materialization and native frames.
    pub fn vm_layout(&self) -> &runmat_vm::VmAssemblyLayout {
        &self.layout
    }

    /// Canonical HIR-derived names for every semantic binding retained by the
    /// portable executable product.
    pub fn binding_names(&self) -> std::collections::BTreeMap<runmat_types::BindingId, String> {
        self.layout
            .storage_bindings
            .iter()
            .map(|(binding, metadata)| (*binding, metadata.name.clone()))
            .collect()
    }

    /// Semantic caller-workspace bindings retained by the synthetic script
    /// entrypoint. Named procedures are intentionally excluded: their binding
    /// locals belong to their own lexical frame.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn entrypoint_workspace_bindings(&self) -> Vec<(runmat_types::BindingId, String)> {
        let Some(entrypoint) = self.script_entrypoint() else {
            return Vec::new();
        };
        let Some(body) = self.mir.bodies.get(&entrypoint) else {
            return Vec::new();
        };
        let names = self.binding_names();
        body.locals
            .iter()
            .filter(|local| local.kind == runmat_mir::MirLocalKind::Binding)
            .filter_map(|local| {
                let binding = local.binding?;
                names.get(&binding).cloned().map(|name| (binding, name))
            })
            .collect()
    }

    pub(crate) fn script_entrypoint(&self) -> Option<runmat_hir::FunctionId> {
        self.mir
            .entrypoints
            .iter()
            .copied()
            .find(|function| {
                self.mir.functions.get(function).is_some_and(|metadata| {
                    metadata.kind == runmat_hir::FunctionKind::SyntheticEntrypoint
                })
            })
            .or_else(|| self.mir.entrypoints.first().copied())
    }

    pub fn procedure_names(&self) -> Vec<String> {
        let mut names = self.functions.names.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }

    /// Resolve the canonical MIR/native function identity by source name.
    /// Interactive VM publication may remap its registry identities into the
    /// session namespace; native compilation retains the immutable MIR identity.
    pub(crate) fn native_function_id(&self, name: &str) -> Option<runmat_hir::FunctionId> {
        self.mir
            .functions
            .iter()
            .find_map(|(function, metadata)| (metadata.name.0 == name).then_some(*function))
    }

    pub(crate) fn procedure_input_count(&self, name: &str) -> Option<usize> {
        self.functions
            .resolve_name(name)
            .and_then(|id| self.functions.get(id))
            .map(|function| function.input_slots.len())
    }

    pub(crate) fn bytecode(&self) -> &runmat_vm::Bytecode {
        &self.bytecode
    }

    pub(crate) fn functions(&self) -> &runmat_vm::FunctionRegistry {
        &self.functions
    }
}
