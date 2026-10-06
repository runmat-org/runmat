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
    #[cfg(not(target_arch = "wasm32"))]
    native_compilation_digest: Rc<std::cell::OnceCell<runmat_execution::Digest>>,
}

impl ExecutableUnit {
    /// Bind immutable program ordinals to fresh session-issued provenance for
    /// one invocation. The returned context retains every lease while pending.
    pub(crate) fn bind_runtime_sources(
        &self,
        runtime: runmat_runtime::context::RuntimeContext,
    ) -> Result<runmat_runtime::context::RuntimeContext, String> {
        self.source_map.validate_content_revisions()?;
        let records = self
            .source_map
            .entries()
            .iter()
            .map(|entry| {
                Ok(runmat_runtime::source_context::ProgramSourceRecord {
                    ordinal: entry.source_id,
                    name: std::sync::Arc::from(entry.display_name.as_str()),
                    fullpath_name: entry.full_path.as_deref().map(std::sync::Arc::<str>::from),
                    text: std::sync::Arc::from(entry.text.as_str()),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let runtime = runtime
            .with_program_sources(records)
            .map_err(|error| error.to_string())?;
        for metadata in self.mir.functions.values() {
            if runtime.program_source_id(metadata.source).is_none() {
                return Err(format!(
                    "MIR function source ordinal {} is absent from the invocation binding",
                    metadata.source.0
                ));
            }
        }
        Ok(runtime)
    }

    fn runtime_source_id(
        runtime: &runmat_runtime::context::RuntimeContext,
        source: runmat_vm::BytecodeSourceRef,
    ) -> Option<runmat_vm::BytecodeSourceRef> {
        let ordinal = source.program()?;
        runtime
            .program_source_id(ordinal)
            .map(runmat_vm::BytecodeSourceRef::Runtime)
    }

    pub(crate) fn runtime_bytecode(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<runmat_vm::Bytecode, String> {
        let mut bytecode = self.bytecode().clone();
        bytecode.source_id = bytecode
            .source_id
            .map(|source| {
                Self::runtime_source_id(runtime, source)
                    .ok_or_else(|| format!("entrypoint source {source:?} is absent"))
            })
            .transpose()?;
        bytecode.function_registry = self.runtime_functions(runtime)?;
        for body in bytecode.bound_functions.values_mut() {
            body.source_id = body
                .source_id
                .map(|source| {
                    Self::runtime_source_id(runtime, source)
                        .ok_or_else(|| format!("bound source {source:?} is absent"))
                })
                .transpose()?;
        }
        Ok(bytecode)
    }

    pub(crate) fn runtime_functions(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<runmat_vm::FunctionRegistry, String> {
        let mut functions = self.functions().clone();
        functions
            .project_current_source_ids(|source| Self::runtime_source_id(runtime, source))
            .map_err(|source| format!("function source {source:?} is absent"))?;
        Ok(functions)
    }

    pub(crate) fn new(
        source: ExecutableSource,
        revision: ExecutableRevision,
        source_map: ExecutableSourceMap,
        mir: runmat_mir::MirAssembly,
        analysis: runmat_mir::analysis::AnalysisStore,
        mut bytecode: runmat_vm::Bytecode,
    ) -> Result<Self, String> {
        source_map.validate_content_revisions()?;
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
        let region_contracts = analysis
            .regions
            .iter()
            .map(|region| region.contract.clone())
            .collect::<Vec<_>>();
        bytecode.install_regions(&region_contracts)?;
        bytecode.install_parallel_regions(&mir, &analysis.parallel)?;
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
            #[cfg(not(target_arch = "wasm32"))]
            native_compilation_digest: Rc::new(std::cell::OnceCell::new()),
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

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn native_compilation_digest(&self) -> Result<runmat_execution::Digest, String> {
        if let Some(digest) = self.native_compilation_digest.get() {
            return Ok(*digest);
        }
        // Fingerprint the exact lowering inputs once, only when a native
        // product is requested. Clones share this immutable unit's digest.
        let digest = self.native_product_digest()?;
        let _ = self.native_compilation_digest.set(digest);
        Ok(digest)
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
            .map(|entry| (entry.source_id, entry.relative_path.clone()))
            .collect();
        let names = runmat_mir::analysis::ReachabilityNames {
            sources,
            bindings: self.binding_names(),
        };
        runmat_mir::analysis::analyze_reachability(&self.mir, &names)
    }

    /// Whether the executable's closed, statically visible surface requires
    /// the Python foreign runtime. Dynamic calls remain represented by the
    /// reachability report's unknown edges and are not guessed here.
    pub fn requires_python_runtime(&self) -> bool {
        reachability_requires_python(&self.reachability_report())
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
        self.mir.functions.iter().find_map(|(function, metadata)| {
            (metadata.kind != runmat_hir::FunctionKind::SyntheticEntrypoint
                && metadata.name.0 == name)
                .then_some(*function)
        })
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

pub fn reachability_requires_python(report: &runmat_mir::analysis::ReachabilityReport) -> bool {
    report.nodes.iter().any(|node| {
        node.symbol.starts_with("py.")
            || matches!(
                node.symbol.to_ascii_lowercase().as_str(),
                "pyenv" | "pyrun" | "pyrunfile" | "pyargs"
            )
            || (node.kind == runmat_mir::analysis::ReachabilityNodeKind::ArtifactDependency
                && node.symbol == "python")
    })
}
