use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;

use runmat_execution::executable::ExecutableStaticImport;
use runmat_hir::FunctionId;

use super::{CoveragePlan, ExecutableRevision, ExecutableSource, ExecutableSourceMap};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutableFunctionParameter {
    pub name: String,
    pub value: runmat_types::ValueFact,
    pub arity: runmat_types::OperationParameterArity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutableFunctionInterface {
    pub declaration_span: runmat_types::Span,
    pub inputs: Vec<ExecutableFunctionParameter>,
    pub outputs: Vec<runmat_types::ValueFact>,
    pub effects: runmat_types::EffectSet,
    pub cooperative_suspension: bool,
    pub capabilities: runmat_types::CapabilitySet,
}

#[path = "unit/compiled_owners.rs"]
mod compiled_owners;
use compiled_owners::compiled_function_owners;
#[path = "unit/draft.rs"]
mod draft;
pub(crate) use draft::ExecutableUnitDraft;
#[path = "unit/class_products.rs"]
mod class_products;
pub(crate) use class_products::RuntimeConstructorAdmissions;
#[path = "unit/code_cache.rs"]
mod code_cache;
#[path = "unit/function_identity.rs"]
mod function_identity;
#[path = "unit/selected_runtime.rs"]
mod selected_runtime;

/// Immutable canonical compilation product reusable across invocations.
#[derive(Clone, Debug)]
pub struct ExecutableUnit {
    source: Rc<ExecutableSource>,
    revision: ExecutableRevision,
    source_map: Rc<ExecutableSourceMap>,
    coverage: Rc<CoveragePlan>,
    mir: Rc<runmat_mir::MirAssembly>,
    analysis: Arc<runmat_mir::analysis::AnalysisStore>,
    bytecode: Rc<runmat_vm::Bytecode>,
    layout: Rc<runmat_vm::VmAssemblyLayout>,
    functions: Rc<runmat_vm::FunctionRegistry>,
    portable_selection: Option<Rc<PortableSelection>>,
    direct_code_cache: Rc<code_cache::DirectCodeCache>,
    /// Original compiler products this unit's deferred callables depend on.
    pub(super) original_dependencies:
        std::collections::BTreeMap<runmat_types::FunctionId, Rc<ExecutableUnit>>,
    /// Contracts of a restored portable product, reproduced exactly on re-export.
    pub(super) imported_contract: Option<Rc<ImportedPortableContract>>,
    #[cfg(not(target_arch = "wasm32"))]
    native_compilation_digest: Rc<std::cell::OnceCell<runmat_execution::Digest>>,
}

/// Manifest contracts retained from an imported portable product.
#[derive(Debug)]
pub(super) struct ImportedPortableContract {
    pub interop: runmat_types::InteropManifest,
    pub parallel: runmat_types::ParallelManifest,
    pub optional_sections: Vec<runmat_execution::ExecutableOptionalSection>,
}

/// Compiler-selected bodies and their exact outbound product references.
/// The wire section and its original selected dependency owners are kept
/// together. Targets never point back to this unit or to its direct-code cache.
#[derive(Debug)]
struct PortableSelection {
    current: HashSet<FunctionId>,
    imports: Vec<ExecutableStaticImport>,
    targets: Vec<crate::session::SelectedStaticTarget>,
}

impl ExecutableUnit {
    pub(crate) fn with_portable_selection(
        mut self,
        current: HashSet<FunctionId>,
        imports: Vec<ExecutableStaticImport>,
        targets: Vec<crate::session::SelectedStaticTarget>,
    ) -> Result<Self, String> {
        let compiled = self.compiled_product_owners()?;
        if !current
            .iter()
            .all(|id| compiled.registered_bodies.contains(id))
        {
            return Err("selected portable body is not owned by this compiled product".into());
        }
        if !imports
            .iter()
            .eq(targets.iter().map(|target| &target.import))
        {
            return Err("selected portable imports differ from their original targets".into());
        }
        self.reset_direct_code_cache();
        self.portable_selection = Some(Rc::new(PortableSelection {
            current,
            imports,
            targets,
        }));
        Ok(self)
    }

    pub(crate) fn original_static_targets(&self) -> &[crate::session::SelectedStaticTarget] {
        self.portable_selection
            .as_ref()
            .map_or(&[], |selection| &selection.targets)
    }

    pub(crate) fn original_static_imports(&self) -> &[ExecutableStaticImport] {
        self.portable_selection
            .as_ref()
            .map_or(&[], |selection| selection.imports.as_slice())
    }

    /// Restore the existing selection from VM-admitted sites without claiming
    /// that any old runtime definition/source leases survived transport.
    pub(super) fn with_admitted_portable_selection(
        mut self,
        current: HashSet<FunctionId>,
        imports: Vec<ExecutableStaticImport>,
    ) -> Result<Self, String> {
        if current != self.compiled_current_body_ids()? {
            return Err("admitted selection differs from its original compiled bodies".into());
        }
        self.portable_selection = Some(Rc::new(PortableSelection {
            current,
            imports,
            targets: Vec::new(),
        }));
        Ok(self)
    }

    pub(crate) fn compiled_current_body_ids(&self) -> Result<HashSet<FunctionId>, String> {
        Ok(self
            .compiled_product_owners()?
            .registered_bodies
            .into_iter()
            .collect())
    }

    pub(crate) fn compiled_script_owner_ids(
        &self,
    ) -> Result<std::collections::BTreeSet<FunctionId>, String> {
        let owners = self.compiled_product_owners()?;
        Ok(owners
            .assembly
            .difference(&owners.registered_bodies)
            .copied()
            .collect())
    }

    fn compiled_product_owners(&self) -> Result<compiled_owners::CompiledProductOwners, String> {
        let program = self
            .revision
            .program_revision
            .identity_digest()
            .map_err(|error| error.to_string())?;
        compiled_function_owners(program, &self.mir, &self.bytecode)
    }

    pub(super) fn portable_selection(
        &self,
    ) -> Option<(&HashSet<FunctionId>, &[ExecutableStaticImport])> {
        self.portable_selection
            .as_deref()
            .map(|selection| (&selection.current, selection.imports.as_slice()))
    }
    /// Retain the canonical frame layout before installing executable regions.
    pub(crate) fn retain_vm_layout(
        bytecode: &mut runmat_vm::Bytecode,
        hir: &runmat_hir::HirAssembly,
        mir: &runmat_mir::MirAssembly,
    ) -> Result<(), runmat_vm::LayoutError> {
        if bytecode.layout.is_none() {
            bytecode.layout = Some(runmat_vm::derive_layout(hir, mir)?);
        }
        Ok(())
    }

    /// Bind immutable program ordinals to fresh session-issued provenance for
    /// one invocation. The returned context retains every lease while pending.
    pub(crate) fn bind_runtime_sources(
        &self,
        runtime: runmat_runtime::context::RuntimeContext,
    ) -> Result<runmat_runtime::context::RuntimeContext, String> {
        let runtime = self.source_map.bind_runtime_sources(runtime)?;
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
            .program_source_ref(ordinal)
            .map(runmat_vm::BytecodeSourceRef::Runtime)
    }

    pub(crate) fn runtime_bytecode(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<runmat_vm::Bytecode, String> {
        let mut bytecode = self.bytecode().clone();
        bytecode.project_source_ids(|source| Self::runtime_source_id(runtime, source))?;
        let mut registry = bytecode.function_registry();
        self.attach_runtime_source_owner(&mut registry, runtime)?;
        for (id, selected) in &registry.functions {
            if let Some(bound) = bytecode.bound_functions.get_mut(id) {
                if bound.binding_origin == selected.binding_origin
                    && bound.source_id == selected.source_id
                    && std::sync::Arc::ptr_eq(&bound.definition_owner, &selected.definition_owner)
                {
                    bound.selected_source_owner = selected.selected_source_owner.clone();
                }
            }
        }
        bytecode.function_registry = registry.clone();
        Ok(bytecode)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn runtime_functions(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<runmat_vm::FunctionRegistry, String> {
        let mut functions = self.functions().clone();
        functions
            .project_current_source_ids(|source| Self::runtime_source_id(runtime, source))
            .map_err(|source| format!("function source {source:?} is absent"))?;
        self.attach_runtime_source_owner(&mut functions, runtime)?;
        let constructors =
            self.materialize_runtime_constructor_definitions(&mut functions, None)?;
        self.attach_runtime_class_products(&mut functions, &constructors)?;
        Ok(functions)
    }

    fn attach_runtime_source_owner(
        &self,
        registry: &mut runmat_vm::FunctionRegistry,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<(), String> {
        let owner = self.selected_source_owner(runtime)?;
        let program = self
            .revision
            .program_revision
            .identity_digest()
            .map_err(|error| error.to_string())?;
        let selected_ids = registry
            .functions
            .iter()
            .filter_map(|(id, body)| {
                body.binding_origin
                    .is_some_and(|origin| origin.program == program)
                    .then_some(*id)
            })
            .collect::<Vec<_>>();
        registry
            .attach_selected_program_source_owner(selected_ids, program, &owner)
            .map_err(str::to_string)
    }

    /// Retain the already-issued source binding without execution ports.
    pub(crate) fn selected_source_owner(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<runmat_runtime::source_context::SelectedProgramSourceOwner, String> {
        runmat_runtime::source_context::SelectedProgramSourceOwner::capture(
            runtime,
            &self.source_map.portable_v2()?,
        )
        .map_err(|error| error.to_string())
    }

    pub(crate) fn new(
        source: ExecutableSource,
        revision: ExecutableRevision,
        source_map: ExecutableSourceMap,
        mir: runmat_mir::MirAssembly,
        analysis: runmat_mir::analysis::AnalysisStore,
        bytecode: runmat_vm::Bytecode,
    ) -> Result<Self, String> {
        ExecutableUnitDraft::prepare(source, revision, source_map, mir, analysis, bytecode)
            .map(ExecutableUnitDraft::seal)
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

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn analysis_owner(&self) -> Arc<runmat_mir::analysis::AnalysisStore> {
        Arc::clone(&self.analysis)
    }

    /// Exact static facts for a function in this immutable executable revision.
    pub fn function_analysis(
        &self,
        function: runmat_types::ProgramFunctionId,
    ) -> Option<&runmat_mir::analysis::FunctionAnalysis> {
        self.analysis
            .functions
            .iter()
            .find(|analysis| analysis.function == function)
    }

    /// The compiler's primary coordinate, checked against the captured source
    /// descriptor. Source-table order and a function's display name are not
    /// primary-source authority.
    pub(crate) fn primary_source_id(&self) -> Result<runmat_types::ProgramSourceId, String> {
        let source = self
            .bytecode
            .source_id
            .and_then(runmat_vm::BytecodeSourceRef::program)
            .ok_or("executable has no compiled primary-source coordinate")?;
        let entry = self
            .source_map
            .entries()
            .iter()
            .find(|entry| entry.source_id == source)
            .ok_or("compiled primary source is absent from the source table")?;
        if entry.owner_identity != self.source.owner_identity
            || entry.relative_path != self.source.relative_path
        {
            return Err("compiled primary source differs from its captured descriptor".into());
        }
        Ok(source)
    }

    /// Return the exact compiler facts for one named, top-level function defined by the primary
    /// source of this executable. Companion-project functions with the same name are never used as
    /// a substitute.
    pub fn primary_function_interface(
        &self,
        name: &str,
    ) -> Result<ExecutableFunctionInterface, String> {
        let primary_source = self.primary_source_id()?;
        let matches = self
            .mir
            .functions
            .iter()
            .filter(|(_, metadata)| {
                metadata.parent.is_none()
                    && matches!(metadata.kind, runmat_hir::FunctionKind::Named)
                    && metadata.name.0 == name
                    && metadata.source == primary_source
            })
            .collect::<Vec<_>>();
        let (function, metadata) = match matches.as_slice() {
            [entry] => *entry,
            [] => {
                return Err(format!(
                    "primary source `{}` does not define top-level function `{name}`",
                    self.source.relative_path
                ))
            }
            _ => {
                return Err(format!(
                    "primary source `{}` defines top-level function `{name}` more than once",
                    self.source.relative_path
                ))
            }
        };
        let bytecode = self
            .functions
            .get(*function)
            .ok_or_else(|| format!("function `{name}` has no executable interface"))?;
        let program_function = bytecode.portable_function;
        let origin = bytecode
            .binding_origin
            .ok_or("function has no original binding identity")?;
        if self.local_function_for_origin(origin)? != Some(*function) {
            return Err("function interface is not owned by this executable revision".into());
        }
        let analysis = self
            .function_analysis(program_function)
            .ok_or_else(|| format!("function `{name}` has no static analysis facts"))?;
        if !analysis.callable.parameters_complete || !analysis.callable.outputs_complete {
            return Err(format!(
                "static callable facts for function `{name}` are incomplete"
            ));
        }
        if bytecode.input_slots.len() != analysis.callable.parameters.len() {
            return Err(format!(
                "parameter identities and static facts for function `{name}` do not agree"
            ));
        }
        let inputs = bytecode
            .input_slots
            .iter()
            .zip(&analysis.callable.parameters)
            .map(|(slot, value)| {
                let name = bytecode.var_names.get(slot).cloned().ok_or_else(|| {
                    format!("function `{name}` parameter slot {slot} has no source name")
                })?;
                let arity = if bytecode.varargin_slot == Some(*slot) {
                    runmat_types::OperationParameterArity::Variadic
                } else if bytecode.argument_validations.iter().any(|validation| {
                    validation.input_slot == *slot && validation.default_value.is_some()
                }) {
                    runmat_types::OperationParameterArity::Optional
                } else {
                    runmat_types::OperationParameterArity::Required
                };
                Ok(ExecutableFunctionParameter {
                    name,
                    value: value.clone(),
                    arity,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(ExecutableFunctionInterface {
            declaration_span: metadata.span,
            inputs,
            outputs: analysis.outputs.clone(),
            effects: analysis.effects.clone(),
            cooperative_suspension: analysis.cooperative_suspension,
            capabilities: analysis.capabilities.clone(),
        })
    }

    /// Canonical analyzed MIR retained for native lowering and artifact production.
    pub fn mir(&self) -> &runmat_mir::MirAssembly {
        &self.mir
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn cached_native_compilation_digest(&self) -> Option<runmat_execution::Digest> {
        self.native_compilation_digest.get().copied()
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
        self.reachability_analysis().0
    }

    /// Typed function retention and explanatory report from the same MIR walk.
    pub fn reachability_analysis(
        &self,
    ) -> (
        runmat_mir::analysis::ReachabilityReport,
        std::collections::BTreeSet<runmat_hir::FunctionId>,
    ) {
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
        runmat_mir::analysis::analyze_reachability_with_functions(&self.mir, &names)
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
            .filter(|local| {
                matches!(
                    local.kind,
                    runmat_mir::MirLocalKind::Binding | runmat_mir::MirLocalKind::IncomingWorkspace
                )
            })
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
    /// A class-source request may use its typed constructor owner when no
    /// function has the exact public name.
    #[cfg(any(not(target_arch = "wasm32"), test))]
    pub(crate) fn native_function_id(&self, name: &str) -> Option<runmat_hir::FunctionId> {
        native_function_id_in_mir(&self.mir, name)
    }

    /// Select the constructor declared by this exact primary class source.
    /// Companion definitions and same-leaf classes cannot become its entrypoint.
    pub(crate) fn source_constructor_function_id(
        &self,
        class: &runmat_types::ClassIdentity,
        primary_source: runmat_types::ProgramSourceId,
    ) -> Option<runmat_hir::FunctionId> {
        constructor_function_id_in_mir(&self.mir, class, Some(primary_source))
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

#[cfg(any(not(target_arch = "wasm32"), test))]
fn native_function_id_in_mir(
    mir: &runmat_mir::MirAssembly,
    name: &str,
) -> Option<runmat_hir::FunctionId> {
    let exact = mir.functions.iter().find_map(|(function, metadata)| {
        (metadata.kind != runmat_hir::FunctionKind::SyntheticEntrypoint && metadata.name.0 == name)
            .then_some(*function)
    });
    exact.or_else(|| {
        let requested_class = runmat_types::ClassIdentity::new(name).ok()?;
        constructor_function_id_in_mir(mir, &requested_class, None)
    })
}

fn constructor_function_id_in_mir(
    mir: &runmat_mir::MirAssembly,
    class: &runmat_types::ClassIdentity,
    source: Option<runmat_types::ProgramSourceId>,
) -> Option<runmat_hir::FunctionId> {
    let mut constructors = mir.functions.iter().filter_map(|(function, metadata)| {
        metadata.class_method_owner.as_ref().and_then(|owner| {
            (source.is_none_or(|source| metadata.source == source)
                && matches!(
                    metadata.kind,
                    runmat_hir::FunctionKind::ClassMethod {
                        is_static: false,
                        role: runmat_types::ClassMethodRole::Constructor,
                    }
                )
                && owner.role == runmat_types::ClassMethodRole::Constructor
                && !owner.is_static
                && &owner.declaring_class == class)
                .then_some(*function)
        })
    });
    let selected = constructors.next()?;
    constructors.next().is_none().then_some(selected)
}

#[cfg(test)]
#[path = "unit/tests.rs"]
mod tests;

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
