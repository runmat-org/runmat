use runmat_execution::{ExecutableComponentKind, ExecutableUnitEnvelope};
use std::sync::Arc;

use crate::{Bytecode, FunctionRegistry, VmAssemblyLayout};

#[path = "unit_admission/entrypoint.rs"]
mod entrypoint;
#[path = "unit_admission/graph_codec.rs"]
mod graph_codec;
#[path = "unit_admission/graph_static.rs"]
mod graph_static;
#[path = "unit_admission/retained_code.rs"]
mod retained_code;
#[path = "unit_admission/retained_source_owner.rs"]
mod retained_source_owner;
pub(super) use retained_code::RetainedExecutableUnit;
use retained_code::UnitAdmissionIdentity;
#[path = "unit_admission/selected_interop.rs"]
mod selected_interop;
#[path = "unit_admission/static_import.rs"]
mod static_import;
pub use graph_codec::{
    selected_graph_callable_facts, selected_graph_callable_facts_for_owner, GraphCodecAdmission,
    SelectedGraphCallableFacts,
};
pub use retained_source_owner::RetainedExecutableSourceOwner;
pub use selected_interop::SelectedUnitInteropReceipt;

/// Fully decoded executable unit whose component identities and revisions have
/// been validated, but whose portable source ordinals have not yet issued
/// runtime-local source leases.
#[derive(Clone, Debug)]
pub struct AdmittedExecutableUnit {
    envelope: ExecutableUnitEnvelope,
    canonical_v3: Arc<[u8]>,
    bytecode: Bytecode,
    registry: FunctionRegistry,
    layout: VmAssemblyLayout,
    source_map: runmat_execution::ExecutableSourceMapV2,
    entrypoint: entrypoint::AdmittedEntrypoint,
    static_imports: Vec<runmat_execution::executable::ExecutableStaticImport>,
    linked_native_imports: Vec<runmat_execution::executable::LinkedNativeStaticImport>,
}

/// Active executable unit bound to one exact runtime source owner. Admission
/// and execution keep that runtime with the registry. Checked retention moves
/// the admitted code and actual source owner, never the invocation's grants.
#[derive(Debug)]
pub struct RuntimeBoundExecutableUnit {
    admission: Option<UnitAdmissionIdentity>,
    canonical_v3: Option<Arc<[u8]>>,
    selected_source_map: Option<runmat_execution::ExecutableSourceMapV2>,
    selected_source_owner: Option<runmat_runtime::source_context::SelectedProgramSourceOwner>,
    bytecode: Bytecode,
    registry: FunctionRegistry,
    layout: VmAssemblyLayout,
    runtime: runmat_runtime::context::RuntimeContext,
    entrypoint: RuntimeBoundEntrypoint,
    static_imports: Vec<runmat_execution::executable::ExecutableStaticImport>,
    linked_native_imports: Vec<runmat_execution::executable::LinkedNativeStaticImport>,
    declares_interop: bool,
}

#[derive(Debug)]
enum RuntimeBoundEntrypoint {
    Function(runmat_hir::FunctionId),
    Script,
    DeclarationOnly(runmat_types::ClassIdentity),
    #[cfg(test)]
    Fixture,
}

impl AdmittedExecutableUnit {
    /// Move accepted code into a compiler-owned session reconstruction.
    /// Nonwire execution contracts remain attached to these exact bodies;
    /// decoding the registry again would discard their admitted MIR evidence.
    /// This grants no runtime source ownership or invocation capabilities.
    pub fn into_compiler_components(self) -> (Bytecode, FunctionRegistry, VmAssemblyLayout) {
        (self.bytecode, self.registry, self.layout)
    }

    /// Validated code and class declarations retained by this admitted unit.
    pub fn bytecode(&self) -> &Bytecode {
        &self.bytecode
    }

    /// Registry-local function admitted by the manifest's immutable origin.
    pub fn entrypoint_function(&self) -> Option<runmat_hir::FunctionId> {
        match &self.entrypoint {
            entrypoint::AdmittedEntrypoint::Function { local_function, .. } => {
                Some(*local_function)
            }
            entrypoint::AdmittedEntrypoint::Script { .. }
            | entrypoint::AdmittedEntrypoint::DeclarationOnly { .. } => None,
        }
    }

    pub fn envelope(&self) -> &ExecutableUnitEnvelope {
        &self.envelope
    }

    pub fn admitted_source_map(&self) -> &runmat_execution::ExecutableSourceMapV2 {
        &self.source_map
    }

    pub fn canonical_v3(&self) -> &Arc<[u8]> {
        &self.canonical_v3
    }

    pub fn registry(&self) -> &FunctionRegistry {
        &self.registry
    }

    pub fn static_imports(&self) -> &[runmat_execution::executable::ExecutableStaticImport] {
        &self.static_imports
    }

    pub fn linked_native_imports(
        &self,
    ) -> &[runmat_execution::executable::LinkedNativeStaticImport] {
        &self.linked_native_imports
    }

    pub fn fingerprint_imports(&self) -> Vec<runmat_execution::executable::ExecutableStaticImport> {
        self.static_imports
            .iter()
            .copied()
            .chain(self.linked_native_imports.iter().copied().map(
                runmat_execution::executable::LinkedNativeStaticImport::fingerprint_projection,
            ))
            .collect()
    }

    /// Canonical per-origin definition fingerprints for this admitted unit.
    /// Portable source positions are resolved through this unit's validated
    /// source map and cannot be confused with immutable source identity.
    pub fn retained_callable_origin_fingerprints(
        &self,
    ) -> Result<
        std::collections::HashMap<runmat_execution::FunctionBindingOrigin, String>,
        crate::RetainedCallableStageError,
    > {
        crate::bytecode::program::retained_artifacts::admitted_callable_origin_fingerprints_with_imports(
            &self.registry,
            &self.source_map,
            &self.fingerprint_imports(),
        )
    }

    pub fn admitted_semantic_body_digests(
        &self,
    ) -> Result<
        std::collections::HashMap<
            runmat_execution::FunctionBindingOrigin,
            runmat_execution::executable::SelectedBodySemanticDigest,
        >,
        crate::RetainedCallableStageError,
    > {
        crate::bytecode::program::retained_artifacts::admitted_semantic_body_digests(
            &self.registry,
            &self.source_map,
            &self.fingerprint_imports(),
        )
    }

    /// Resolve the manifest entrypoint to the exact immutable definition
    /// origin admitted with this unit. A display name or a matching numeric
    /// ordinal alone is not executable authority.
    pub fn entrypoint_origin(
        &self,
    ) -> Result<runmat_execution::FunctionBindingOrigin, ExecutableUnitAdmissionError> {
        match &self.entrypoint {
            entrypoint::AdmittedEntrypoint::Function { origin, .. }
            | entrypoint::AdmittedEntrypoint::Script { origin } => Ok(*origin),
            entrypoint::AdmittedEntrypoint::DeclarationOnly { .. } => {
                Err(ExecutableUnitAdmissionError::SourceBinding(
                    "declaration-only product has no callable entrypoint origin".into(),
                ))
            }
        }
    }

    pub fn declaration_class(&self) -> Option<&runmat_types::ClassIdentity> {
        match &self.entrypoint {
            entrypoint::AdmittedEntrypoint::DeclarationOnly { class } => Some(class),
            entrypoint::AdmittedEntrypoint::Function { .. }
            | entrypoint::AdmittedEntrypoint::Script { .. } => None,
        }
    }

    pub fn bind_runtime_sources(
        self,
        runtime: runmat_runtime::context::RuntimeContext,
    ) -> Result<RuntimeBoundExecutableUnit, ExecutableUnitAdmissionError> {
        self.bind_runtime_sources_with_optional_records(runtime, None, None, None)
            .map(|(unit, _)| unit)
    }

    /// Bind the exact admitted V3 and its matching unit-owned foreign bundle
    /// under one issued source owner. A declaration-only product has no body
    /// to carry the returned checked receipt, so its selected class product
    /// retains the receipt for the duration of graph execution.
    pub fn bind_runtime_sources_with_bundle(
        self,
        runtime: runmat_runtime::context::RuntimeContext,
        bundle: Option<runmat_execution_artifact::ExecutionBundle>,
        limits: runmat_execution_artifact::archive::ArchiveLimits,
    ) -> Result<
        (
            RuntimeBoundExecutableUnit,
            Option<std::rc::Rc<crate::SelectedUnitInteropReceipt>>,
        ),
        ExecutableUnitAdmissionError,
    > {
        self.bind_runtime_sources_with_optional_records(runtime, None, bundle, Some(limits))
    }

    /// Bind an admitted portable unit to the originating host's captured
    /// source records in one issuance. Records are checked against the
    /// authenticated portable map; host display paths do not enter its hash.
    pub fn bind_runtime_sources_with_records(
        self,
        runtime: runmat_runtime::context::RuntimeContext,
        records: Vec<runmat_runtime::source_context::ProgramSourceRecord>,
    ) -> Result<RuntimeBoundExecutableUnit, ExecutableUnitAdmissionError> {
        self.bind_runtime_sources_with_optional_records(runtime, Some(records), None, None)
            .map(|(unit, _)| unit)
    }

    fn bind_runtime_sources_with_optional_records(
        mut self,
        runtime: runmat_runtime::context::RuntimeContext,
        records: Option<Vec<runmat_runtime::source_context::ProgramSourceRecord>>,
        bundle: Option<runmat_execution_artifact::ExecutionBundle>,
        limits: Option<runmat_execution_artifact::archive::ArchiveLimits>,
    ) -> Result<
        (
            RuntimeBoundExecutableUnit,
            Option<std::rc::Rc<crate::SelectedUnitInteropReceipt>>,
        ),
        ExecutableUnitAdmissionError,
    > {
        let runtime = super::unit_sources::bind_unit_sources(
            &self.envelope,
            &self.source_map,
            &mut self.bytecode,
            &mut self.registry,
            runtime,
            records,
        )
        .map_err(ExecutableUnitAdmissionError::SourceBinding)?;
        let source_owner = runmat_runtime::source_context::SelectedProgramSourceOwner::capture(
            &runtime,
            &self.source_map,
        )
        .map_err(|error| ExecutableUnitAdmissionError::SourceBinding(error.to_string()))?;
        let interop = &self.envelope.manifest.interop;
        let has_interop = !interop.foreign_types.is_empty()
            || !interop.adapters.is_empty()
            || !interop.adapter_contracts.is_empty();
        let interop_receipt = if has_interop || bundle.is_some() {
            crate::SelectedUnitInteropReceipt::from_bundle(
                bundle,
                &self.canonical_v3,
                &source_owner,
                limits.unwrap_or_default(),
            )
            .map_err(ExecutableUnitAdmissionError::SourceBinding)?
        } else {
            None
        };
        let program = self
            .envelope
            .manifest
            .identity
            .program
            .identity_digest()
            .map_err(|error| ExecutableUnitAdmissionError::SourceBinding(error.to_string()))?;
        let ids = self.registry.functions.keys().copied().collect::<Vec<_>>();
        self.registry
            .attach_selected_program_source_owner(ids, program, &source_owner)
            .map_err(|error| ExecutableUnitAdmissionError::SourceBinding(error.into()))?;
        for (id, selected) in &self.registry.functions {
            if let Some(bound) = self.bytecode.bound_functions.get_mut(id) {
                if bound.binding_origin == selected.binding_origin
                    && bound.source_id == selected.source_id
                    && std::sync::Arc::ptr_eq(&bound.definition_owner, &selected.definition_owner)
                {
                    bound.selected_source_owner = selected.selected_source_owner.clone();
                }
            }
        }
        self.bytecode.function_registry = self.registry.clone();
        self.bytecode.layout = Some(self.layout.clone());
        let entrypoint = match self.entrypoint {
            entrypoint::AdmittedEntrypoint::Function { local_function, .. } => {
                RuntimeBoundEntrypoint::Function(local_function)
            }
            entrypoint::AdmittedEntrypoint::Script { .. } => RuntimeBoundEntrypoint::Script,
            entrypoint::AdmittedEntrypoint::DeclarationOnly { class } => {
                RuntimeBoundEntrypoint::DeclarationOnly(class)
            }
        };
        Ok((
            RuntimeBoundExecutableUnit {
                admission: Some(UnitAdmissionIdentity {
                    revision: self.envelope.manifest.identity.program,
                    interop: self.envelope.manifest.interop,
                }),
                canonical_v3: Some(self.canonical_v3),
                selected_source_map: Some(self.source_map),
                selected_source_owner: Some(source_owner),
                bytecode: self.bytecode,
                registry: self.registry,
                layout: self.layout,
                runtime,
                entrypoint,
                static_imports: self.static_imports,
                linked_native_imports: self.linked_native_imports,
                declares_interop: has_interop,
            },
            interop_receipt,
        ))
    }
}

impl RuntimeBoundExecutableUnit {
    /// Pin the checked code/source product on every body before retained
    /// staging remaps those bodies into the destination's retired registry.
    /// Dynamic source loading currently admits only units without foreign
    /// artifacts; a foreign-bearing unit needs its checked archive receipt.
    pub fn attach_local_selected_product(&mut self) -> Result<(), String> {
        let bytes = self
            .canonical_v3
            .as_ref()
            .ok_or("selected unit has no canonical V3 product")?;
        let owner = self
            .selected_source_owner
            .as_ref()
            .ok_or("selected unit has no issued source owner")?;
        if self.declares_interop {
            return Err("dynamic selected foreign unit needs its checked archive receipt".into());
        }
        let product = crate::SelectedExecutableProduct::admit(Arc::clone(bytes), owner.clone())?;
        let ids = self.registry.functions.keys().copied().collect::<Vec<_>>();
        self.registry
            .attach_current_selected_executable_product(ids.iter().copied(), product)
            .map_err(|error| error.to_string())?;
        for id in ids {
            let selected = self
                .registry
                .get_current(id)
                .ok_or("selected unit lost its current definition")?;
            if let Some(bound) = self.bytecode.bound_functions.get_mut(&id) {
                if bound.binding_origin != selected.binding_origin
                    || bound.source_id != selected.source_id
                    || !Arc::ptr_eq(&bound.definition_owner, &selected.definition_owner)
                {
                    return Err("selected unit bound body differs from its registry owner".into());
                }
                bound.selected_executable_product = selected.selected_executable_product.clone();
            }
        }
        self.bytecode.function_registry = self.registry.clone();
        Ok(())
    }

    pub fn static_imports(&self) -> &[runmat_execution::executable::ExecutableStaticImport] {
        &self.static_imports
    }

    pub fn linked_native_imports(
        &self,
    ) -> &[runmat_execution::executable::LinkedNativeStaticImport] {
        &self.linked_native_imports
    }

    pub fn declares_interop(&self) -> bool {
        self.declares_interop
    }

    pub fn canonical_v3(&self) -> Option<&Arc<[u8]>> {
        self.canonical_v3.as_ref()
    }

    pub fn selected_source_map(&self) -> Option<&runmat_execution::ExecutableSourceMapV2> {
        self.selected_source_map.as_ref()
    }

    pub fn selected_source_owner(
        &self,
    ) -> Option<&runmat_runtime::source_context::SelectedProgramSourceOwner> {
        self.selected_source_owner.as_ref()
    }

    pub fn matches_bound_definition(
        &self,
        origin: runmat_execution::FunctionBindingOrigin,
        lease: &runmat_value::FunctionDefinitionLease,
    ) -> bool {
        let mut bodies = self
            .registry
            .selected_execution_bodies()
            .into_iter()
            .filter(|body| body.binding_origin == Some(origin));
        let Some(body) = bodies.next() else {
            return false;
        };
        bodies.next().is_none() && lease.matches(&body.definition_owner)
    }

    /// Whether a decoded value or an in-flight staged graph still owns any
    /// definition from this source-bearing unit.
    pub fn has_live_definition_owners(&self) -> bool {
        self.registry
            .functions
            .values()
            .any(|body| body.definition_owner.has_handles())
    }

    /// Confirm that this unit's fresh source leases were issued into the
    /// selected destination session. Matching serialized or numeric source
    /// identities are not sufficient authority.
    pub fn validate_runtime_owner(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<(), &'static str> {
        if self.runtime.local_identity() != runtime.local_identity() {
            return Err("executable unit is bound to a different runtime owner");
        }
        self.bytecode.validate_selected_source_owners(runtime)?;
        self.registry.validate_selected_source_owners(runtime)
    }

    pub fn bytecode(&self) -> &Bytecode {
        &self.bytecode
    }

    pub fn registry(&self) -> &FunctionRegistry {
        &self.registry
    }

    pub fn layout(&self) -> &VmAssemblyLayout {
        &self.layout
    }

    pub fn runtime(&self) -> &runmat_runtime::context::RuntimeContext {
        &self.runtime
    }

    /// Exact registry key selected while admitting the function entrypoint.
    pub fn entrypoint_function(&self) -> Option<runmat_hir::FunctionId> {
        match self.entrypoint {
            RuntimeBoundEntrypoint::Function(function) => Some(function),
            RuntimeBoundEntrypoint::Script | RuntimeBoundEntrypoint::DeclarationOnly(_) => None,
            #[cfg(test)]
            RuntimeBoundEntrypoint::Fixture => None,
        }
    }

    pub fn declaration_class(&self) -> Option<&runmat_types::ClassIdentity> {
        match &self.entrypoint {
            RuntimeBoundEntrypoint::DeclarationOnly(class) => Some(class),
            RuntimeBoundEntrypoint::Function(_) | RuntimeBoundEntrypoint::Script => None,
            #[cfg(test)]
            RuntimeBoundEntrypoint::Fixture => None,
        }
    }

    pub fn into_parts(
        self,
    ) -> (
        Bytecode,
        FunctionRegistry,
        VmAssemblyLayout,
        runmat_runtime::context::RuntimeContext,
    ) {
        (self.bytecode, self.registry, self.layout, self.runtime)
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        registry: FunctionRegistry,
        runtime: runmat_runtime::context::RuntimeContext,
    ) -> Self {
        Self {
            canonical_v3: None,
            admission: None,
            selected_source_map: None,
            selected_source_owner: None,
            bytecode: Bytecode::with_instructions(Vec::new(), 0),
            registry,
            layout: VmAssemblyLayout {
                functions: Default::default(),
                entrypoints: Default::default(),
                storage_bindings: Default::default(),
            },
            runtime,
            entrypoint: RuntimeBoundEntrypoint::Fixture,
            static_imports: Vec::new(),
            linked_native_imports: Vec::new(),
            declares_interop: false,
        }
    }
}

#[derive(Debug)]
pub enum ExecutableUnitAdmissionError {
    InvalidHeader(runmat_execution::ContractError),
    CompilerRevision(runmat_execution::ContractError),
    ComponentRevisions(String),
    InvalidEnvelope(runmat_execution::ContractError),
    InvalidMir(serde_json::Error),
    InvalidSourceMap(runmat_execution::ContractError),
    MissingComponent(&'static str),
    InvalidBytecode(serde_json::Error),
    DuplicateComponentAuthority,
    InvalidRegistry(serde_json::Error),
    InvalidLayout(serde_json::Error),
    SourceBinding(String),
}

impl ExecutableUnitAdmissionError {
    pub(super) fn worker_message(&self) -> String {
        match self {
            Self::ComponentRevisions(message) | Self::SourceBinding(message) => message.clone(),
            _ => format!("worker rejected {self}"),
        }
    }
}

impl std::fmt::Display for ExecutableUnitAdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHeader(error) => {
                write!(formatter, "an invalid executable unit header: {error}")
            }
            Self::CompilerRevision(error) => {
                write!(formatter, "compiler-incompatible executable unit: {error}")
            }
            Self::ComponentRevisions(message) | Self::SourceBinding(message) => {
                formatter.write_str(message)
            }
            Self::InvalidEnvelope(error) => {
                write!(formatter, "an invalid executable unit: {error}")
            }
            Self::InvalidMir(error) => write!(formatter, "invalid executable MIR: {error}"),
            Self::InvalidSourceMap(error) => {
                write!(formatter, "invalid executable source map: {error}")
            }
            Self::MissingComponent(component) => {
                write!(formatter, "executable unit without {component}")
            }
            Self::InvalidBytecode(error) => {
                write!(formatter, "invalid executable bytecode: {error}")
            }
            Self::DuplicateComponentAuthority => {
                formatter.write_str("executable bytecode contains duplicate component authorities")
            }
            Self::InvalidRegistry(error) => {
                write!(formatter, "invalid executable registry: {error}")
            }
            Self::InvalidLayout(error) => {
                write!(formatter, "invalid executable VM layout: {error}")
            }
        }
    }
}

impl std::error::Error for ExecutableUnitAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidHeader(error)
            | Self::CompilerRevision(error)
            | Self::InvalidEnvelope(error)
            | Self::InvalidSourceMap(error) => Some(error),
            Self::InvalidMir(error)
            | Self::InvalidBytecode(error)
            | Self::InvalidRegistry(error)
            | Self::InvalidLayout(error) => Some(error),
            Self::ComponentRevisions(_)
            | Self::MissingComponent(_)
            | Self::DuplicateComponentAuthority
            | Self::SourceBinding(_) => None,
        }
    }
}

/// Validate and decode one canonical unit without issuing runtime-local source
/// leases or mutating a destination definition registry.
pub fn admit_executable_unit(
    bytes: &[u8],
) -> Result<AdmittedExecutableUnit, ExecutableUnitAdmissionError> {
    let admission = ExecutableUnitEnvelope::admission(bytes)
        .map_err(ExecutableUnitAdmissionError::InvalidHeader)?;
    admission
        .identity
        .program
        .validate_current_compiler()
        .map_err(ExecutableUnitAdmissionError::CompilerRevision)?;
    if let Some(message) = super::unsupported_executable_revisions(&admission.revisions) {
        return Err(ExecutableUnitAdmissionError::ComponentRevisions(message));
    }
    let envelope = ExecutableUnitEnvelope::from_canonical_bytes(bytes)
        .map_err(ExecutableUnitAdmissionError::InvalidEnvelope)?;
    let supported_sections = runmat_execution::ExecutableSectionSupport::known([
        runmat_execution::executable::KnownExecutableSection::StaticImports,
        runmat_execution::executable::KnownExecutableSection::LinkedNativeImports,
    ])
    .map_err(ExecutableUnitAdmissionError::InvalidEnvelope)?;
    envelope
        .manifest
        .validate_for(&supported_sections)
        .map_err(ExecutableUnitAdmissionError::InvalidEnvelope)?;
    if let Some(message) = super::unsupported_executable_revisions(&envelope.manifest.revisions) {
        return Err(ExecutableUnitAdmissionError::ComponentRevisions(message));
    }
    let mir_payload = envelope
        .component(ExecutableComponentKind::Mir)
        .ok_or(ExecutableUnitAdmissionError::MissingComponent("MIR"))?;
    let mir = serde_json::from_slice(&mir_payload.bytes)
        .map_err(ExecutableUnitAdmissionError::InvalidMir)?;
    let analysis_payload = envelope
        .component(ExecutableComponentKind::Analysis)
        .ok_or(ExecutableUnitAdmissionError::MissingComponent("analysis"))?;
    let analysis: runmat_mir::analysis::AnalysisStore =
        serde_json::from_slice(&analysis_payload.bytes).map_err(|error| {
            ExecutableUnitAdmissionError::SourceBinding(format!(
                "invalid analysis component: {error}"
            ))
        })?;
    analysis.revision.validate_current().map_err(|error| {
        ExecutableUnitAdmissionError::SourceBinding(format!("invalid analysis revision: {error}"))
    })?;
    let bytecode_payload = envelope
        .component(ExecutableComponentKind::Bytecode)
        .ok_or(ExecutableUnitAdmissionError::MissingComponent("bytecode"))?;
    let bytecode: Bytecode = serde_json::from_slice(&bytecode_payload.bytes)
        .map_err(ExecutableUnitAdmissionError::InvalidBytecode)?;
    if !bytecode.bound_functions.is_empty()
        || !bytecode.function_registry.functions.is_empty()
        || bytecode.layout.is_some()
    {
        return Err(ExecutableUnitAdmissionError::DuplicateComponentAuthority);
    }
    let registry_payload = envelope
        .component(ExecutableComponentKind::FunctionRegistry)
        .ok_or(ExecutableUnitAdmissionError::MissingComponent(
            "function registry",
        ))?;
    let mut registry: FunctionRegistry = serde_json::from_slice(&registry_payload.bytes)
        .map_err(ExecutableUnitAdmissionError::InvalidRegistry)?;
    crate::validate_compiled_parallel_contracts(
        &bytecode,
        &registry,
        &envelope.manifest.parallel,
        None,
    )
    .map_err(ExecutableUnitAdmissionError::SourceBinding)?;
    validate_class_method_roles(&bytecode, &registry)?;
    let layout_payload = envelope
        .component(ExecutableComponentKind::VmLayout)
        .ok_or(ExecutableUnitAdmissionError::MissingComponent("VM layout"))?;
    let layout = serde_json::from_slice(&layout_payload.bytes)
        .map_err(ExecutableUnitAdmissionError::InvalidLayout)?;
    let source_map_payload = envelope
        .component(ExecutableComponentKind::SourceMap)
        .ok_or(ExecutableUnitAdmissionError::MissingComponent("source map"))?;
    let source_map = runmat_execution::ExecutableSourceMapV2::decode(&source_map_payload.bytes)
        .map_err(ExecutableUnitAdmissionError::InvalidSourceMap)?;
    let entrypoint = entrypoint::admit(
        &envelope,
        &mir,
        &analysis,
        &bytecode,
        &registry,
        &layout,
        &source_map,
    )?;
    let (static_imports, linked_native_imports) =
        static_import::admit(&envelope, &bytecode, &registry)?;
    // The validated MIR, registry and layout share this admitted unit. Wire
    // reconstruction cannot supply a local compiler spawn certificate.
    for body in registry.functions.values_mut() {
        body.spawn_contract = runmat_mir::analysis::function_spawn_safety(&mir, body.function)
            .map(crate::bytecode::program::CompilerSpawnContract::issue);
    }
    Ok(AdmittedExecutableUnit {
        envelope,
        canonical_v3: Arc::from(bytes),
        bytecode,
        registry,
        layout,
        source_map,
        entrypoint,
        static_imports,
        linked_native_imports,
    })
}

/// A sealed class registration may refer to a method body in this unit or to
/// a separately selected dependency. When a body for the typed class/member
/// is local, its role and static mode must agree with the registration before
/// the unit can issue runtime effects. This consistency check does not select
/// a body by display name; selected Graph admission proves the exact method
/// unit, definition lease, and source owner separately.
fn validate_class_method_roles(
    bytecode: &Bytecode,
    registry: &FunctionRegistry,
) -> Result<(), ExecutableUnitAdmissionError> {
    use runmat_types::{ClassIdentity, ClassMethodRole, MethodName};
    use std::collections::HashMap;

    let mut roles = HashMap::<(ClassIdentity, MethodName), (ClassMethodRole, bool)>::new();
    for body in registry.functions.values() {
        let Some(owner) = body.class_method_owner.as_ref() else {
            continue;
        };
        let key = (owner.declaring_class.clone(), owner.method.clone());
        let actual = (owner.role, owner.is_static);
        if roles
            .insert(key, actual)
            .is_some_and(|prior| prior != actual)
        {
            return Err(ExecutableUnitAdmissionError::SourceBinding(
                "class method bodies disagree on their typed role".into(),
            ));
        }
    }
    for instruction in bytecode.instructions.iter().chain(
        registry
            .functions
            .values()
            .flat_map(|body| body.instructions.iter()),
    ) {
        let crate::Instr::RegisterClass { name, methods, .. } = instruction else {
            continue;
        };
        for method in methods {
            let key = (name.clone(), method.name.clone());
            if roles.get(&key).is_some_and(|(role, is_static)| {
                role != &method.role || is_static != &method.is_static
            }) {
                return Err(ExecutableUnitAdmissionError::SourceBinding(
                    "class registration method role or static mode differs from a locally admitted class-method body".into(),
                ));
            }
        }
    }
    Ok(())
}
