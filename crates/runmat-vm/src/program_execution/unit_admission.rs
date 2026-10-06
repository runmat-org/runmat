use runmat_execution::{ExecutableComponentKind, ExecutableUnitEnvelope};

use crate::{Bytecode, FunctionRegistry, VmAssemblyLayout};

/// Fully decoded executable unit whose component identities and revisions have
/// been validated, but whose portable source ordinals have not yet issued
/// runtime-local source leases.
#[derive(Clone, Debug)]
pub struct AdmittedExecutableUnit {
    envelope: ExecutableUnitEnvelope,
    bytecode: Bytecode,
    registry: FunctionRegistry,
    layout: VmAssemblyLayout,
}

/// Executable unit bound to one exact runtime source owner. The runtime must
/// stay with the registry: its source leases are part of executable authority.
#[derive(Debug)]
pub struct RuntimeBoundExecutableUnit {
    bytecode: Bytecode,
    registry: FunctionRegistry,
    layout: VmAssemblyLayout,
    runtime: runmat_runtime::context::RuntimeContext,
}

impl AdmittedExecutableUnit {
    pub fn envelope(&self) -> &ExecutableUnitEnvelope {
        &self.envelope
    }

    pub fn registry(&self) -> &FunctionRegistry {
        &self.registry
    }

    /// Resolve the manifest entrypoint to the exact immutable definition
    /// origin admitted with this unit. A display name or a matching numeric
    /// ordinal alone is not executable authority.
    pub fn entrypoint_origin(
        &self,
    ) -> Result<runmat_execution::FunctionBindingOrigin, ExecutableUnitAdmissionError> {
        let identity = &self.envelope.manifest.identity;
        let function = match identity.entrypoint_kind {
            runmat_execution::ExecutableEntrypointKind::Function => self
                .registry
                .resolve_name(&identity.entrypoint)
                .and_then(|function| self.registry.get(function)),
            runmat_execution::ExecutableEntrypointKind::Script => self
                .registry
                .functions
                .values()
                .find(|function| function.portable_function == identity.entrypoint_function),
        }
        .ok_or_else(|| {
            ExecutableUnitAdmissionError::SourceBinding(
                "executable entrypoint is absent from its admitted function registry".into(),
            )
        })?;
        if function.portable_function != identity.entrypoint_function {
            return Err(ExecutableUnitAdmissionError::SourceBinding(
                "executable entrypoint identity disagrees with its admitted function body".into(),
            ));
        }
        let expected = runmat_execution::FunctionBindingOrigin::from_revision(
            &identity.program,
            function.portable_function,
        )
        .map_err(|error| {
            ExecutableUnitAdmissionError::SourceBinding(format!(
                "executable entrypoint has an invalid immutable origin: {error}"
            ))
        })?;
        if function.binding_origin != Some(expected) {
            return Err(ExecutableUnitAdmissionError::SourceBinding(
                "executable entrypoint lies outside its admitted revision".into(),
            ));
        }
        Ok(expected)
    }

    pub fn bind_runtime_sources(
        mut self,
        runtime: runmat_runtime::context::RuntimeContext,
    ) -> Result<RuntimeBoundExecutableUnit, ExecutableUnitAdmissionError> {
        let runtime = super::unit_sources::bind_unit_sources(
            &self.envelope,
            &mut self.bytecode,
            &mut self.registry,
            runtime,
        )
        .map_err(ExecutableUnitAdmissionError::SourceBinding)?;
        self.bytecode.layout = Some(self.layout.clone());
        Ok(RuntimeBoundExecutableUnit {
            bytecode: self.bytecode,
            registry: self.registry,
            layout: self.layout,
            runtime,
        })
    }
}

impl RuntimeBoundExecutableUnit {
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
            bytecode: Bytecode::with_instructions(Vec::new(), 0),
            registry,
            layout: VmAssemblyLayout {
                functions: Default::default(),
                entrypoints: Default::default(),
                storage_bindings: Default::default(),
            },
            runtime,
        }
    }
}

#[derive(Debug)]
pub enum ExecutableUnitAdmissionError {
    InvalidHeader(runmat_execution::ContractError),
    CompilerRevision(runmat_execution::ContractError),
    ComponentRevisions(String),
    InvalidEnvelope(runmat_execution::ContractError),
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
            | Self::InvalidEnvelope(error) => Some(error),
            Self::InvalidBytecode(error)
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
    if let Some(message) = super::unsupported_executable_revisions(&envelope.manifest.revisions) {
        return Err(ExecutableUnitAdmissionError::ComponentRevisions(message));
    }
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
    let registry = serde_json::from_slice(&registry_payload.bytes)
        .map_err(ExecutableUnitAdmissionError::InvalidRegistry)?;
    let layout_payload = envelope
        .component(ExecutableComponentKind::VmLayout)
        .ok_or(ExecutableUnitAdmissionError::MissingComponent("VM layout"))?;
    let layout = serde_json::from_slice(&layout_payload.bytes)
        .map_err(ExecutableUnitAdmissionError::InvalidLayout)?;
    let admitted = AdmittedExecutableUnit {
        envelope,
        bytecode,
        registry,
        layout,
    };
    admitted.entrypoint_origin()?;
    Ok(admitted)
}
