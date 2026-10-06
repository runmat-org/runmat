use serde::Serialize;

use super::ExecutableUnit;

const SOURCE_MAP_SCHEMA_VERSION: u16 = 2;

#[derive(Serialize)]
struct PortableSourceMap<'a> {
    entries: Vec<PortableSourceEntry<'a>>,
}

#[derive(Serialize)]
struct PortableSourceEntry<'a> {
    source_id: runmat_types::ProgramSourceId,
    owner_identity: &'a str,
    relative_path: &'a str,
    content_revision: &'a runmat_execution::Digest,
    display_name: &'a str,
    text: &'a str,
}

impl ExecutableUnit {
    /// Build the complete physical-path-free executable product for package,
    /// test, browser, remote, native-codegen, and cache consumers.
    pub fn portable_envelope(&self) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        self.portable_envelope_for(None)
    }

    /// Build the complete product with an optional explicitly selected function.
    /// A missing explicit function is an error; implicit selection falls back to
    /// a script only when the source does not define its stem as a function.
    pub fn portable_envelope_for(
        &self,
        preferred_function: Option<&str>,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        self.portable_envelope_for_with_interop(
            preferred_function,
            runmat_types::InteropManifest::empty(),
        )
    }

    /// Build a complete product with an explicit foreign-interoperability
    /// contract. Package and adapter composition use this entrypoint once
    /// foreign requirements have been resolved; ordinary source compilation
    /// records the deterministic empty contract.
    pub fn portable_envelope_for_with_interop(
        &self,
        preferred_function: Option<&str>,
        interop: runmat_types::InteropManifest,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        self.portable_envelope_with_entrypoint(interop, |unit| {
            super::portable_entrypoint::resolve(unit, preferred_function)
        })
    }

    pub(crate) fn portable_envelope_for_current_definition(
        &self,
        current: &std::collections::HashSet<runmat_hir::FunctionId>,
        primary_source: runmat_types::ProgramSourceId,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        self.portable_envelope_with_entrypoint(runmat_types::InteropManifest::empty(), |unit| {
            super::portable_entrypoint::resolve_current_definition(unit, current, primary_source)
        })
    }

    fn portable_envelope_with_entrypoint(
        &self,
        interop: runmat_types::InteropManifest,
        resolve: impl FnOnce(&Self) -> Result<super::portable_entrypoint::PortableEntrypoint, String>,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        let interop =
            runmat_types::InteropManifest::merge([self.inferred_interop_manifest(), interop])
                .map_err(|error| format!("{}: {}", error.path, error.message))?;
        let payloads = self.component_payloads()?;
        let revisions = self.component_revisions()?;
        let components = component_descriptors(&payloads, &revisions)?;
        let entrypoint = resolve(self)?;
        let mut capabilities = runmat_types::CapabilitySet(
            self.analysis()
                .functions
                .iter()
                .flat_map(|function| function.capabilities.0.iter().copied())
                .collect(),
        );
        for adapter in &interop.adapters {
            capabilities
                .0
                .extend(adapter.capabilities.0.iter().copied());
        }
        if !interop.foreign_types.is_empty() || !interop.adapters.is_empty() {
            capabilities
                .0
                .insert(runmat_types::CapabilityRequirement::ForeignRuntime);
        }
        let manifest = runmat_execution::ExecutableUnitManifest {
            schema_version: runmat_execution::EXECUTABLE_UNIT_SCHEMA_VERSION,
            identity: runmat_execution::ExecutableIdentity {
                program: self.revision().program_revision.clone(),
                root_package: self.source().owner_identity.clone(),
                entrypoint: entrypoint.name,
                entrypoint_function: entrypoint.function,
                entrypoint_kind: entrypoint.kind,
                source_digest: self
                    .revision()
                    .source_digest
                    .parse()
                    .map_err(|error: runmat_execution::ContractError| error.to_string())?,
            },
            revisions,
            components,
            capabilities,
            regions: self
                .analysis()
                .regions
                .iter()
                .map(|region| region.contract.clone())
                .collect(),
            interop,
            parallel: self.analysis().parallel.clone(),
            optional_sections: Vec::new(),
        };
        runmat_execution::ExecutableUnitEnvelope::new(manifest, payloads)
            .map_err(|error| error.to_string())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn native_product_digest(&self) -> Result<runmat_execution::Digest, String> {
        let revisions = self.component_revisions()?;
        let components = component_descriptors(&self.component_payloads()?, &revisions)?;
        // VM layout, registry and bytecode also own the retained interpreter
        // capture, resume locations and coverage instrumentation.
        Ok(runmat_execution::Digest::sha256(canonical_json(&(
            "runmat-native-compilation-v1",
            revisions,
            components,
            self.binding_names(),
            self.inferred_interop_manifest(),
            &self.source().owner_identity,
            &self.source().relative_path,
        ))?))
    }

    fn inferred_interop_manifest(&self) -> runmat_types::InteropManifest {
        if !self.requires_python_runtime() {
            return runmat_types::InteropManifest::empty();
        }
        runmat_types::InteropManifest {
            schema_version: runmat_types::INTEROP_MANIFEST_SCHEMA_VERSION,
            foreign_types: vec![runmat_types::ForeignRequirement {
                type_identity: runmat_types::ForeignTypeIdentity {
                    family: "python".into(),
                    name: "environment".into(),
                    version: 1,
                },
                ownership: runmat_types::ForeignOwnership::Shared,
                affinity: runmat_types::ForeignAffinity::OriginProcess,
                lifetime: runmat_types::ForeignLifetime::Session,
                capabilities: vec![runmat_types::ForeignCapability::Invoke],
                wasm: runmat_types::WasmInteropPolicy::Reject,
            }],
            adapters: vec![runmat_types::ForeignAdapterRequirement {
                adapter: runmat_types::ForeignAdapterId::new("python").unwrap(),
                minimum_version: 1,
                capabilities: runmat_types::CapabilitySet(std::collections::BTreeSet::from([
                    runmat_types::CapabilityRequirement::ForeignRuntime,
                ])),
                execution_stack: runmat_types::ExecutionStackRequirement::Process,
                artifact_identities: Vec::new(),
            }],
            adapter_contracts: Vec::new(),
        }
    }

    fn component_revisions(
        &self,
    ) -> Result<runmat_execution::ExecutableComponentRevisions, String> {
        let analysis = &self.analysis().revision;
        Ok(runmat_execution::ExecutableComponentRevisions {
            catalog_schema: u16::try_from(analysis.catalog_schema)
                .map_err(|_| "builtin catalog schema exceeds the executable schema".to_string())?,
            catalog_fingerprint: runmat_execution::Digest::from_bytes(analysis.catalog_fingerprint),
            contract_schema: runmat_types::RUNMAT_TYPES_SCHEMA.major,
            contract_fingerprint: runmat_execution::Digest::sha256(format!(
                "runmat-types-contract-v{}.{}",
                runmat_types::RUNMAT_TYPES_SCHEMA.major,
                runmat_types::RUNMAT_TYPES_SCHEMA.minor
            )),
            analysis_schema: analysis.schema_version,
            mir_schema: runmat_mir::MIR_SCHEMA_VERSION,
            bytecode_schema: runmat_vm::BYTECODE_SCHEMA_VERSION,
            vm_layout_schema: runmat_vm::VM_LAYOUT_SCHEMA_VERSION,
            function_registry_schema: runmat_vm::FUNCTION_REGISTRY_SCHEMA_VERSION,
            source_map_schema: SOURCE_MAP_SCHEMA_VERSION,
            region_schema: runmat_types::REGION_CONTRACT_SCHEMA_VERSION,
            interop_schema: runmat_types::INTEROP_MANIFEST_SCHEMA_VERSION,
            parallel_schema: runmat_types::PARALLEL_MANIFEST_SCHEMA_VERSION,
        })
    }

    fn component_payloads(
        &self,
    ) -> Result<Vec<runmat_execution::ExecutableComponentPayload>, String> {
        use runmat_execution::{ExecutableComponentKind as Kind, ExecutableComponentPayload};

        self.source_map().validate_content_revisions()?;

        let source_map = PortableSourceMap {
            entries: self
                .source_map()
                .entries()
                .iter()
                .map(|entry| PortableSourceEntry {
                    source_id: entry.source_id,
                    owner_identity: &entry.owner_identity,
                    relative_path: &entry.relative_path,
                    content_revision: &entry.content_revision,
                    display_name: &entry.display_name,
                    text: &entry.text,
                })
                .collect(),
        };
        let mut bytecode = self.bytecode().clone();
        bytecode.bound_functions.clear();
        bytecode.function_registry = runmat_vm::FunctionRegistry::default();
        bytecode.layout = None;
        bytecode.validate_portable_source_origins()?;
        let mut functions = self.functions().clone();
        functions.validate_portable_source_origins()?;
        for function in functions.functions.values_mut() {
            if function.binding_origin.is_none() {
                function.binding_origin = Some(
                    runmat_execution::FunctionBindingOrigin::from_revision(
                        &self.revision().program_revision,
                        function.portable_function,
                    )
                    .map_err(|error| error.to_string())?,
                );
            }
            function.source_file = function
                .source_id
                .and_then(runmat_vm::BytecodeSourceRef::program)
                .and_then(|ordinal| {
                    self.source_map()
                        .entries()
                        .get(usize::try_from(ordinal.0).ok()?)
                })
                .map(|entry| entry.relative_path.clone());
        }
        [
            (Kind::Mir, canonical_json(self.mir())?),
            (Kind::Analysis, canonical_json(self.analysis())?),
            (Kind::Bytecode, canonical_json(&bytecode)?),
            (Kind::VmLayout, canonical_json(self.vm_layout())?),
            (Kind::FunctionRegistry, canonical_json(&functions)?),
            (Kind::SourceMap, canonical_json(&source_map)?),
        ]
        .into_iter()
        .map(|(kind, bytes)| {
            ExecutableComponentPayload::new(kind, bytes).map_err(|error| error.to_string())
        })
        .collect()
    }
}

pub(super) fn canonical_json(value: &impl Serialize) -> Result<Vec<u8>, String> {
    let value = serde_json::to_value(value).map_err(|error| error.to_string())?;
    serde_json::to_vec(&value).map_err(|error| error.to_string())
}

fn component_descriptors(
    payloads: &[runmat_execution::ExecutableComponentPayload],
    revisions: &runmat_execution::ExecutableComponentRevisions,
) -> Result<Vec<runmat_execution::ExecutableComponentDescriptor>, String> {
    payloads
        .iter()
        .map(|payload| {
            runmat_execution::ExecutableComponentDescriptor::from_payload(
                payload.kind,
                payload.kind.schema_version(revisions),
                &payload.bytes,
            )
            .map_err(|error| error.to_string())
        })
        .collect()
}

trait ComponentSchema {
    fn schema_version(self, revisions: &runmat_execution::ExecutableComponentRevisions) -> u16;
}

impl ComponentSchema for runmat_execution::ExecutableComponentKind {
    fn schema_version(self, revisions: &runmat_execution::ExecutableComponentRevisions) -> u16 {
        match self {
            Self::Mir => revisions.mir_schema,
            Self::Analysis => revisions.analysis_schema,
            Self::Bytecode => revisions.bytecode_schema,
            Self::VmLayout => revisions.vm_layout_schema,
            Self::FunctionRegistry => revisions.function_registry_schema,
            Self::SourceMap => revisions.source_map_schema,
        }
    }
}
