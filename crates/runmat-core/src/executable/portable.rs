use serde::Serialize;

use super::ExecutableUnit;

const BYTECODE_SCHEMA_VERSION: u16 = 1;
const VM_LAYOUT_SCHEMA_VERSION: u16 = 1;
const FUNCTION_REGISTRY_SCHEMA_VERSION: u16 = 1;
const SOURCE_MAP_SCHEMA_VERSION: u16 = 1;

#[derive(Serialize)]
struct PortableSourceMap<'a> {
    entries: Vec<PortableSourceEntry<'a>>,
}

#[derive(Serialize)]
struct PortableSourceEntry<'a> {
    source_id: usize,
    owner_identity: &'a str,
    relative_path: &'a str,
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
        let payloads = self.component_payloads()?;
        let revisions = self.component_revisions()?;
        let components = payloads
            .iter()
            .map(|payload| {
                runmat_execution::ExecutableComponentDescriptor::from_payload(
                    payload.kind,
                    payload.kind.schema_version(&revisions),
                    &payload.bytes,
                )
                .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let source_stem = std::path::Path::new(&self.source().relative_path)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("script");
        let selected = preferred_function
            .map(|name| {
                self.functions()
                    .resolve_name(name)
                    .map(|function| (name.to_string(), function))
                    .ok_or_else(|| format!("source does not define requested function '{name}'"))
            })
            .transpose()?
            .or_else(|| {
                self.functions()
                    .resolve_name(source_stem)
                    .map(|function| (source_stem.to_string(), function))
            });
        let (entrypoint, entrypoint_function, entrypoint_kind) =
            if let Some((name, function)) = selected {
                (
                    name,
                    function,
                    runmat_execution::ExecutableEntrypointKind::Function,
                )
            } else if !self.bytecode().instructions.is_empty() {
                (
                    "script".to_string(),
                    runmat_hir::FunctionId(0),
                    runmat_execution::ExecutableEntrypointKind::Script,
                )
            } else {
                return Err("executable has no callable function or script entrypoint".to_string());
            };
        let entrypoint_function = u32::try_from(entrypoint_function.0)
            .map(runmat_types::ProgramFunctionId)
            .map_err(|_| "entrypoint identity exceeds the portable schema".to_string())?;
        let capabilities = runmat_types::CapabilitySet(
            self.analysis()
                .functions
                .iter()
                .flat_map(|function| function.capabilities.0.iter().copied())
                .collect(),
        );
        let manifest = runmat_execution::ExecutableUnitManifest {
            schema_version: runmat_execution::EXECUTABLE_UNIT_SCHEMA_VERSION,
            identity: runmat_execution::ExecutableIdentity {
                program: self.revision().program_revision.clone(),
                root_package: self.source().owner_identity.clone(),
                entrypoint,
                entrypoint_function,
                entrypoint_kind,
                source_digest: self
                    .revision()
                    .source_digest
                    .parse()
                    .map_err(|error: runmat_execution::ContractError| error.to_string())?,
            },
            revisions,
            components,
            capabilities,
            regions: Vec::new(),
            interop: runmat_types::InteropManifest::empty(),
            parallel: runmat_types::ParallelManifest::empty(),
            optional_sections: Vec::new(),
        };
        runmat_execution::ExecutableUnitEnvelope::new(manifest, payloads)
            .map_err(|error| error.to_string())
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
            bytecode_schema: BYTECODE_SCHEMA_VERSION,
            vm_layout_schema: VM_LAYOUT_SCHEMA_VERSION,
            function_registry_schema: FUNCTION_REGISTRY_SCHEMA_VERSION,
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

        let source_map = PortableSourceMap {
            entries: self
                .source_map()
                .entries()
                .iter()
                .map(|entry| PortableSourceEntry {
                    source_id: entry.source_id,
                    owner_identity: &entry.owner_identity,
                    relative_path: &entry.relative_path,
                    display_name: &entry.display_name,
                    text: &entry.text,
                })
                .collect(),
        };
        let mut bytecode = self.bytecode().clone();
        bytecode.bound_functions.clear();
        bytecode.function_registry = runmat_vm::FunctionRegistry::default();
        bytecode.layout = None;
        [
            (Kind::Mir, canonical_json(self.mir())?),
            (Kind::Analysis, canonical_json(self.analysis())?),
            (Kind::Bytecode, canonical_json(&bytecode)?),
            (Kind::VmLayout, canonical_json(self.vm_layout())?),
            (Kind::FunctionRegistry, canonical_json(self.functions())?),
            (Kind::SourceMap, canonical_json(&source_map)?),
        ]
        .into_iter()
        .map(|(kind, bytes)| {
            ExecutableComponentPayload::new(kind, bytes).map_err(|error| error.to_string())
        })
        .collect()
    }
}

fn canonical_json(value: &impl Serialize) -> Result<Vec<u8>, String> {
    let value = serde_json::to_value(value).map_err(|error| error.to_string())?;
    serde_json::to_vec(&value).map_err(|error| error.to_string())
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
