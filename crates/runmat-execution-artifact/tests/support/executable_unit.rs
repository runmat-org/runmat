use std::collections::BTreeSet;

use runmat_execution::{
    Digest, ExecutableComponentDescriptor, ExecutableComponentKind, ExecutableComponentPayload,
    ExecutableComponentRevisions, ExecutableEntrypointKind, ExecutableIdentity,
    ExecutableUnitEnvelope, ExecutableUnitManifest, ProgramRevision,
    EXECUTABLE_UNIT_SCHEMA_VERSION,
};
use runmat_execution_artifact::ProgramBuildRecipe;
use runmat_types::{
    CapabilityRequirement, CapabilitySet, CollectiveContract, CollectiveId, CollectiveOperation,
    DistributedValueContract, DistributedValueId, DistributionScheme, ForeignAffinity,
    ForeignCapability, ForeignLifetime, ForeignOwnership, ForeignRequirement, ForeignTypeIdentity,
    InteropManifest, LabCount, LabRank, ParallelAccess, ParallelManifest, ParallelRandomnessPolicy,
    ParallelRegionId, ParallelVariableContract, ParallelVariableRole, ParforContract,
    ProgramFunctionId, ProgramPointId, ProgramSourceId, ProgramSpan, RegionContract, RegionId,
    RegionProvenance, RegionValueId, SpmdContract, SpmdLabRequirement, ValueFact,
    WasmInteropPolicy,
};

pub fn recipe(mut recipe: ProgramBuildRecipe) -> ProgramBuildRecipe {
    recipe.entrypoint = "0".into();
    recipe.target_profile = "portable-executable-unit-v3".into();
    recipe
}

pub fn bytes(revision: ProgramRevision) -> Vec<u8> {
    let payloads = ExecutableComponentKind::REQUIRED
        .into_iter()
        .map(|kind| {
            ExecutableComponentPayload::new(kind, format!("{kind:?}-fixture").into_bytes()).unwrap()
        })
        .collect::<Vec<_>>();
    let components = payloads
        .iter()
        .map(|payload| {
            ExecutableComponentDescriptor::from_payload(payload.kind, 1, &payload.bytes).unwrap()
        })
        .collect();
    let catalog_fingerprint = *revision.catalog_fingerprint();
    let function = ProgramFunctionId(0);
    let parfor = ParallelRegionId(RegionId {
        function,
        ordinal: 1,
    });
    let spmd = ParallelRegionId(RegionId {
        function,
        ordinal: 2,
    });
    let loop_variable = RegionValueId { function, local: 0 };
    let distributed = DistributedValueId {
        function,
        ordinal: 0,
    };
    ExecutableUnitEnvelope::new(
        ExecutableUnitManifest {
            schema_version: EXECUTABLE_UNIT_SCHEMA_VERSION,
            identity: ExecutableIdentity {
                program: revision,
                root_package: "artifact-fixture@1.0.0".into(),
                entrypoint: "main".into(),
                entrypoint_function: function,
                entrypoint_kind: ExecutableEntrypointKind::Function,
                source_digest: Digest::sha256(b"src/main.m"),
            },
            revisions: ExecutableComponentRevisions {
                catalog_schema: 1,
                catalog_fingerprint,
                contract_schema: 1,
                contract_fingerprint: Digest::sha256(b"contracts-v1"),
                analysis_schema: 1,
                mir_schema: 1,
                bytecode_schema: 1,
                vm_layout_schema: 1,
                function_registry_schema: 1,
                source_map_schema: 1,
                region_schema: runmat_types::REGION_CONTRACT_SCHEMA_VERSION,
                interop_schema: runmat_types::INTEROP_MANIFEST_SCHEMA_VERSION,
                parallel_schema: runmat_types::PARALLEL_MANIFEST_SCHEMA_VERSION,
            },
            components,
            capabilities: CapabilitySet(BTreeSet::from([
                CapabilityRequirement::ForeignRuntime,
                CapabilityRequirement::ParallelRuntime,
                CapabilityRequirement::DistributedRuntime,
            ])),
            regions: vec![region(parfor.0), region(spmd.0)],
            interop: InteropManifest {
                schema_version: runmat_types::INTEROP_MANIFEST_SCHEMA_VERSION,
                foreign_types: vec![ForeignRequirement {
                    type_identity: ForeignTypeIdentity {
                        family: "java".into(),
                        name: "java.lang.Object".into(),
                        version: 1,
                    },
                    ownership: ForeignOwnership::Shared,
                    affinity: ForeignAffinity::OriginProcess,
                    lifetime: ForeignLifetime::Session,
                    capabilities: vec![ForeignCapability::Invoke],
                    wasm: WasmInteropPolicy::HostBridge,
                }],
                adapters: Vec::new(),
            },
            parallel: ParallelManifest {
                schema_version: runmat_types::PARALLEL_MANIFEST_SCHEMA_VERSION,
                parfor_regions: vec![ParforContract {
                    id: parfor,
                    loop_variable,
                    iterable: dynamic_fact(),
                    variables: vec![ParallelVariableContract {
                        value: loop_variable,
                        role: ParallelVariableRole::Loop,
                        access: ParallelAccess::ReadWrite,
                        fact: dynamic_fact(),
                        transferable: true,
                    }],
                    maximum_workers: Some(LabCount(4)),
                    capabilities: Default::default(),
                    randomness: ParallelRandomnessPolicy::DeterministicSubstreams,
                }],
                spmd_regions: vec![SpmdContract {
                    id: spmd,
                    labs: SpmdLabRequirement::Range {
                        minimum: LabCount(1),
                        maximum: LabCount(4),
                    },
                    captures: Vec::new(),
                    capabilities: Default::default(),
                }],
                distributed_values: vec![DistributedValueContract {
                    id: distributed,
                    value: dynamic_fact(),
                    scheme: DistributionScheme::Block { dimension: 0 },
                    owner_region: spmd,
                    materializable: true,
                }],
                collectives: vec![CollectiveContract {
                    id: CollectiveId {
                        region: spmd,
                        ordinal: 0,
                    },
                    operation: CollectiveOperation::Broadcast {
                        input: distributed,
                        output: distributed,
                        root: LabRank(0),
                    },
                }],
            },
            optional_sections: Vec::new(),
        },
        payloads,
    )
    .unwrap()
    .canonical_bytes()
    .unwrap()
}

fn dynamic_fact() -> ValueFact {
    ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue)
}

fn region(id: RegionId) -> RegionContract {
    RegionContract {
        schema_version: runmat_types::REGION_CONTRACT_SCHEMA_VERSION,
        id,
        source: ProgramSourceId(0),
        span: ProgramSpan {
            start: u64::from(id.ordinal),
            end: u64::from(id.ordinal) + 1,
        },
        entry: ProgramPointId {
            function: id.function,
            block: id.ordinal,
            position: 0,
        },
        exits: Vec::new(),
        live_in: Vec::new(),
        live_out: Vec::new(),
        value_facts: Vec::new(),
        effects: Default::default(),
        capabilities: Default::default(),
        guards: Vec::new(),
        provenance: RegionProvenance::Inferred,
    }
}
