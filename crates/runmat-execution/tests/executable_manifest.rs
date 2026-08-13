use std::collections::BTreeSet;

use runmat_execution::{
    Digest, ExecutableComponentRevisions, ExecutableIdentity, ExecutableOptionalSection,
    ExecutableSectionSupport, ExecutableUnitManifest, ProgramEnvironment, ProgramRevision,
    SectionRequirement, EXECUTABLE_UNIT_SCHEMA_VERSION,
};
use runmat_types::{
    CapabilityRequirement, CapabilitySet, ForeignAffinity, ForeignCapability, ForeignLifetime,
    ForeignOwnership, ForeignRequirement, ForeignTypeIdentity, InteropManifest, LabCount,
    ParallelAccess, ParallelManifest, ParallelRandomnessPolicy, ParallelRegionId,
    ParallelVariableContract, ParallelVariableRole, ParforContract, ProgramFunctionId,
    ProgramPointId, ProgramSourceId, ProgramSpan, RegionContract, RegionId, RegionProvenance,
    RegionValueId, SpmdContract, SpmdLabRequirement, ValueFact, ValueKindFact, WasmInteropPolicy,
    INTEROP_MANIFEST_SCHEMA_VERSION, PARALLEL_MANIFEST_SCHEMA_VERSION,
    REGION_CONTRACT_SCHEMA_VERSION,
};

fn program() -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(b"graph"),
        Digest::sha256(b"sources"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"runtime"),
            Digest::sha256(b"catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
}

fn region(ordinal: u32) -> RegionContract {
    let function = ProgramFunctionId(0);
    RegionContract {
        schema_version: REGION_CONTRACT_SCHEMA_VERSION,
        id: RegionId { function, ordinal },
        source: ProgramSourceId(0),
        span: ProgramSpan {
            start: u64::from(ordinal),
            end: u64::from(ordinal) + 1,
        },
        entry: ProgramPointId {
            function,
            block: ordinal,
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

fn manifest() -> ExecutableUnitManifest {
    let program = program();
    let catalog_fingerprint = *program.catalog_fingerprint();
    let parfor = ParallelRegionId(RegionId {
        function: ProgramFunctionId(0),
        ordinal: 1,
    });
    let spmd = ParallelRegionId(RegionId {
        function: ProgramFunctionId(0),
        ordinal: 2,
    });
    let loop_variable = RegionValueId {
        function: ProgramFunctionId(0),
        local: 0,
    };
    ExecutableUnitManifest {
        schema_version: EXECUTABLE_UNIT_SCHEMA_VERSION,
        identity: ExecutableIdentity {
            program,
            root_package: "example@1.0.0".into(),
            entrypoint: "main".into(),
            entrypoint_function: ProgramFunctionId(0),
            source_digest: Digest::sha256(b"main.m"),
        },
        revisions: ExecutableComponentRevisions {
            catalog_schema: 1,
            catalog_fingerprint,
            contract_schema: 1,
            contract_fingerprint: Digest::sha256(b"contracts"),
            analysis_schema: 1,
            mir_schema: 1,
            bytecode_schema: 1,
            region_schema: REGION_CONTRACT_SCHEMA_VERSION,
            interop_schema: INTEROP_MANIFEST_SCHEMA_VERSION,
            parallel_schema: PARALLEL_MANIFEST_SCHEMA_VERSION,
        },
        capabilities: CapabilitySet(BTreeSet::from([
            CapabilityRequirement::ForeignRuntime,
            CapabilityRequirement::ParallelRuntime,
            CapabilityRequirement::DistributedRuntime,
        ])),
        regions: vec![region(1), region(2)],
        interop: InteropManifest {
            schema_version: INTEROP_MANIFEST_SCHEMA_VERSION,
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
            schema_version: PARALLEL_MANIFEST_SCHEMA_VERSION,
            parfor_regions: vec![ParforContract {
                id: parfor,
                loop_variable,
                iterable: ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
                variables: vec![ParallelVariableContract {
                    value: loop_variable,
                    role: ParallelVariableRole::Loop,
                    access: ParallelAccess::ReadWrite,
                    fact: ValueFact::scalar(ValueKindFact::Logical),
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
            distributed_values: Vec::new(),
            collectives: Vec::new(),
        },
        optional_sections: vec![ExecutableOptionalSection::new(
            "runmat.profile",
            1,
            SectionRequirement::Optional,
            b"profile-v1".to_vec(),
        )],
    }
}

fn round_trip_vector() {
    let manifest = manifest();
    manifest.validate().unwrap();
    let bytes = manifest.canonical_bytes().unwrap();
    assert_eq!(
        ExecutableUnitManifest::from_canonical_bytes(&bytes).unwrap(),
        manifest
    );
    assert_eq!(manifest.cache_key().unwrap(), Digest::sha256(&bytes));
}

#[test]
fn executable_manifest_round_trips_all_contract_families() {
    round_trip_vector();
    assert_eq!(
        manifest().cache_key().unwrap().to_string(),
        "sha256:4bb3c984c2baf38aa6635a58c43517b5a4e90d3bd476a2b3df864387cedf513b"
    );
}

#[test]
fn optional_sections_are_preserved_but_required_sections_need_support() {
    let mut manifest = manifest();
    let unsupported = ExecutableSectionSupport::default();
    manifest.validate_for(&unsupported).unwrap();

    manifest.optional_sections[0].requirement = SectionRequirement::Required;
    assert!(manifest.validate_for(&unsupported).is_err());
    let supported = ExecutableSectionSupport::new([("runmat.profile".to_string(), 1)]).unwrap();
    manifest.validate_for(&supported).unwrap();
}

#[test]
fn versions_capabilities_digests_and_cache_inputs_are_enforced() {
    let baseline = manifest();
    let baseline_key = baseline.cache_key().unwrap();

    let mut changed = baseline.clone();
    changed.revisions.contract_fingerprint = Digest::sha256(b"contracts-v2");
    assert_ne!(changed.cache_key().unwrap(), baseline_key);

    let mut wrong_version = baseline.clone();
    wrong_version.schema_version += 1;
    assert!(wrong_version.validate().is_err());

    let mut missing_capability = baseline.clone();
    missing_capability
        .capabilities
        .0
        .remove(&CapabilityRequirement::ForeignRuntime);
    assert!(missing_capability.validate().is_err());

    let mut tampered = baseline;
    tampered.optional_sections[0].payload.push(0);
    assert!(tampered.validate().is_err());
}

#[test]
fn decoder_rejects_unknown_core_fields_and_noncanonical_json() {
    let bytes = manifest().canonical_bytes().unwrap();
    let mut value = serde_json::from_slice::<serde_json::Value>(&bytes).unwrap();
    value["future_core_field"] = serde_json::json!(true);
    assert!(serde_json::from_value::<ExecutableUnitManifest>(value).is_err());

    let mut nested_program = serde_json::from_slice::<serde_json::Value>(&bytes).unwrap();
    nested_program["identity"]["program"]["future_program_field"] = serde_json::json!(true);
    assert!(serde_json::from_value::<ExecutableUnitManifest>(nested_program).is_err());

    let mut nested_contract = serde_json::from_slice::<serde_json::Value>(&bytes).unwrap();
    nested_contract["regions"][0]["span"]["future_span_field"] = serde_json::json!(true);
    assert!(serde_json::from_value::<ExecutableUnitManifest>(nested_contract).is_err());

    let mut noncanonical = bytes;
    noncanonical.push(b'\n');
    assert!(ExecutableUnitManifest::from_canonical_bytes(&noncanonical).is_err());
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn wasm_uses_the_same_executable_manifest_vector() {
    round_trip_vector();
}
