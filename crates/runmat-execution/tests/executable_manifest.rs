use std::collections::BTreeSet;

use runmat_execution::{
    Digest, ExecutableComponentDescriptor, ExecutableComponentKind, ExecutableComponentPayload,
    ExecutableComponentRevisions, ExecutableEntrypointKind, ExecutableIdentity,
    ExecutableOptionalSection, ExecutableSectionSupport, ExecutableUnitEnvelope,
    ExecutableUnitManifest, ProgramEnvironment, ProgramRevision, SectionRequirement,
    EXECUTABLE_UNIT_SCHEMA_VERSION,
};
use runmat_types::{
    CapabilityRequirement, CapabilitySet, ForeignAdapterContractReference, ForeignAffinity,
    ForeignCapability, ForeignLifetime, ForeignOwnership, ForeignRequirement, ForeignTypeIdentity,
    InteropManifest, LabCount, ParallelAccess, ParallelManifest, ParallelRandomnessPolicy,
    ParallelRegionId, ParallelVariableContract, ParallelVariableRole, ParforContract,
    PlannedForeignAdapter, ProgramFunctionId, ProgramPointId, ProgramSourceId, ProgramSpan,
    RegionContract, RegionId, RegionProvenance, RegionValueId, SpmdContract, SpmdLabRequirement,
    ValueFact, ValueKindFact, WasmInteropPolicy, INTEROP_MANIFEST_SCHEMA_VERSION,
    PARALLEL_MANIFEST_SCHEMA_VERSION, REGION_CONTRACT_SCHEMA_VERSION,
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
    let revisions = ExecutableComponentRevisions {
        catalog_schema: 1,
        catalog_fingerprint,
        contract_schema: 1,
        contract_fingerprint: Digest::sha256(b"contracts"),
        analysis_schema: 1,
        mir_schema: 1,
        bytecode_schema: 1,
        vm_layout_schema: 1,
        function_registry_schema: 1,
        source_map_schema: 1,
        region_schema: REGION_CONTRACT_SCHEMA_VERSION,
        interop_schema: INTEROP_MANIFEST_SCHEMA_VERSION,
        parallel_schema: PARALLEL_MANIFEST_SCHEMA_VERSION,
    };
    let components = component_payloads()
        .iter()
        .map(|payload| {
            ExecutableComponentDescriptor::from_payload(payload.kind, 1, &payload.bytes).unwrap()
        })
        .collect();
    ExecutableUnitManifest {
        schema_version: EXECUTABLE_UNIT_SCHEMA_VERSION,
        identity: ExecutableIdentity {
            program,
            root_package: "example@1.0.0".into(),
            entrypoint: "main".into(),
            entrypoint_function: Some(ProgramFunctionId(0)),
            entrypoint_kind: ExecutableEntrypointKind::Function,
            declaration_class: None,
            source_digest: Digest::sha256(b"main.m"),
        },
        revisions,
        components,
        dependencies: Vec::new(),
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
            adapter_contracts: vec![
                ForeignAdapterContractReference::planned(PlannedForeignAdapter::DotNet),
                ForeignAdapterContractReference::planned(PlannedForeignAdapter::WindowsCom),
            ],
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
                    transferability: runmat_types::ParallelTransferability::ProvenTransferable,
                }],
                maximum_workers: Some(LabCount(4)),
                effects: Default::default(),
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
                outputs: Vec::new(),
                effects: Default::default(),
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

fn component_payloads() -> Vec<ExecutableComponentPayload> {
    ExecutableComponentKind::REQUIRED
        .into_iter()
        .map(|kind| {
            ExecutableComponentPayload::new(kind, format!("{kind:?}-v1").into_bytes()).unwrap()
        })
        .collect()
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
    // Includes the explicit original dependency inventory in the manifest.
    assert_eq!(
        manifest().cache_key().unwrap().to_string(),
        "sha256:25e3e3b15dc6572dbc7eaf9476787f7130d459175ade2a21f23bc9cd4441cd9d"
    );
}

#[test]
fn declaration_only_entrypoint_has_no_callable_ordinal() {
    let class = runmat_types::ClassIdentity::new("Base").unwrap();
    let mut declaration = manifest();
    declaration.identity.entrypoint = class.display_name().into();
    declaration.identity.entrypoint_function = None;
    declaration.identity.entrypoint_kind = ExecutableEntrypointKind::DeclarationOnly;
    declaration.identity.declaration_class = Some(class.clone());
    declaration
        .validate()
        .expect("typed declaration-only manifest");
    assert_eq!(
        declaration.identity.validated_entrypoint().unwrap(),
        runmat_execution::ValidatedExecutableEntrypoint::DeclarationOnly(class)
    );
    let bytes = declaration.canonical_bytes().unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(value["identity"].get("entrypoint_function").is_none());

    let mut fabricated = declaration.clone();
    fabricated.identity.entrypoint_function = Some(ProgramFunctionId(0));
    assert!(fabricated.validate().is_err());
    let mut mismatched = declaration.clone();
    mismatched.identity.entrypoint = "Other".into();
    assert!(mismatched.validate().is_err());
    let mut missing_class = declaration.clone();
    missing_class.identity.declaration_class = None;
    assert!(missing_class.validate().is_err());
    let mut old_schema = declaration;
    old_schema.schema_version = EXECUTABLE_UNIT_SCHEMA_VERSION - 1;
    assert!(old_schema.validate().is_err());
}

#[test]
fn complete_envelope_round_trips_and_binds_every_payload() {
    let envelope = ExecutableUnitEnvelope::new(manifest(), component_payloads()).unwrap();
    let bytes = envelope.canonical_bytes().unwrap();
    assert_eq!(
        ExecutableUnitEnvelope::from_canonical_bytes(&bytes).unwrap(),
        envelope
    );
    assert_eq!(envelope.cache_key().unwrap(), Digest::sha256(bytes));

    let mut tampered = envelope.clone();
    tampered.payloads[0].bytes.push(0);
    assert!(tampered.validate().is_err());

    let mut reordered = envelope;
    reordered.payloads.swap(0, 1);
    assert!(reordered.validate().is_err());
}

fn source_map_v2_envelope(entries: serde_json::Value) -> ExecutableUnitEnvelope {
    let mut manifest = manifest();
    manifest.revisions.source_map_schema = 2;
    let mut payloads = component_payloads();
    let source_index = payloads
        .iter()
        .position(|payload| payload.kind == ExecutableComponentKind::SourceMap)
        .unwrap();
    let bytes = serde_json::to_vec(&serde_json::json!({ "entries": entries })).unwrap();
    payloads[source_index] =
        ExecutableComponentPayload::new(ExecutableComponentKind::SourceMap, bytes).unwrap();
    manifest.components[source_index] = ExecutableComponentDescriptor::from_payload(
        ExecutableComponentKind::SourceMap,
        2,
        &payloads[source_index].bytes,
    )
    .unwrap();
    ExecutableUnitEnvelope {
        schema_version: EXECUTABLE_UNIT_SCHEMA_VERSION,
        manifest,
        payloads,
        dependency_products: Vec::new(),
    }
}

fn source_entry(ordinal: usize, text: &str) -> serde_json::Value {
    serde_json::json!({
        "source_id": ordinal,
        "owner_identity": "example@1.0.0",
        "relative_path": format!("source-{ordinal}.m"),
        "content_revision": Digest::sha256(text.as_bytes()),
        "display_name": format!("source-{ordinal}.m"),
        "text": text,
    })
}

#[test]
fn imported_v2_source_map_checks_exact_text_after_component_digest() {
    let valid = source_map_v2_envelope(serde_json::json!([source_entry(0, "x = 1;\n")]));
    let bytes = valid.canonical_bytes().unwrap();
    assert_eq!(
        ExecutableUnitEnvelope::from_canonical_bytes(&bytes).unwrap(),
        valid
    );

    let mut stale = source_entry(0, "x = 1;\n");
    stale["text"] = serde_json::json!("x = 2;\n");
    let forged = source_map_v2_envelope(serde_json::json!([stale]));
    let imported = serde_json::to_vec(&forged).unwrap();
    assert!(ExecutableUnitEnvelope::from_canonical_bytes(&imported).is_err());
    assert!(forged.validate().is_err());
}

#[test]
fn imported_v2_source_map_requires_dense_unique_ordinals() {
    for entries in [
        serde_json::json!([source_entry(0, "a"), source_entry(0, "b")]),
        serde_json::json!([source_entry(0, "a"), source_entry(2, "b")]),
        serde_json::json!([source_entry(1, "a")]),
    ] {
        let forged = source_map_v2_envelope(entries);
        let imported = serde_json::to_vec(&forged).unwrap();
        assert!(ExecutableUnitEnvelope::from_canonical_bytes(&imported).is_err());
    }
}

#[test]
fn manifest_requires_each_component_once_with_matching_schema() {
    let mut missing = manifest();
    missing.components.pop();
    assert!(missing.validate().is_err());

    let mut reordered = manifest();
    reordered.components.swap(0, 1);
    assert!(reordered.validate().is_err());

    let mut wrong_schema = manifest();
    wrong_schema.components[0].schema_version += 1;
    assert!(wrong_schema.validate().is_err());
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

    let available = CapabilitySet(BTreeSet::from([
        CapabilityRequirement::ForeignRuntime,
        CapabilityRequirement::ParallelRuntime,
        CapabilityRequirement::DistributedRuntime,
    ]));
    baseline.validate_capabilities_for(&available).unwrap();
    let unavailable = CapabilitySet(BTreeSet::from([
        CapabilityRequirement::ParallelRuntime,
        CapabilityRequirement::DistributedRuntime,
    ]));
    assert!(baseline.validate_capabilities_for(&unavailable).is_err());

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

#[path = "executable_manifest/dependencies.rs"]
mod dependencies;
