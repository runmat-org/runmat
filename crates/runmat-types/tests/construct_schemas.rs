use runmat_types::{
    CapabilitySet, CollectiveContract, CollectiveId, CollectiveOperation, DeoptimizationPointId,
    DistributedValueContract, DistributedValueId, DistributionScheme, ForeignAdapterRequirement,
    ForeignAffinity, ForeignCapability, ForeignLifetime, ForeignOwnership, ForeignRequirement,
    ForeignTypeIdentity, InteropManifest, LabCount, ParallelAccess, ParallelManifest,
    ParallelRandomnessPolicy, ParallelRegionId, ParallelVariableContract, ParallelVariableRole,
    ParforContract, ProgramFunctionId, ProgramPointId, RegionContract, RegionGuardCondition,
    RegionGuardContract, RegionGuardId, RegionId, RegionProvenance, RegionValueFact, RegionValueId,
    SourceId, Span, SpmdContract, ValueFact, ValueKindFact, WasmInteropPolicy,
    INTEROP_MANIFEST_SCHEMA_VERSION, PARALLEL_MANIFEST_SCHEMA_VERSION,
    REGION_CONTRACT_SCHEMA_VERSION,
};

fn region_id(ordinal: u32) -> RegionId {
    RegionId {
        function: ProgramFunctionId(7),
        ordinal,
    }
}

fn region() -> RegionContract {
    let id = region_id(1);
    let value = RegionValueId {
        function: id.function,
        local: 2,
    };
    RegionContract {
        schema_version: REGION_CONTRACT_SCHEMA_VERSION,
        id,
        source: SourceId(3),
        span: Span { start: 10, end: 20 },
        entry: ProgramPointId {
            function: id.function,
            block: 0,
            position: 0,
        },
        exits: vec![ProgramPointId {
            function: id.function,
            block: 1,
            position: 0,
        }],
        live_in: vec![value],
        live_out: vec![value],
        value_facts: vec![RegionValueFact {
            value,
            fact: ValueFact::scalar(ValueKindFact::Logical),
        }],
        effects: Default::default(),
        capabilities: Default::default(),
        guards: vec![RegionGuardContract {
            id: RegionGuardId {
                region: id,
                ordinal: 0,
            },
            condition: RegionGuardCondition::ValueFact {
                value,
                expected: ValueFact::scalar(ValueKindFact::Logical),
            },
            deopt: DeoptimizationPointId {
                function: id.function,
                ordinal: 0,
            },
        }],
        provenance: RegionProvenance::Inferred,
    }
}

fn interop() -> InteropManifest {
    InteropManifest {
        schema_version: INTEROP_MANIFEST_SCHEMA_VERSION,
        foreign_types: vec![ForeignRequirement {
            type_identity: ForeignTypeIdentity {
                family: "c".into(),
                name: "example.widget".into(),
                version: 1,
            },
            ownership: ForeignOwnership::Owned,
            affinity: ForeignAffinity::OriginProcess,
            lifetime: ForeignLifetime::Session,
            capabilities: vec![ForeignCapability::Invoke, ForeignCapability::Read],
            wasm: WasmInteropPolicy::Reject,
        }],
        adapters: vec![ForeignAdapterRequirement {
            adapter: "mex-c".into(),
            minimum_version: 1,
            capabilities: CapabilitySet::default(),
            artifact_identities: vec!["sha256:abc".into()],
        }],
    }
}

fn parallel() -> ParallelManifest {
    let region = ParallelRegionId(region_id(2));
    let distributed = DistributedValueId {
        function: region.0.function,
        ordinal: 0,
    };
    ParallelManifest {
        schema_version: PARALLEL_MANIFEST_SCHEMA_VERSION,
        parfor_regions: vec![ParforContract {
            id: region,
            loop_variable: RegionValueId {
                function: region.0.function,
                local: 0,
            },
            iterable: ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
            variables: vec![ParallelVariableContract {
                value: RegionValueId {
                    function: region.0.function,
                    local: 0,
                },
                role: ParallelVariableRole::Loop,
                access: ParallelAccess::ReadWrite,
                fact: ValueFact::scalar(ValueKindFact::Logical),
                transferable: true,
            }],
            maximum_workers: Some(LabCount(4)),
            capabilities: CapabilitySet::default(),
            randomness: ParallelRandomnessPolicy::DeterministicSubstreams,
        }],
        spmd_regions: vec![SpmdContract {
            id: ParallelRegionId(region_id(3)),
            minimum_labs: LabCount(1),
            maximum_labs: Some(LabCount(4)),
            captures: Vec::new(),
            capabilities: CapabilitySet::default(),
        }],
        distributed_values: vec![DistributedValueContract {
            id: distributed,
            value: ValueFact::scalar(ValueKindFact::Logical),
            scheme: DistributionScheme::Block { dimension: 0 },
            owner_region: region,
            materializable: true,
        }],
        collectives: vec![CollectiveContract {
            id: CollectiveId { region, ordinal: 0 },
            operation: CollectiveOperation::Broadcast {
                input: distributed,
                output: distributed,
                root: runmat_types::LabRank(0),
            },
        }],
    }
}

fn schema_round_trip_vectors() {
    let region = region();
    region.validate().unwrap();
    assert_eq!(
        serde_json::from_slice::<RegionContract>(&serde_json::to_vec(&region).unwrap()).unwrap(),
        region
    );

    let interop = interop();
    interop.validate().unwrap();
    assert_eq!(
        serde_json::from_slice::<InteropManifest>(&serde_json::to_vec(&interop).unwrap()).unwrap(),
        interop
    );

    let parallel = parallel();
    parallel.validate().unwrap();
    assert_eq!(
        serde_json::from_slice::<ParallelManifest>(&serde_json::to_vec(&parallel).unwrap())
            .unwrap(),
        parallel
    );
}

#[test]
fn construct_schemas_are_canonical_and_portable() {
    schema_round_trip_vectors();
}

#[test]
fn construct_schemas_reject_version_order_and_kind_drift() {
    let mut region = region();
    region.schema_version += 1;
    assert_eq!(region.validate().unwrap_err().path, "region.schema_version");

    let mut interop = interop();
    interop.foreign_types.push(interop.foreign_types[0].clone());
    assert_eq!(
        interop.validate().unwrap_err().path,
        "interop.foreign_types"
    );

    let mut parallel = parallel();
    parallel.collectives[0].operation = CollectiveOperation::Send {
        input: DistributedValueId {
            function: ProgramFunctionId(99),
            ordinal: 0,
        },
        peer: runmat_types::LabRank(1),
    };
    assert_eq!(
        parallel.validate().unwrap_err().path,
        "parallel.collectives.values"
    );
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn wasm_uses_the_same_construct_schema_vectors() {
    schema_round_trip_vectors();
}
