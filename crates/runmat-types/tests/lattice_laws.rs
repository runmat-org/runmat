use runmat_types::{
    AliasFact, CallableFact, CallableIdentity, CapabilityRequirement, CapabilitySet, CellFact,
    CertaintyFact, ContiguityFact, DimensionFact, DistributedFact, DistributedValueId,
    DistributionScheme, DynamicReason, ExecutionFact, FactJoin, FactSatisfaction, FactWiden,
    ForeignAffinityFact, ForeignFact, ForeignLifetimeFact, ForeignOwnershipFact, InvalidationCause,
    InvalidationVector, LayoutFact, MutationFact, NumericClass, NumericDomain, NumericFact,
    OutputListFact, ParallelRegionId, ProgramFunctionId, RegionId, ResidencyFact, ShapeFact,
    StorageFact, StructFact, SymbolName, ValueFact, ValueKindFact, ValueSequenceFact, ViewFact,
};
use std::collections::{BTreeMap, BTreeSet};

fn numeric(class: NumericClass) -> ValueFact {
    ValueFact {
        kind: ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape: ShapeFact::Scalar,
        storage: StorageFact::Scalar,
        layout: LayoutFact::ColumnMajor,
        contiguity: ContiguityFact::Contiguous,
        view: ViewFact::Materialized,
        residency: ResidencyFact::Host,
        alias: AliasFact::Unique,
        mutation: MutationFact::ValueSemantics,
        certainty: CertaintyFact::Proven,
        invalidation: InvalidationVector::default(),
    }
}

#[test]
fn join_is_commutative_associative_and_idempotent() {
    let facts = representative_facts();
    for a in &facts {
        assert_eq!(a.join(a), *a, "join must be idempotent for {a:?}");
        for b in &facts {
            assert_eq!(a.join(b), b.join(a), "join must commute for {a:?}, {b:?}");
            for c in &facts {
                assert_eq!(
                    a.join(b).join(c),
                    a.join(&b.join(c)),
                    "join must associate for {a:?}, {b:?}, {c:?}"
                );
            }
        }
    }
}

#[test]
fn struct_position_facts_are_local_precision_not_wire_state() {
    let common = BTreeMap::from([("value".into(), numeric(NumericClass::Double))]);
    let fact = StructFact::array(common.clone(), true, vec![common], true);
    let encoded = serde_json::to_string(&fact).expect("serialize structure fact");
    assert!(!encoded.contains("elements"));
    let decoded: StructFact = serde_json::from_str(&encoded).expect("deserialize structure fact");
    assert_eq!(decoded.fields, fact.fields);
    assert!(decoded.fields_complete);
    assert!(decoded.elements.is_empty());
    assert!(!decoded.elements_complete);
}

fn representative_facts() -> Vec<ValueFact> {
    let mut shaped = numeric(NumericClass::Double);
    shaped.shape = ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(1), DimensionFact::Unknown],
    };
    let mut ranked = numeric(NumericClass::Double);
    ranked.shape = ShapeFact::Ranked { rank: 2 };
    let recursive = ValueFact {
        kind: ValueKindFact::Cell(CellFact {
            element: Box::new(numeric(NumericClass::UInt16)),
            elements: vec![numeric(NumericClass::UInt16)],
            elements_complete: true,
        }),
        shape: ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(1), DimensionFact::Known(2)],
        },
        storage: StorageFact::Dense,
        layout: LayoutFact::ColumnMajor,
        contiguity: ContiguityFact::Contiguous,
        view: ViewFact::Materialized,
        residency: ResidencyFact::Host,
        alias: AliasFact::Unique,
        mutation: MutationFact::ValueSemantics,
        certainty: CertaintyFact::Proven,
        invalidation: InvalidationVector::default(),
    };
    let mut structure = numeric(NumericClass::Double);
    structure.kind = ValueKindFact::Struct(StructFact::scalar(
        BTreeMap::from([
            ("count".into(), numeric(NumericClass::UInt64)),
            ("payload".into(), recursive.clone()),
        ]),
        true,
    ));
    let mut callable = numeric(NumericClass::Double);
    callable.kind = ValueKindFact::Callable(CallableFact {
        identity: Some(CallableIdentity::DynamicName(SymbolName("f".into()))),
        capabilities: Default::default(),
        parameters: vec![numeric(NumericClass::Single)],
        parameters_complete: true,
        outputs: vec![numeric(NumericClass::Double)],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: vec![structure.clone()],
        captures_complete: true,
    });
    let mut future = numeric(NumericClass::Double);
    future.kind = ValueKindFact::Execution(ExecutionFact::Future {
        output: ValueSequenceFact::single(numeric(NumericClass::UInt32)),
        state: runmat_types::FutureStateFact::Lazy,
    });
    future.alias = AliasFact::Identity;
    future.mutation = MutationFact::HandleSemantics;
    let mut foreign = numeric(NumericClass::Double);
    foreign.kind = ValueKindFact::Foreign(ForeignFact {
        family: "c".into(),
        type_name: Some("widget".into()),
        type_version: Some(1),
        ownership: ForeignOwnershipFact::Owned,
        affinity: ForeignAffinityFact::OriginThread,
        lifetime: ForeignLifetimeFact::Session,
    });
    let mut distributed = numeric(NumericClass::Double);
    distributed.kind = ValueKindFact::Distributed(DistributedFact {
        id: DistributedValueId {
            function: ProgramFunctionId(0),
            ordinal: 1,
        },
        owner: runmat_types::DistributedOwner::Region(ParallelRegionId(RegionId {
            function: ProgramFunctionId(0),
            ordinal: 2,
        })),
        scheme: Some(DistributionScheme::Block { dimension: 1 }),
        value: Box::new(numeric(NumericClass::Double)),
        materializable: true,
    });
    let mut outputs = numeric(NumericClass::Double);
    outputs.kind = ValueKindFact::OutputList(OutputListFact {
        outputs: vec![numeric(NumericClass::Int16), structure.clone()],
        variadic: false,
    });
    let mut invalidated = shaped.clone();
    invalidated.invalidation = InvalidationVector(BTreeSet::from([
        InvalidationCause::SourceChanged,
        InvalidationCause::CatalogChanged,
    ]));
    vec![
        ValueFact::never(),
        ValueFact::unknown(DynamicReason::RuntimeValue),
        numeric(NumericClass::Double),
        numeric(NumericClass::Single),
        shaped,
        ranked,
        recursive,
        structure,
        callable,
        future,
        distributed,
        foreign,
        outputs,
        invalidated,
    ]
}

#[test]
fn never_is_bottom_and_unknown_does_not_gain_precision() {
    let known = numeric(NumericClass::UInt64);
    assert_eq!(ValueFact::never().join(&known), known);
    let unknown = ValueFact::unknown(DynamicReason::ExternalData);
    assert!(matches!(unknown.join(&known).kind, ValueKindFact::Unknown));
}

#[test]
fn widening_is_monotone_and_stable() {
    let a = numeric(NumericClass::Double);
    let b = numeric(NumericClass::Single);
    let widened = a.widen(&b);
    assert_eq!(widened.widen(&widened), widened);
    assert_eq!(a.widen(&b), b.widen(&a));
}

#[test]
fn joins_do_not_turn_conflicting_runtime_metadata_into_negative_proofs() {
    let mut value_object = numeric(NumericClass::Double);
    value_object.kind = ValueKindFact::Object(runmat_types::ObjectFact {
        class: None,
        runtime_class: None,
        properties: BTreeMap::new(),
        properties_complete: false,
        handle_semantics: Some(false),
    });
    let mut handle_object = value_object.clone();
    let ValueKindFact::Object(object) = &mut handle_object.kind else {
        unreachable!();
    };
    object.handle_semantics = Some(true);
    let ValueKindFact::Object(joined_object) = value_object.join(&handle_object).kind else {
        panic!("object facts should retain their common kind");
    };
    assert_eq!(joined_object.handle_semantics, None);

    let mut known_callable = numeric(NumericClass::Double);
    known_callable.kind = ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: Default::default(),
        parameters: vec![numeric(NumericClass::Double)],
        parameters_complete: true,
        outputs: vec![numeric(NumericClass::Double)],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: Vec::new(),
        captures_complete: true,
    });
    let mut different_callable = known_callable.clone();
    let ValueKindFact::Callable(callable) = &mut different_callable.kind else {
        unreachable!();
    };
    callable.parameters.push(numeric(NumericClass::Double));
    let ValueKindFact::Callable(joined_callable) = known_callable.join(&different_callable).kind
    else {
        panic!("callable facts should retain their common kind");
    };
    assert!(!joined_callable.parameters_complete);
    assert!(joined_callable.variadic_inputs);
}

#[test]
fn callable_joins_preserve_every_branch_capability_requirement() {
    let callable = |capability| {
        let mut fact = numeric(NumericClass::Double);
        fact.kind = ValueKindFact::Callable(CallableFact {
            identity: Some(CallableIdentity::DynamicName(SymbolName("target".into()))),
            capabilities: CapabilitySet(BTreeSet::from([capability])),
            parameters: vec![numeric(NumericClass::Double)],
            parameters_complete: true,
            outputs: vec![numeric(NumericClass::Double)],
            outputs_complete: true,
            variadic_inputs: false,
            variadic_outputs: false,
            captures: Vec::new(),
            captures_complete: true,
        });
        fact
    };
    let accelerator = callable(CapabilityRequirement::Accelerator);
    let filesystem = callable(CapabilityRequirement::Filesystem);
    let joined = accelerator.join(&filesystem);
    let ValueKindFact::Callable(joined_callable) = &joined.kind else {
        panic!("a callable join must remain callable");
    };

    assert_eq!(
        joined_callable.capabilities,
        CapabilitySet(BTreeSet::from([
            CapabilityRequirement::Accelerator,
            CapabilityRequirement::Filesystem,
        ]))
    );
    assert!(joined.satisfies(&accelerator));
    assert!(joined.satisfies(&filesystem));
    assert!(!accelerator.satisfies(&joined));
}
