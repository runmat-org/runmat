use super::*;
use crate::{BuiltinAsyncBehavior, BuiltinCompatibility, BuiltinPurity, BuiltinSemanticKind};
use runmat_types::{CapabilityRequirement, EffectKind};

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = pilot()",
    inputs: &[],
    outputs: &OUTPUTS,
}];
const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    summary: "Pilot builtin.",
    keywords: &["pilot"],
    related: &[],
    introduced: None,
    status: None,
    examples: &[],
};
const PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Optional,
    residency: BuiltinResidencyPolicy::PreserveInputs,
    fusion: BuiltinFusionPolicy::Candidate,
    distributed: crate::BuiltinDistributedPolicy::Unsupported,
};
const LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: runmat_types::ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};
const EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
const CAPABILITIES: [CapabilityRequirement; 1] = [CapabilityRequirement::HostRuntime];

const PILOT_ID: BuiltinCatalogIdentity = BuiltinCatalogIdentity { name: "pilot" };
const PILOT_BINDINGS: [BuiltinBindingDeclaration; 1] = [BuiltinBindingDeclaration {
    identity: BuiltinBindingIdentity {
        builtin: PILOT_ID,
        variant: "default",
    },
    availability: BuiltinBindingAvailability::Required,
}];
const PILOT: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: PILOT_ID,
    category: "test",
    documentation: DOCUMENTATION,
    descriptor: &DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRuleId("test.pilot"),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &CAPABILITIES,
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: &PILOT_BINDINGS,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: None,
    suppress_auto_output: false,
};

const SECOND_ID: BuiltinCatalogIdentity = BuiltinCatalogIdentity { name: "second" };
const SECOND_BINDINGS: [BuiltinBindingDeclaration; 1] = [BuiltinBindingDeclaration {
    identity: BuiltinBindingIdentity {
        builtin: SECOND_ID,
        variant: "default",
    },
    availability: BuiltinBindingAvailability::Required,
}];
const SECOND: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: SECOND_ID,
    bindings: &SECOND_BINDINGS,
    ..PILOT
};

#[test]
fn valid_catalog_has_stable_order_independent_fingerprint() {
    assert!(validate_builtin_catalog(&[&PILOT, &SECOND]).is_empty());
    assert_eq!(
        canonical_catalog_fingerprint(&[&PILOT, &SECOND]).unwrap(),
        canonical_catalog_fingerprint(&[&SECOND, &PILOT]).unwrap()
    );
}

#[test]
fn validation_rejects_duplicate_catalog_and_binding_identities() {
    let errors = validate_builtin_catalog(&[&PILOT, &PILOT]);
    assert!(errors
        .iter()
        .any(|error| error.message == "duplicate builtin catalog identity"));
    assert!(errors
        .iter()
        .any(|error| error.message == "duplicate builtin binding identity"));
}

#[test]
fn migrated_registry_is_valid_and_case_insensitive() {
    let errors = validate_builtin_catalog(builtin_catalog_entries());
    assert!(errors.is_empty(), "catalog errors: {errors:#?}");
    assert_eq!(
        builtin_catalog_entry_by_name("FULL")
            .expect("full catalog entry")
            .contract
            .inference_rule
            .0,
        "array.full"
    );
}

#[test]
fn distributed_execution_policy_is_declared_by_the_canonical_catalog() {
    assert_eq!(
        builtin_catalog_entry_by_name("abs")
            .expect("abs catalog entry")
            .placement
            .distributed,
        BuiltinDistributedPolicy::MapUnary
    );
    assert_eq!(
        builtin_catalog_entry_by_name("gather")
            .expect("gather catalog entry")
            .placement
            .distributed,
        BuiltinDistributedPolicy::MaterializeArguments
    );
    assert_eq!(
        builtin_catalog_entry_by_name("full")
            .expect("full catalog entry")
            .placement
            .distributed,
        BuiltinDistributedPolicy::Unsupported
    );
}

#[test]
fn gpu_array_catalog_owns_its_accelerator_contract() {
    let entry = builtin_catalog_entry_by_name("gpuArray").expect("gpuArray catalog entry");

    assert_eq!(
        entry.contract.capability_set().0,
        [CapabilityRequirement::Accelerator].into_iter().collect()
    );
    assert_eq!(
        entry.placement.accelerator,
        BuiltinAcceleratorPolicy::Required
    );
    assert_eq!(
        entry.placement.residency,
        BuiltinResidencyPolicy::ProduceResident
    );
    assert_eq!(entry.descriptor.signatures.len(), 5);
    assert_eq!(entry.descriptor.errors.len(), 15);
}

#[test]
fn gpu_array_inference_preserves_or_converts_typed_facts_before_device_placement() {
    use runmat_types::{
        CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
        OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };

    let entry = builtin_catalog_entry_by_name("gpuArray").expect("gpuArray catalog entry");
    let mut source = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt16,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(4), Some(1)]),
        StorageFact::Dense,
    );
    source.residency = runmat_types::ResidencyFact::Host;

    let preserve = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![source.clone()],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(preserve.outputs[0].kind, source.kind);
    assert_eq!(preserve.outputs[0].shape, source.shape);
    assert_eq!(
        preserve.outputs[0].residency,
        runmat_types::ResidencyFact::Device { provider: None }
    );

    let convert = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![source.clone(), ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("int32".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(
        convert.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int32,
            domain: NumericDomain::Real,
        })
    );

    let reshape = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![
                source,
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Number(2.0),
                LiteralValue::Number(2.0),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(
        reshape.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
}

#[test]
fn distributed_call_inference_preserves_the_execution_owned_value_contract() {
    use runmat_types::{
        CallRequest, DistributedFact, DistributedOwner, DistributedValueId, DistributionScheme,
        LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
        ProgramFunctionId, RequestedOutputCount, ValueFact, ValueKindFact,
    };

    let local = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Int16,
        domain: NumericDomain::Complex,
    }));
    let distributed = DistributedFact {
        id: DistributedValueId {
            function: ProgramFunctionId(7),
            ordinal: 3,
        },
        owner: DistributedOwner::Client(ProgramFunctionId(7)),
        scheme: Some(DistributionScheme::Replicated),
        value: Box::new(local),
        materializable: true,
    };
    let request = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::Distributed(
            distributed.clone(),
        ))],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };

    let mapped = infer_catalog_call(
        builtin_catalog_entry_by_name("abs").expect("abs entry"),
        &request,
    );
    assert!(mapped.diagnostics.is_empty(), "{:#?}", mapped.diagnostics);
    let ValueKindFact::Distributed(mapped) = &mapped.outputs[0].kind else {
        panic!("partition-local mapping must retain a distributed result fact");
    };
    assert_eq!(mapped.id, distributed.id);
    assert_eq!(mapped.owner, distributed.owner);
    assert_eq!(mapped.scheme, distributed.scheme);
    assert_eq!(mapped.materializable, distributed.materializable);
    assert_eq!(
        mapped.value.kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int16,
            domain: NumericDomain::Real,
        })
    );

    let gathered = infer_catalog_call(
        builtin_catalog_entry_by_name("gather").expect("gather entry"),
        &request,
    );
    assert!(
        gathered.diagnostics.is_empty(),
        "{:#?}",
        gathered.diagnostics
    );
    assert!(matches!(
        gathered.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int16,
            domain: NumericDomain::Complex,
        })
    ));
}

#[test]
fn current_parallel_context_contracts_are_typed_and_parallel_capable() {
    for name in ["getCurrentTask", "getCurrentWorker", "getCurrentJob"] {
        let entry = builtin_catalog_entry_by_name(name).expect("parallel context catalog entry");
        let effects = entry.contract.effect_set();
        let capabilities = entry.contract.capability_set();

        assert_eq!(
            effects.0,
            [EffectKind::MayThrow].into_iter().collect(),
            "{name} must not acquire an unknown effect"
        );
        assert_eq!(
            capabilities.0,
            [CapabilityRequirement::ParallelRuntime]
                .into_iter()
                .collect(),
            "{name} must retain its execution capability through analysis"
        );
        assert_eq!(
            entry.contract.maturity,
            BuiltinContractMaturity::DynamicByDesign
        );
    }
}

#[test]
fn parallel_surface_retains_pool_future_and_fetch_facts() {
    use runmat_types::{
        CallRequest, CallableFact, ExecutionFact, FutureStateFact, LiteralContext, LiteralValue,
        NumericClass, NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ValueFact,
        ValueKindFact,
    };

    let request = CallRequest {
        arguments: Vec::new(),
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let pool = infer_catalog_call(
        builtin_catalog_entry_by_name("parpool").expect("parpool entry"),
        &request,
    );
    assert!(matches!(
        pool.outputs[0].kind,
        ValueKindFact::Execution(ExecutionFact::Pool)
    ));
    assert_eq!(
        pool.capabilities.0,
        [CapabilityRequirement::ParallelRuntime]
            .into_iter()
            .collect()
    );

    let callable_output = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt16,
        domain: NumericDomain::Real,
    }));
    let callable = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: Default::default(),
        parameters: vec![ValueFact::unknown(
            runmat_types::DynamicReason::RuntimeValue,
        )],
        parameters_complete: true,
        outputs: vec![callable_output.clone()],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: Vec::new(),
        captures_complete: true,
    }));
    let future = infer_catalog_call(
        builtin_catalog_entry_by_name("parfeval").expect("parfeval entry"),
        &CallRequest {
            arguments: vec![
                callable,
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
                ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Number(1.0),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(matches!(
        &future.outputs[0].kind,
        ValueKindFact::Execution(ExecutionFact::Future {
            output,
            state: FutureStateFact::Unknown,
        }) if **output == callable_output
    ));

    let fetched = infer_catalog_call(
        builtin_catalog_entry_by_name("fetchOutputs").expect("fetchOutputs entry"),
        &CallRequest {
            arguments: future.outputs,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(fetched.outputs, vec![callable_output]);
    assert!(fetched.effects.0.contains(&EffectKind::MaySuspend));
    assert!(fetched.effects.0.contains(&EffectKind::MayThrow));
}

#[test]
fn codistributor_constructors_have_static_value_class_facts() {
    use runmat_types::{
        CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ValueKindFact,
    };

    let request = CallRequest {
        arguments: Vec::new(),
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    for (builtin, expected_class) in [
        ("codistributor1d", "codistributor1d"),
        ("codistributor2dbc", "codistributor2dbc"),
    ] {
        let inference = infer_catalog_call(
            builtin_catalog_entry_by_name(builtin).expect("codistributor catalog entry"),
            &request,
        );
        assert!(inference.diagnostics.is_empty(), "{builtin}");
        let ValueKindFact::Object(object) = &inference.outputs[0].kind else {
            panic!("{builtin} must produce an object fact");
        };
        assert_eq!(
            object
                .runtime_class
                .as_ref()
                .and_then(|name| name.0.first())
                .map(|name| name.0.as_str()),
            Some(expected_class)
        );
        assert_eq!(object.handle_semantics, Some(false));
    }
}

#[test]
fn zeros_contract_uses_literal_dimensions_class_and_like_residency() {
    use runmat_types::{
        CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
        OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
        ValueKindFact,
    };

    let dimension = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let request = CallRequest {
        arguments: vec![dimension.clone(), dimension],
        literals: LiteralContext::new(vec![LiteralValue::Number(2.0), LiteralValue::Number(3.0)]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("zeros").expect("zeros entry"),
        &request,
    );
    assert!(inference.diagnostics.is_empty());
    assert_eq!(
        inference.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(inference.outputs[0].storage, StorageFact::Dense);

    let mut prototype = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(4), Some(5)]),
        StorageFact::Dense,
    );
    prototype.residency = ResidencyFact::Device {
        provider: Some("pilot".into()),
    };
    let like = ValueFact::scalar(ValueKindFact::String);
    let request = CallRequest {
        arguments: vec![like, prototype.clone()],
        literals: LiteralContext::new(vec![
            LiteralValue::Keyword("like".into()),
            LiteralValue::Unknown,
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("zeros").expect("zeros entry"),
        &request,
    );
    assert_eq!(inference.outputs[0].kind, prototype.kind);
    assert_eq!(inference.outputs[0].shape, prototype.shape);
    assert_eq!(inference.outputs[0].residency, prototype.residency);
}

#[test]
fn full_contract_densifies_sparse_facts_without_losing_class_shape_or_residency() {
    use runmat_types::{
        CallRequest, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };

    let mut input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt32,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Sparse,
    );
    input.residency = runmat_types::ResidencyFact::Host;
    let request = CallRequest {
        arguments: vec![input.clone()],
        literals: runmat_types::LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("full").expect("full entry"),
        &request,
    );
    assert!(inference.diagnostics.is_empty());
    assert_eq!(inference.outputs[0].kind, input.kind);
    assert_eq!(inference.outputs[0].shape, input.shape);
    assert_eq!(inference.outputs[0].residency, input.residency);
    assert_eq!(inference.outputs[0].storage, StorageFact::Dense);
}

#[test]
fn abs_contract_preserves_class_shape_storage_and_residency_but_makes_complex_real() {
    use runmat_types::{
        CallRequest, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };
    let mut input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int64,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Sparse,
    );
    input.residency = runmat_types::ResidencyFact::Device {
        provider: Some("pilot".into()),
    };
    let request = CallRequest {
        arguments: vec![input.clone()],
        literals: runmat_types::LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("abs").expect("abs entry"),
        &request,
    );
    assert!(inference.diagnostics.is_empty());
    assert_eq!(
        inference.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int64,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(inference.outputs[0].shape, input.shape);
    assert_eq!(inference.outputs[0].storage, input.storage);
    assert_eq!(inference.outputs[0].residency, input.residency);
}

#[test]
fn gather_contract_maps_corresponding_outputs_to_host_and_checks_output_count() {
    use runmat_types::{
        CallRequest, OutputSelection, RequestedOutputCount, ResidencyFact, ValueFact, ValueKindFact,
    };
    let mut first = ValueFact::scalar(ValueKindFact::Logical);
    first.residency = ResidencyFact::Device {
        provider: Some("gpu-a".into()),
    };
    let mut second = ValueFact::scalar(ValueKindFact::String);
    second.residency = ResidencyFact::Device {
        provider: Some("gpu-b".into()),
    };
    let request = CallRequest {
        arguments: vec![first.clone(), second.clone()],
        literals: runmat_types::LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("gather").expect("gather entry"),
        &request,
    );
    assert!(inference.diagnostics.is_empty());
    assert_eq!(inference.outputs.len(), 2);
    assert_eq!(inference.outputs[0].kind, first.kind);
    assert_eq!(inference.outputs[1].kind, second.kind);
    assert!(inference
        .outputs
        .iter()
        .all(|fact| fact.residency == ResidencyFact::Host));

    let bad_request = CallRequest {
        outputs: OutputSelection::new(RequestedOutputCount::One),
        ..request
    };
    let bad = infer_catalog_call(
        builtin_catalog_entry_by_name("gather").expect("gather entry"),
        &bad_request,
    );
    assert!(bad
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-GATHER-OUTPUTS"));
}

#[test]
fn struct_contract_tracks_literal_field_names_and_value_facts() {
    use runmat_types::{
        CallRequest, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount,
        ValueFact, ValueKindFact,
    };
    let field_name = ValueFact::scalar(ValueKindFact::String);
    let value = ValueFact::scalar(ValueKindFact::Logical);
    let request = CallRequest {
        arguments: vec![field_name, value.clone()],
        literals: LiteralContext::new(vec![
            LiteralValue::String("Flag".into()),
            LiteralValue::Unknown,
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("struct").expect("struct entry"),
        &request,
    );
    assert!(inference.diagnostics.is_empty());
    let ValueKindFact::Struct(fact) = &inference.outputs[0].kind else {
        panic!("expected struct fact")
    };
    assert!(fact.fields_complete);
    assert_eq!(fact.fields.get("Flag"), Some(&value));
}

#[test]
fn feval_contract_preserves_known_callable_outputs_and_dynamic_effects() {
    use runmat_types::{
        CallRequest, CallableFact, EffectKind, LiteralContext, OutputSelection,
        RequestedOutputCount, ValueFact, ValueKindFact,
    };
    assert_eq!(
        builtin_catalog_entry_by_name("feval")
            .expect("feval entry")
            .link
            .execution_stack,
        runmat_types::ExecutionStackRequirement::Process
    );
    let first_output = ValueFact::scalar(ValueKindFact::Logical);
    let second_output = ValueFact::scalar(ValueKindFact::String);
    let callable = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: Default::default(),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: vec![first_output.clone(), second_output.clone()],
        outputs_complete: true,
        variadic_inputs: true,
        variadic_outputs: false,
        captures: Vec::new(),
        captures_complete: true,
    }));
    let request = CallRequest {
        arguments: vec![callable],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("feval").expect("feval entry"),
        &request,
    );
    assert!(inference.diagnostics.is_empty());
    assert_eq!(inference.outputs, vec![first_output, second_output]);
    assert!(inference.effects.0.contains(&EffectKind::HostCallback));
    assert!(inference.effects.0.contains(&EffectKind::MaySuspend));
    assert!(inference.effects.0.contains(&EffectKind::MayThrow));
    assert!(inference.effects.0.contains(&EffectKind::Unknown));
    assert!(inference.capabilities.0.is_empty());

    let partially_known_callable = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: Default::default(),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: vec![ValueFact::scalar(ValueKindFact::Character)],
        outputs_complete: false,
        variadic_inputs: true,
        variadic_outputs: false,
        captures: Vec::new(),
        captures_complete: false,
    }));
    let dynamic = infer_catalog_call(
        builtin_catalog_entry_by_name("feval").expect("feval entry"),
        &CallRequest {
            arguments: vec![partially_known_callable],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
        },
    );
    assert!(dynamic.diagnostics.is_empty());
    assert_eq!(dynamic.outputs[0].kind, ValueKindFact::Character);
    assert!(dynamic.outputs[1..]
        .iter()
        .all(|output| output.kind == ValueKindFact::Unknown));

    let missing_target = infer_catalog_call(
        builtin_catalog_entry_by_name("feval").expect("feval entry"),
        &CallRequest {
            arguments: Vec::new(),
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(missing_target
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FEVAL-ARITY"));
}
