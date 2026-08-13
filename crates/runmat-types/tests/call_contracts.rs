use runmat_types::{
    infer_call, CallContract, CallRequest, CapabilityRequirement, DynamicReason, EffectKind,
    LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ValueFact, ValueKindFact,
};

fn output(class: NumericClass) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    }))
}

fn request(count: RequestedOutputCount) -> CallRequest {
    CallRequest {
        arguments: Vec::new(),
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(count),
    }
}

#[test]
fn fixed_contract_materializes_exact_requested_slots() {
    let contract = CallContract::fixed(vec![
        output(NumericClass::Double),
        output(NumericClass::UInt64),
    ]);
    let inferred = infer_call(&contract, &request(RequestedOutputCount::Exactly(2)));
    assert_eq!(inferred.outputs, contract.outputs);
    assert!(!inferred.dynamic_outputs);
    assert!(inferred.diagnostics.is_empty());

    let zero = infer_call(&contract, &request(RequestedOutputCount::Zero));
    assert!(zero.outputs.is_empty());
}

#[test]
fn discards_validate_arity_without_materializing_bindings() {
    let contract = CallContract::fixed(vec![
        output(NumericClass::Double),
        output(NumericClass::Double),
    ]);
    let mut request = request(RequestedOutputCount::Exactly(2));
    request.outputs.discarded.insert(0);
    let inferred = infer_call(&contract, &request);
    assert_eq!(inferred.outputs.len(), 2);
    assert_eq!(inferred.materialized_outputs().count(), 1);
    assert_eq!(inferred.materialized_outputs().next().unwrap().0, 1);
}

#[test]
fn fixed_and_variadic_arity_failures_are_explicit() {
    let contract = CallContract::fixed(vec![output(NumericClass::Double)]);
    let inferred = infer_call(&contract, &request(RequestedOutputCount::Exactly(2)));
    assert_eq!(inferred.diagnostics[0].code, "RM-TYPE-CALL-ARITY");
    assert_eq!(inferred.outputs.len(), 2);
    assert!(matches!(inferred.outputs[1].kind, ValueKindFact::Unknown));

    let mut variadic = CallContract::fixed(vec![output(NumericClass::Double)]);
    variadic.variadic_output = Some(Box::new(output(NumericClass::UInt8)));
    variadic.maximum_outputs = None;
    let inferred = infer_call(&variadic, &request(RequestedOutputCount::Exactly(3)));
    assert_eq!(inferred.outputs[1], output(NumericClass::UInt8));
    assert!(inferred.diagnostics.is_empty());
}

#[test]
fn current_nargout_remains_dynamic() {
    let contract = CallContract::dynamic(DynamicReason::DynamicDispatch);
    let inferred = infer_call(
        &contract,
        &request(RequestedOutputCount::CurrentFunctionNargout),
    );
    assert!(inferred.dynamic_outputs);
    assert!(inferred.outputs.is_empty());
}

#[test]
fn call_contracts_and_requests_round_trip_without_runtime_state() {
    let contract = CallContract::dynamic(DynamicReason::DynamicDispatch);
    let encoded = serde_json::to_vec(&contract).unwrap();
    assert_eq!(
        serde_json::from_slice::<CallContract>(&encoded).unwrap(),
        contract
    );

    let mut request = request(RequestedOutputCount::Exactly(3));
    request.outputs.discarded.insert(1);
    let encoded = serde_json::to_vec(&request).unwrap();
    assert_eq!(
        serde_json::from_slice::<CallRequest>(&encoded).unwrap(),
        request
    );
}

#[test]
fn inference_preserves_declared_effects_and_capabilities_independently_of_outputs() {
    let mut contract = CallContract::fixed(vec![output(NumericClass::Double)]);
    contract
        .effects
        .0
        .extend([EffectKind::FilesystemRead, EffectKind::MayThrow]);
    contract
        .capabilities
        .0
        .insert(CapabilityRequirement::Filesystem);
    let inferred = infer_call(&contract, &request(RequestedOutputCount::Zero));
    assert!(inferred.outputs.is_empty());
    assert_eq!(inferred.effects, contract.effects);
    assert_eq!(inferred.capabilities, contract.capabilities);
}
