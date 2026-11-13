use super::*;
use runmat_types::*;

#[test]
fn single_result_keeps_uniqueness_but_dynamic_replication_cannot() {
    let value = ValueFact::scalar(ValueKindFact::Logical);
    let one = infer_catalog_call(
        &DEAL_CATALOG_ENTRY,
        &request(vec![value.clone()], RequestedOutputCount::One),
    );
    assert_eq!(one.outputs, vec![value.clone()]);
    let dynamic = infer_catalog_call(
        &DEAL_CATALOG_ENTRY,
        &request(vec![value], RequestedOutputCount::CurrentFunctionNargout),
    );
    assert!(dynamic.dynamic_outputs);
    assert!(dynamic
        .outputs
        .iter()
        .all(|value| value.alias != AliasFact::Unique));
}

#[test]
fn replication_weakens_nested_payload_ownership_not_callable_contracts() {
    let mut resident = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    resident.residency = ResidencyFact::Device {
        provider: Some("owner".into()),
    };
    let callable = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: Default::default(),
        parameters: vec![resident.clone()],
        parameters_complete: true,
        outputs: vec![resident.clone()],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: vec![resident.clone()],
        captures_complete: true,
    }));
    let fields = std::collections::BTreeMap::from([("callback".into(), callable)]);
    let structure = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
    let cell = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(structure.clone()),
            elements: vec![structure],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    );
    let result = infer_catalog_call(
        &DEAL_CATALOG_ENTRY,
        &request(vec![cell], RequestedOutputCount::Exactly(2)),
    );
    for output in result.outputs {
        assert_eq!(output.alias, AliasFact::Unknown);
        let ValueKindFact::Cell(cell) = output.kind else {
            panic!("cell");
        };
        for structure in std::iter::once(cell.element.as_ref()).chain(cell.elements.iter()) {
            assert_eq!(structure.alias, AliasFact::Unknown);
            let ValueKindFact::Struct(structure) = &structure.kind else {
                panic!("struct");
            };
            for fields in std::iter::once(&structure.fields).chain(structure.elements.iter()) {
                let callback = &fields["callback"];
                assert_eq!(callback.alias, AliasFact::Unknown);
                let ValueKindFact::Callable(callback) = &callback.kind else {
                    panic!("callable");
                };
                assert_eq!(callback.captures[0].alias, AliasFact::Unknown);
                assert_eq!(callback.captures[0].residency, resident.residency);
                assert_eq!(callback.captures[0].shape, resident.shape);
                assert_eq!(callback.parameters, vec![resident.clone()]);
                assert_eq!(callback.outputs, vec![resident.clone()]);
            }
        }
    }
}
fn request(arguments: Vec<ValueFact>, requested: RequestedOutputCount) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(requested),
    }
}
#[test]
fn replication_preserves_semantic_facts_and_discarded_slots_without_unique_alias_claims() {
    let mut value = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3), Some(4)]),
        StorageFact::Dense,
    );
    let mut call = request(vec![value.clone()], RequestedOutputCount::Exactly(3));
    call.outputs.discarded.insert(1);
    let result = infer_catalog_call(&DEAL_CATALOG_ENTRY, &call);
    value.alias = AliasFact::Unknown;
    assert_eq!(result.outputs, vec![value; 3]);
    assert_eq!(result.discarded, vec![1]);
    assert!(result.diagnostics.is_empty());
}
#[test]
fn positional_counts_zero_and_dynamic_requests_remain_distinct() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    for (inputs, outputs, invalid) in [
        (0, 0, false),
        (0, 1, true),
        (1, 0, false),
        (1, 3, false),
        (2, 1, true),
        (2, 2, false),
        (2, 0, true),
    ] {
        let result = infer_catalog_call(
            &DEAL_CATALOG_ENTRY,
            &request(
                vec![unknown.clone(); inputs],
                RequestedOutputCount::Exactly(outputs),
            ),
        );
        assert_eq!(
            !result.diagnostics.is_empty(),
            invalid,
            "{inputs}/{outputs}"
        );
    }
    let result = infer_catalog_call(
        &DEAL_CATALOG_ENTRY,
        &request(vec![unknown], RequestedOutputCount::CurrentFunctionNargout),
    );
    assert!(result.dynamic_outputs);
    assert!(result.diagnostics.is_empty());
}
