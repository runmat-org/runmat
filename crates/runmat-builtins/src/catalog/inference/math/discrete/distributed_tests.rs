use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    CallRequest, DistributedFact, DistributedOwner, DistributedValueId, DistributionScheme,
    LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection, ProgramFunctionId,
    RequestedOutputCount, ValueFact, ValueKindFact,
};

fn distributed_numeric(class: NumericClass) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Distributed(DistributedFact {
        id: DistributedValueId {
            function: ProgramFunctionId(11),
            ordinal: 0,
        },
        owner: DistributedOwner::Client(ProgramFunctionId(11)),
        scheme: Some(DistributionScheme::Replicated),
        value: Box::new(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }))),
        materializable: true,
    }))
}

fn request(class: NumericClass) -> CallRequest {
    CallRequest {
        arguments: vec![distributed_numeric(class)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn factorial_distributed_contract_rejects_64_bit_integer_classes() {
    let entry = builtin_catalog_entry_by_name("factorial").expect("factorial catalog entry");
    let supported = infer_catalog_call(entry, &request(NumericClass::UInt32));
    assert!(
        supported.diagnostics.is_empty(),
        "{:#?}",
        supported.diagnostics
    );
    assert!(matches!(
        supported.outputs[0].kind,
        ValueKindFact::Distributed(_)
    ));

    for class in [NumericClass::Int64, NumericClass::UInt64] {
        let rejected = infer_catalog_call(entry, &request(class));
        assert_eq!(rejected.diagnostics.len(), 1);
        assert_eq!(
            rejected.diagnostics[0].code,
            "RM-CATALOG-DISTRIBUTED-NUMERIC-CLASS"
        );
    }
}
