use runmat_types::{
    CallRequest, DistributedFact, DistributedOwner, DistributedValueId, DistributionScheme,
    LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact, OutputSelection,
    ProgramFunctionId, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

use crate::{builtin_catalog_entry_by_name, infer_catalog_call};

fn class_request(class: &str) -> CallRequest {
    CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::String)],
        literals: LiteralContext::new(vec![LiteralValue::String(class.into())]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn class_names_determine_scalar_output_class() {
    for (builtin, source_name, class) in [
        ("intmin", "uint64", NumericClass::UInt64),
        ("intmax", "uint64", NumericClass::UInt64),
        ("realmin", "single", NumericClass::Single),
        ("realmax", "single", NumericClass::Single),
        ("flintmax", "single", NumericClass::Single),
    ] {
        let inference = infer_catalog_call(
            builtin_catalog_entry_by_name(builtin).expect("numeric limit entry"),
            &class_request(source_name),
        );
        assert!(inference.diagnostics.is_empty(), "{builtin}");
        assert_eq!(
            inference.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            })
        );
        assert_eq!(inference.outputs[0].shape, ShapeFact::Scalar);
    }
}

#[test]
fn like_preserves_distributed_placement_and_value_representation() {
    let mut local = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(8), Some(2)]),
        StorageFact::Dense,
    );
    local.residency = ResidencyFact::Host;
    let distributed = DistributedFact {
        id: DistributedValueId {
            function: ProgramFunctionId(11),
            ordinal: 2,
        },
        owner: DistributedOwner::Client(ProgramFunctionId(11)),
        scheme: Some(DistributionScheme::Replicated),
        value: Box::new(local),
        materializable: true,
    };
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("realmax").expect("realmax entry"),
        &CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Distributed(distributed.clone())),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Keyword("like".into()),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(
        inference.diagnostics.is_empty(),
        "{:#?}",
        inference.diagnostics
    );
    let ValueKindFact::Distributed(output) = &inference.outputs[0].kind else {
        panic!("numeric limit like form must retain distributed placement");
    };
    assert_eq!(output.id, distributed.id);
    assert_eq!(output.scheme, distributed.scheme);
    assert_eq!(output.value.shape, ShapeFact::Scalar);
    assert_eq!(
        output.value.kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        })
    );
}
