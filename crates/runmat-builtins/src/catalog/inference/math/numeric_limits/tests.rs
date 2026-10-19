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
        ("eps", "single", NumericClass::Single),
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
        assert_eq!(
            inference.outputs[0].storage,
            if class == NumericClass::Single {
                StorageFact::Dense
            } else {
                StorageFact::Scalar
            }
        );
    }
}

#[test]
fn eps_array_fact_is_fresh_real_and_shape_preserving() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(2), Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let result = infer_catalog_call(
        builtin_catalog_entry_by_name("eps").unwrap(),
        &CallRequest {
            arguments: vec![input.clone()],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].shape, input.shape);
    assert_eq!(result.outputs[0].storage, StorageFact::Dense);
    assert_eq!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(result.outputs[0].alias, runmat_types::AliasFact::Unique);
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

#[test]
fn eps_distributed_value_maps_locally_and_like_constructs_a_scalar() {
    let local = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(8), Some(2)]),
        StorageFact::Dense,
    );
    let source = DistributedFact {
        id: DistributedValueId {
            function: ProgramFunctionId(12),
            ordinal: 1,
        },
        owner: DistributedOwner::Client(ProgramFunctionId(12)),
        scheme: Some(DistributionScheme::Replicated),
        value: Box::new(local),
        materializable: true,
    };
    let entry = builtin_catalog_entry_by_name("eps").unwrap();
    for (arguments, literals, expected_shape, expected_domain) in [
        (
            vec![ValueFact::scalar(ValueKindFact::Distributed(
                source.clone(),
            ))],
            LiteralContext::default(),
            ShapeFact::from(vec![Some(8), Some(2)]),
            NumericDomain::Real,
        ),
        (
            vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Distributed(source.clone())),
            ],
            LiteralContext::new(vec![
                LiteralValue::Keyword("like".into()),
                LiteralValue::Unknown,
            ]),
            ShapeFact::Scalar,
            NumericDomain::Complex,
        ),
    ] {
        let result = infer_catalog_call(
            entry,
            &CallRequest {
                arguments,
                literals,
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        );
        assert!(result.diagnostics.is_empty(), "{:#?}", result.diagnostics);
        let ValueKindFact::Distributed(output) = &result.outputs[0].kind else {
            panic!("eps must retain distributed placement");
        };
        assert_eq!(output.id, source.id);
        assert_eq!(output.value.shape, expected_shape);
        assert_eq!(
            output.value.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: expected_domain,
            })
        );
    }
}
