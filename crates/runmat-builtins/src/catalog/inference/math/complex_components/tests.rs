use crate::{
    builtin_catalog_entry_by_name, infer_catalog_call, BuiltinInferenceRule, MathInferenceRule,
    NumericComponentRule,
};

mod input_contracts;

#[test]
fn one_element_complex_tensors_keep_the_runtime_representation() {
    use runmat_types::{
        CallRequest, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };
    for (name, class, expected) in [
        ("conj", NumericClass::Double, StorageFact::Dense),
        ("conj", NumericClass::Single, StorageFact::Dense),
        ("conj", NumericClass::UInt64, StorageFact::Dense),
        ("real", NumericClass::Single, StorageFact::Dense),
        ("imag", NumericClass::Single, StorageFact::Dense),
        ("real", NumericClass::Double, StorageFact::Scalar),
        ("imag", NumericClass::UInt64, StorageFact::Scalar),
    ] {
        let input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Complex,
            }),
            ShapeFact::from(vec![Some(1), Some(1)]),
            StorageFact::Dense,
        );
        let inferred = infer_catalog_call(
            builtin_catalog_entry_by_name(name).unwrap(),
            &CallRequest {
                arguments: vec![input],
                literals: Default::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        );
        assert!(inferred.diagnostics.is_empty(), "{name}");
        assert_eq!(inferred.outputs[0].storage, expected, "{name} {class:?}");
    }
}

#[test]
fn numeric_component_contracts_preserve_exact_facts_and_transform_only_the_component_domain() {
    use runmat_types::{
        AliasFact, CallRequest, ContiguityFact, LayoutFact, MutationFact, NumericClass,
        NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ResidencyFact,
        ShapeFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
    };

    let mut input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(2), Some(3), Some(4)]),
        StorageFact::Dense,
    );
    input.residency = ResidencyFact::Device {
        provider: Some("component-provider".into()),
    };
    input.layout = LayoutFact::ColumnMajor;
    input.contiguity = ContiguityFact::Contiguous;
    input.view = ViewFact::ReadOnlyView;
    input.alias = AliasFact::Shared;
    input.mutation = MutationFact::Immutable;

    for (name, rule, expected_domain) in [
        (
            "conj",
            NumericComponentRule::Conjugate,
            NumericDomain::Complex,
        ),
        ("real", NumericComponentRule::RealPart, NumericDomain::Real),
        (
            "imag",
            NumericComponentRule::ImaginaryPart,
            NumericDomain::Real,
        ),
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("numeric component entry");
        assert_eq!(
            entry.contract.inference_rule,
            BuiltinInferenceRule::Math(MathInferenceRule::NumericComponent(rule))
        );
        let inference = infer_catalog_call(
            entry,
            &CallRequest {
                arguments: vec![input.clone()],
                literals: Default::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        );
        assert!(inference.diagnostics.is_empty(), "{name}");
        let output = &inference.outputs[0];
        assert_eq!(
            output.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: expected_domain,
            }),
            "{name}"
        );
        assert_eq!(output.shape, input.shape, "{name}");
        assert_eq!(output.storage, StorageFact::Unknown, "{name}");
        assert_eq!(output.residency, ResidencyFact::Unknown, "{name}");
        assert_eq!(output.layout, LayoutFact::Unknown, "{name}");
        assert_eq!(output.contiguity, ContiguityFact::Unknown, "{name}");
        assert_eq!(output.view, ViewFact::Materialized, "{name}");
        assert_eq!(output.alias, AliasFact::Unknown, "{name}");
        assert_eq!(output.mutation, MutationFact::ValueSemantics, "{name}");
    }

    let mut real_input = input;
    let ValueKindFact::Numeric(numeric) = &mut real_input.kind else {
        unreachable!()
    };
    numeric.domain = NumericDomain::Real;
    let identity = infer_catalog_call(
        builtin_catalog_entry_by_name("conj").expect("conj entry"),
        &CallRequest {
            arguments: vec![real_input.clone()],
            literals: Default::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(identity.outputs[0], real_input);
}

#[test]
fn floating_tensor_components_do_not_promise_host_representation_after_preparation() {
    use runmat_types::{
        CallRequest, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };
    for name in ["conj", "real", "imag"] {
        for class in [NumericClass::Double, NumericClass::Single] {
            for shape in [vec![Some(1), Some(1)], vec![Some(2), Some(3)]] {
                let input = ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class,
                        domain: NumericDomain::Real,
                    }),
                    ShapeFact::from(shape),
                    StorageFact::Dense,
                );
                let inferred = infer_catalog_call(
                    builtin_catalog_entry_by_name(name).unwrap(),
                    &CallRequest {
                        arguments: vec![input.clone()],
                        literals: Default::default(),
                        outputs: OutputSelection::new(RequestedOutputCount::One),
                    },
                );
                assert!(inferred.diagnostics.is_empty());
                let output = &inferred.outputs[0];
                assert_eq!(output.kind, input.kind);
                assert_eq!(output.shape, input.shape);
                assert_eq!(output.residency, ResidencyFact::Unknown);
                assert_eq!(output.storage, StorageFact::Unknown);
            }
        }
    }
}
