use super::*;

#[test]
fn log2_contract_distinguishes_logarithm_and_dissection_outputs() {
    use runmat_types::{
        CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
        OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };

    let entry = builtin_catalog_entry_by_name("log2").expect("log2 entry");
    assert!(matches!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::Logarithm(LogarithmKind::Binary))
    ));
    assert_eq!(
        entry.descriptor.output_mode,
        BuiltinOutputMode::ByRequestedOutputCount
    );

    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let one = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![input.clone()],
            literals: LiteralContext::new(vec![LiteralValue::Number(-1.0)]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(one.diagnostics.is_empty());
    assert_eq!(
        one.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        })
    );

    let two = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        },
    );
    assert!(two.diagnostics.is_empty());
    assert_eq!(two.outputs.len(), 2);
    for output in &two.outputs {
        assert_eq!(
            output.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            })
        );
        assert_eq!(output.shape, ShapeFact::from(vec![Some(2), Some(3)]));
    }

    let integer = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }))],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        },
    );
    assert!(integer.diagnostics.is_empty());
    assert!(integer.outputs.iter().all(|output| {
        output.kind
            == ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })
    }));
}

#[test]
fn log2_dissection_contract_rejects_complex_gpu_and_excess_outputs() {
    use runmat_types::{
        CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ResidencyFact, ValueFact, ValueKindFact,
    };

    let entry = builtin_catalog_entry_by_name("log2").expect("log2 entry");
    let request = |argument, requested| CallRequest {
        arguments: vec![argument],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(requested),
    };
    let complex = infer_catalog_call(
        entry,
        &request(
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Complex,
            })),
            RequestedOutputCount::Exactly(2),
        ),
    );
    assert!(complex
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOG2-COMPLEX-DISSECTION"));

    let mut resident = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    resident.residency = ResidencyFact::Device { provider: None };
    let gpu = infer_catalog_call(entry, &request(resident, RequestedOutputCount::Exactly(2)));
    assert!(gpu
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOG2-GPU-DISSECTION"));

    let excessive = infer_catalog_call(
        entry,
        &request(
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
            RequestedOutputCount::Exactly(3),
        ),
    );
    assert!(excessive
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}
