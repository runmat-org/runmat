use super::*;

#[test]
fn signum_contract_preserves_numeric_facts_and_types_conversion_boundaries() {
    use runmat_types::{
        AliasFact, CallRequest, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
        ViewFact,
    };

    let request = |argument| CallRequest {
        arguments: vec![argument],
        literals: Default::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let entry = builtin_catalog_entry_by_name("sign").expect("sign entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::MagnitudePhaseSign(
            MagnitudePhaseSignKind::Sign,
        ))
    );
    for (class, domain) in [
        (NumericClass::UInt64, NumericDomain::Real),
        (NumericClass::Single, NumericDomain::Complex),
    ] {
        let mut input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact { class, domain }),
            ShapeFact::from(vec![Some(4), Some(2)]),
            StorageFact::Dense,
        );
        input.residency = ResidencyFact::Device {
            provider: Some("sign-provider".into()),
        };
        let inference = infer_catalog_call(entry, &request(input.clone()));
        assert!(inference.diagnostics.is_empty(), "{class:?} {domain:?}");
        let output = &inference.outputs[0];
        assert_eq!(output.kind, input.kind);
        assert_eq!(output.shape, input.shape);
        assert_eq!(output.residency, input.residency);
        assert_eq!(output.view, ViewFact::Materialized);
        assert_eq!(output.alias, AliasFact::Unique);
    }

    let mut logical = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    logical.residency = ResidencyFact::Device {
        provider: Some("single-provider".into()),
    };
    let converted = infer_catalog_call(entry, &request(logical));
    assert_eq!(
        converted.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(converted.outputs[0].residency, ResidencyFact::Unknown);

    let rejected = infer_catalog_call(
        entry,
        &request(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int32,
            domain: NumericDomain::Complex,
        }))),
    );
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-SIGNUM-COMPLEX-INTEGER"));
}
