use super::*;

#[test]
fn phase_angle_contract_preserves_float_class_shape_and_residency() {
    use runmat_types::{
        AliasFact, CallRequest, ContiguityFact, LayoutFact, NumericClass, NumericDomain,
        NumericFact, OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact,
        ValueFact, ValueKindFact, ViewFact,
    };

    let mut input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    input.residency = ResidencyFact::Device {
        provider: Some("phase-provider".into()),
    };
    let entry = builtin_catalog_entry_by_name("angle").expect("angle entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::MagnitudePhaseSign(
            MagnitudePhaseSignKind::Phase,
        ))
    );
    let inference = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![input.clone()],
            literals: Default::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(inference.diagnostics.is_empty());
    let output = &inference.outputs[0];
    assert_eq!(
        output.kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(output.shape, input.shape);
    assert_eq!(output.storage, StorageFact::Dense);
    assert_eq!(output.residency, input.residency);
    assert_eq!(output.layout, LayoutFact::ColumnMajor);
    assert_eq!(output.contiguity, ContiguityFact::Contiguous);
    assert_eq!(output.view, ViewFact::Materialized);
    assert_eq!(output.alias, AliasFact::Unique);

    let rejected = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }))],
            literals: Default::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-PHASE-ANGLE-INPUT"));
}
