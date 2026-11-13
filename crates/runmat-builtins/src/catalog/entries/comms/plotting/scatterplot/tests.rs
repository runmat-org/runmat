use super::*;
use runmat_types::*;

fn samples(class: NumericClass, domain: NumericDomain) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(vec![Some(20), Some(1)]),
        StorageFact::Dense,
    )
}

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallInference {
    crate::infer_catalog_call(
        &SCATTERPLOT_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    )
}

#[test]
fn scatterplot_returns_local_double_handle_for_complex_samples() {
    let result = infer(
        vec![samples(NumericClass::Single, NumericDomain::Complex)],
        vec![LiteralValue::Unknown],
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].numeric().map(|n| n.class),
        Some(NumericClass::Double)
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn scatterplot_rejects_offset_outside_decimation_interval() {
    let count = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let result = infer(
        vec![
            samples(NumericClass::Double, NumericDomain::Complex),
            count.clone(),
            count,
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(4.0),
            LiteralValue::Number(4.0),
        ],
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-SCATTERPLOT-OFFSET"));
}

#[test]
fn scatterplot_rejects_known_invalid_figure_class() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let result = infer(
        vec![
            samples(NumericClass::Double, NumericDomain::Complex),
            unknown.clone(),
            unknown.clone(),
            unknown,
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Int64,
                domain: NumericDomain::Real,
            })),
        ],
        vec![LiteralValue::Unknown; 5],
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-SCATTERPLOT-FIGURE"));
}
