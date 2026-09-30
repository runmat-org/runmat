use super::*;
use runmat_types::*;

fn value(shape: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, count: usize) -> CallInference {
    super::inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        },
        &HISTCOUNTS_CATALOG_ENTRY,
    )
}

#[test]
fn explicit_edges_and_bin_counts_produce_double_row_outputs() {
    let result = infer(
        vec![value(&[5, 1]), value(&[1, 6])],
        vec![LiteralValue::Unknown, LiteralValue::Unknown],
        2,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, value(&[1, 5]).shape);
    assert_eq!(result.outputs[1].shape, value(&[1, 6]).shape);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    let count = infer(
        vec![value(&[5, 1]), value(&[1, 1])],
        vec![LiteralValue::Unknown, LiteralValue::Number(3.)],
        2,
    );
    assert_eq!(count.outputs[0].shape, value(&[1, 3]).shape);
}

#[test]
fn defaults_and_dynamic_controls_remain_unknown() {
    assert_eq!(
        infer(vec![value(&[5, 1])], vec![LiteralValue::Unknown], 2).outputs[0].shape,
        ShapeFact::Unknown
    );
    let dynamic = infer(
        vec![value(&[5, 1]), value(&[1, 1])],
        vec![LiteralValue::Unknown, LiteralValue::Unknown],
        2,
    );
    assert_eq!(dynamic.outputs[0].shape, ShapeFact::Unknown);
    let huge = infer(
        vec![value(&[5, 1]), value(&[1, 1])],
        vec![LiteralValue::Unknown, LiteralValue::Number(f64::MAX)],
        2,
    );
    assert_eq!(huge.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn descriptor_covers_core_forms() {
    let labels: Vec<&str> = HISTCOUNTS_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"N = histcounts(X)"));
    assert!(labels.contains(&"N = histcounts(X, bins)"));
    assert!(labels.contains(&"[N, edges] = histcounts(X, Name, Value, ...)"));
}
