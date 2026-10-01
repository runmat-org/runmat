use super::*;
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn real(shape: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape),
        StorageFact::Dense,
    )
}

fn call(arguments: Vec<ValueFact>, outputs: usize) -> runmat_types::CallInference {
    inference::infer(
        &CallRequest {
            literals: LiteralContext::new(Vec::new()),
            arguments,
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &QQPLOT_CATALOG_ENTRY,
    )
}

#[test]
fn vector_sample_proves_host_handle_class_and_ui_effect_not_handle_count() {
    let result = call(vec![real(vec![Some(4), Some(1)])], 1);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
    assert!(result
        .effects
        .0
        .contains(&runmat_types::EffectKind::UserInterface));
}

#[test]
fn sample_shape_and_percentage_domain_reject_without_proven_output() {
    let matrix_3d = call(vec![real(vec![Some(2), Some(2), Some(2)])], 1);
    assert!(!matrix_3d.diagnostics.is_empty());
    assert!(matches!(matrix_3d.outputs[0].kind, ValueKindFact::Unknown));

    let complex_p = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let invalid = call(
        vec![
            real(vec![Some(3), Some(1)]),
            real(vec![Some(0), Some(0)]),
            complex_p,
        ],
        1,
    );
    assert!(!invalid.diagnostics.is_empty());
    assert!(matches!(invalid.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn known_sparse_sample_or_percentages_do_not_prove_a_handle() {
    let mut sparse = real(vec![Some(3), Some(1)]);
    sparse.storage = StorageFact::Sparse;
    let sample = call(vec![sparse.clone()], 1);
    assert!(!sample.diagnostics.is_empty());
    assert!(matches!(sample.outputs[0].kind, ValueKindFact::Unknown));

    let percentages = call(
        vec![
            real(vec![Some(3), Some(1)]),
            real(vec![Some(0), Some(0)]),
            sparse,
        ],
        1,
    );
    assert!(!percentages.diagnostics.is_empty());
    assert!(matches!(
        percentages.outputs[0].kind,
        ValueKindFact::Unknown
    ));
}

#[test]
fn possible_axes_and_excess_outputs_remain_conservative() {
    let possible_axes = call(vec![real(vec![Some(1), Some(1)])], 1);
    assert_eq!(
        possible_axes.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(possible_axes.outputs[0].shape, ShapeFact::Unknown);
    let excess = call(vec![real(vec![Some(3), Some(1)])], 2);
    assert!(!excess.diagnostics.is_empty());
    assert!(matches!(excess.outputs[1].kind, ValueKindFact::Unknown));
}
