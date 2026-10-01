use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
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

fn call(
    args: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
    outputs: usize,
) -> runmat_types::CallInference {
    inference::infer(
        &CallRequest {
            arguments: args,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &BOXPLOT_CATALOG_ENTRY,
    )
}

#[test]
fn canonical_entry_preserves_descriptor_extensions_and_integer_forms() {
    assert_eq!(BOXPLOT_CATALOG_ENTRY.extensions, &BOXPLOT_EXTENSIONS);
    assert_eq!(
        BOXPLOT_CATALOG_ENTRY
            .integer_capabilities
            .iter()
            .map(|capability| capability.form)
            .collect::<Vec<_>>(),
        BOXPLOT_INTEGER_CAPABILITIES
            .iter()
            .map(|capability| capability.form)
            .collect::<Vec<_>>()
    );
    assert_eq!(BOXPLOT_CATALOG_ENTRY.identity.name, "boxplot");
}

#[test]
fn known_valid_matrix_proves_row_vector_double_handle_class() {
    let result = call(vec![real(vec![Some(4), Some(2)])], vec![], 1);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
    assert_eq!(result.outputs[0].storage, StorageFact::Dense);
}

#[test]
fn invalid_data_and_group_lengths_do_not_prove_handles() {
    let invalid_rank = call(vec![real(vec![Some(2), Some(2), Some(2)])], vec![], 1);
    assert!(!invalid_rank.diagnostics.is_empty());
    assert!(matches!(
        invalid_rank.outputs[0].kind,
        ValueKindFact::Unknown
    ));

    let bad_group = call(
        vec![real(vec![Some(4), Some(2)]), real(vec![Some(3), Some(1)])],
        vec![],
        1,
    );
    assert!(!bad_group.diagnostics.is_empty());
    assert!(matches!(bad_group.outputs[0].kind, ValueKindFact::Unknown));

    let row_group = call(
        vec![real(vec![Some(4), Some(2)]), real(vec![Some(4), Some(1)])],
        vec![],
        1,
    );
    assert!(row_group.diagnostics.is_empty());
}

#[test]
fn typed_option_name_and_invalid_host_value_are_diagnosed() {
    let text = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let result = call(
        vec![
            real(vec![Some(4), Some(1)]),
            text,
            real(vec![Some(1), Some(1)]),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("Whisker".into()),
            LiteralValue::Number(-1.0),
        ],
        1,
    );
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn unrepresentable_plot_size_is_not_proven_as_an_admitted_option() {
    let text = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let result = call(
        vec![
            real(vec![Some(4), Some(1)]),
            text,
            real(vec![Some(1), Some(1)]),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("OutlierSize".into()),
            LiteralValue::Number(f64::MAX),
        ],
        1,
    );
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn unknown_values_and_padded_requested_outputs_remain_conservative() {
    let result = call(
        vec![ValueFact::unknown(
            runmat_types::DynamicReason::RuntimeValue,
        )],
        vec![],
        2,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert!(matches!(result.outputs[1].kind, ValueKindFact::Unknown));
}
