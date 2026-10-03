use super::*;

#[test]
fn resident_readback_and_provider_routes_remain_suspending_contracts() {
    assert_eq!(
        FILTER2_CATALOG_ENTRY.contract.async_behavior,
        BuiltinAsyncBehavior::MaySuspend
    );
    assert!(FILTER2_CATALOG_ENTRY
        .contract
        .effects
        .contains(&EffectKind::MaySuspend));
}
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn numeric(class: NumericClass, rows: usize, cols: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(cols)]),
        StorageFact::Dense,
    )
}

fn call(kernel: ValueFact, image: ValueFact, options: &[&str]) -> runmat_types::CallInference {
    let mut arguments = vec![kernel, image];
    arguments.extend(
        options
            .iter()
            .map(|_| ValueFact::scalar(ValueKindFact::String)),
    );
    let mut literals = vec![LiteralValue::Unknown, LiteralValue::Unknown];
    literals.extend(
        options
            .iter()
            .map(|name| LiteralValue::String((*name).into())),
    );
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        &FILTER2_CATALOG_ENTRY,
    )
}

#[test]
fn known_shape_and_class_follow_options_and_single_promotion() {
    let kernel = numeric(NumericClass::Single, 2, 3);
    let image = numeric(NumericClass::Double, 4, 5);
    for (options, expected) in [
        (vec![], vec![Some(4), Some(5)]),
        (vec!["full"], vec![Some(5), Some(7)]),
        (vec!["valid"], vec![Some(3), Some(3)]),
        (vec!["full", "same"], vec![Some(4), Some(5)]),
    ] {
        let inferred = call(kernel.clone(), image.clone(), &options);
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert_eq!(inferred.outputs[0].shape, ShapeFact::from(expected));
        assert_eq!(
            inferred.outputs[0].numeric().unwrap().class,
            NumericClass::Single
        );
    }
}

#[test]
fn unknown_option_does_not_invent_geometry_and_invalid_option_reports() {
    let kernel = numeric(NumericClass::Double, 2, 2);
    let image = numeric(NumericClass::Double, 4, 5);
    let mut request = CallRequest {
        arguments: vec![
            kernel.clone(),
            image.clone(),
            ValueFact::scalar(ValueKindFact::Unknown),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let unknown = inference::infer(&request, &FILTER2_CATALOG_ENTRY);
    assert_eq!(unknown.outputs[0].shape, ShapeFact::Unknown);
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("replicate".into()),
    ]);
    let invalid = inference::infer(&request, &FILTER2_CATALOG_ENTRY);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-FILTER2-OPTION"));
}

#[test]
fn unknown_operand_does_not_prove_double_and_empty_kernel_is_diagnosed() {
    let unknown = call(
        ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
        numeric(NumericClass::Double, 4, 5),
        &[],
    );
    assert_eq!(unknown.outputs[0].kind, ValueKindFact::Unknown);
    let empty = call(
        numeric(NumericClass::Double, 0, 2),
        numeric(NumericClass::Double, 4, 5),
        &[],
    );
    assert!(empty
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-FILTER2-EMPTY"));
}
