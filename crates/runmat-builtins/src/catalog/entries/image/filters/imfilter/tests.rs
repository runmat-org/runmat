use super::*;

#[test]
fn resident_kernel_and_host_fallback_remain_suspending_contracts() {
    assert_eq!(
        IMFILTER_CATALOG_ENTRY.contract.async_behavior,
        BuiltinAsyncBehavior::MaySuspend
    );
    assert!(IMFILTER_CATALOG_ENTRY
        .contract
        .effects
        .contains(&EffectKind::MaySuspend));
}
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn numeric(class: NumericClass, shape: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}

fn call(image: ValueFact, kernel: ValueFact, option: Option<&str>) -> runmat_types::CallInference {
    let mut arguments = vec![image, kernel];
    let mut literals = vec![LiteralValue::Unknown, LiteralValue::Unknown];
    if let Some(option) = option {
        arguments.push(ValueFact::scalar(ValueKindFact::String));
        literals.push(LiteralValue::String(option.into()));
    }
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        &IMFILTER_CATALOG_ENTRY,
    )
}

#[test]
fn shape_modes_and_integer_class_follow_image() {
    let image = numeric(NumericClass::UInt16, &[5, 6, 3]);
    let kernel = numeric(NumericClass::Double, &[2, 3]);
    for (option, dims) in [
        (None, vec![Some(5), Some(6), Some(3)]),
        (Some("FULL"), vec![Some(6), Some(8), Some(3)]),
        (Some("valid"), vec![Some(4), Some(4), Some(3)]),
    ] {
        let inferred = call(image.clone(), kernel.clone(), option);
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert_eq!(inferred.outputs[0].shape, ShapeFact::from(dims));
        assert_eq!(
            inferred.outputs[0].numeric().unwrap().class,
            NumericClass::UInt16
        );
    }
}

#[test]
fn logical_image_preserves_logical_output_kind() {
    let image = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(3), Some(4)]),
        StorageFact::Dense,
    );
    let inferred = call(image, numeric(NumericClass::Double, &[2, 2]), None);
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(4)])
    );
}

#[test]
fn invalid_kernel_class_rank_and_image_width_are_diagnosed() {
    let invalid_kernel = call(
        numeric(NumericClass::Single, &[4, 5]),
        numeric(NumericClass::Single, &[2, 2, 2]),
        None,
    );
    assert!(invalid_kernel
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-IMFILTER-KERNEL"));
    assert!(invalid_kernel
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-IMFILTER-KERNEL-SHAPE"));
    let invalid_image = call(
        numeric(NumericClass::UInt64, &[4, 5]),
        numeric(NumericClass::Double, &[2, 2]),
        None,
    );
    assert!(invalid_image
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-IMFILTER-CLASS"));
}

#[test]
fn unknown_image_class_stays_dynamic_and_unknown_option_does_not_preserve_shape() {
    let mut request = CallRequest {
        arguments: vec![
            ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
            numeric(NumericClass::Double, &[2, 2]),
            ValueFact::scalar(ValueKindFact::Unknown),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let unknown = inference::infer(&request, &IMFILTER_CATALOG_ENTRY);
    assert_eq!(unknown.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(unknown.outputs[0].shape, ShapeFact::Unknown);
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("invalid".into()),
    ]);
    let invalid = inference::infer(&request, &IMFILTER_CATALOG_ENTRY);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-IMFILTER-OPTION"));
}
