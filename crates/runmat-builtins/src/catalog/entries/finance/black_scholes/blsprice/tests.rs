use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn real(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

#[test]
fn scalar_broadcast_and_output_count_are_sound() {
    let args = vec![
        real(ShapeFact::from(vec![Some(3), Some(1)])),
        real(ShapeFact::Scalar),
        real(ShapeFact::Scalar),
        real(ShapeFact::Scalar),
        real(ShapeFact::Scalar),
    ];
    let inferred = crate::infer_catalog_call(&BLSPRICE_CATALOG_ENTRY, &request(args.clone(), 2));
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(inferred.outputs.len(), 2);
    for output in inferred.outputs {
        assert_eq!(output.shape, ShapeFact::from(vec![Some(3), Some(1)]));
        assert_eq!(
            output.numeric().map(|numeric| numeric.class),
            Some(NumericClass::Double)
        );
        assert_eq!(output.residency, ResidencyFact::Host);
    }
    let excess = crate::infer_catalog_call(&BLSPRICE_CATALOG_ENTRY, &request(args, 3));
    assert!(excess
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-BLSPRICE-OUTPUTS"));
}

#[test]
fn resident_result_keeps_known_double_class_without_claiming_placement() {
    let mut resident = real(ShapeFact::from(vec![Some(3), Some(1)]));
    resident.residency = ResidencyFact::Unknown;
    let args = vec![
        resident,
        real(ShapeFact::Scalar),
        real(ShapeFact::Scalar),
        real(ShapeFact::Scalar),
        real(ShapeFact::Scalar),
    ];
    let inferred = crate::infer_catalog_call(&BLSPRICE_CATALOG_ENTRY, &request(args, 1));
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(
        inferred.outputs[0].numeric().map(|numeric| numeric.class),
        Some(NumericClass::Double)
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Unknown);
}

#[test]
fn single_arrays_and_scalar_double_controls_select_single_independent_of_residency() {
    let mut single = real(ShapeFact::from(vec![Some(3), Some(1)]));
    single.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Real,
    });
    single.residency = ResidencyFact::Device { provider: None };
    let scalar = real(ShapeFact::Scalar);
    let mut args = vec![
        single,
        scalar.clone(),
        scalar.clone(),
        scalar.clone(),
        scalar.clone(),
    ];
    let selected = crate::infer_catalog_call(&BLSPRICE_CATALOG_ENTRY, &request(args.clone(), 2));
    assert!(
        selected.diagnostics.is_empty(),
        "{:?}",
        selected.diagnostics
    );
    assert!(selected
        .outputs
        .iter()
        .all(|value| value.numeric().map(|numeric| numeric.class) == Some(NumericClass::Single)));
    assert!(selected
        .outputs
        .iter()
        .all(|value| value.residency == ResidencyFact::Unknown));
    args[1] = real(ShapeFact::from(vec![Some(3), Some(1)]));
    let mixed = crate::infer_catalog_call(&BLSPRICE_CATALOG_ENTRY, &request(args, 1));
    assert_eq!(
        mixed.outputs[0].numeric().map(|numeric| numeric.class),
        Some(NumericClass::Double)
    );
}

#[test]
fn unresolved_double_geometry_does_not_claim_single() {
    let mut single = real(ShapeFact::from(vec![Some(3), Some(1)]));
    single.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Real,
    });
    let mut scalar = real(ShapeFact::Unknown);
    scalar.residency = ResidencyFact::Unknown;
    let result = crate::infer_catalog_call(
        &BLSPRICE_CATALOG_ENTRY,
        &request(
            vec![
                single,
                scalar.clone(),
                scalar.clone(),
                scalar.clone(),
                scalar,
            ],
            1,
        ),
    );
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn known_empty_host_yield_normalizes_before_class_and_type_inference() {
    let mut single = real(ShapeFact::from(vec![Some(2), Some(1)]));
    single.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Real,
    });
    let required = vec![
        single,
        real(ShapeFact::Scalar),
        real(ShapeFact::Scalar),
        real(ShapeFact::Scalar),
        real(ShapeFact::Scalar),
    ];
    let empty = ShapeFact::from(vec![Some(0), Some(0)]);
    let empty_cell = ValueKindFact::Cell(CellFact {
        element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
        elements: Vec::new(),
        elements_complete: true,
    });
    for kind in [
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ValueKindFact::String,
        empty_cell,
    ] {
        let mut args = required.clone();
        args.push(ValueFact::proven(kind, empty.clone(), StorageFact::Dense));
        let inferred = crate::infer_catalog_call(&BLSPRICE_CATALOG_ENTRY, &request(args, 1));
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert_eq!(
            inferred.outputs[0].numeric().map(|numeric| numeric.class),
            Some(NumericClass::Single)
        );
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(1)])
        );
    }
}

#[test]
fn potentially_empty_text_yield_has_conservative_class_without_a_false_error() {
    let mut single = real(ShapeFact::from(vec![Some(2), Some(1)]));
    single.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Real,
    });
    let mut args = vec![single];
    args.extend((0..4).map(|_| real(ShapeFact::Scalar)));
    args.push(ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::Unknown,
        StorageFact::Dense,
    ));
    let inferred = crate::infer_catalog_call(&BLSPRICE_CATALOG_ENTRY, &request(args, 1));
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn unknown_host_empty_yield_does_not_infer_an_empty_price_result() {
    let mut single = real(ShapeFact::from(vec![Some(2), Some(1)]));
    single.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Real,
    });
    let mut args = vec![single];
    args.extend((0..4).map(|_| real(ShapeFact::Scalar)));
    let mut yield_value = ValueFact::unknown(DynamicReason::RuntimeValue);
    yield_value.shape = ShapeFact::from(vec![Some(0), Some(0)]);
    yield_value.residency = ResidencyFact::Host;
    args.push(yield_value);
    let inferred = crate::infer_catalog_call(&BLSPRICE_CATALOG_ENTRY, &request(args, 1));
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
}

#[test]
fn unknown_size_integer_or_logical_yield_keeps_precision_conditional_on_emptiness() {
    let mut single = real(ShapeFact::from(vec![Some(2), Some(1)]));
    single.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Real,
    });
    for kind in [
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ValueKindFact::Logical,
    ] {
        let mut args = vec![single.clone()];
        args.extend((0..4).map(|_| real(ShapeFact::Scalar)));
        args.push(ValueFact::proven(
            kind,
            ShapeFact::Unknown,
            StorageFact::Dense,
        ));
        let inferred = crate::infer_catalog_call(&BLSPRICE_CATALOG_ENTRY, &request(args, 1));
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
        assert_eq!(inferred.outputs[0].shape, ShapeFact::Unknown);
    }
}

#[test]
fn empty_complex_or_sparse_yield_does_not_bypass_input_admission() {
    for (numeric, storage) in [
        (
            NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Complex,
            },
            StorageFact::Dense,
        ),
        (
            NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            },
            StorageFact::Sparse,
        ),
    ] {
        let mut args = (0..5).map(|_| real(ShapeFact::Scalar)).collect::<Vec<_>>();
        args.push(ValueFact::proven(
            ValueKindFact::Numeric(numeric),
            ShapeFact::from(vec![Some(0), Some(0)]),
            storage,
        ));
        let inferred = crate::infer_catalog_call(&BLSPRICE_CATALOG_ENTRY, &request(args, 1));
        assert!(inferred
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-BLSPRICE-INPUT"));
    }
}

#[test]
fn legacy_visible_examples_and_provider_contract_are_present() {
    assert_eq!(BLSPRICE_CATALOG_ENTRY.documentation.examples.len(), 2);
    for example in BLSPRICE_CATALOG_ENTRY.documentation.examples {
        assert!(example.program.contains("fprintf"));
        assert!(example.display_output.is_some());
        assert!(matches!(
            example.verification,
            BuiltinExampleVerification::Assertions { .. }
        ));
    }
    assert_eq!(
        BLSPRICE_CATALOG_ENTRY.placement.accelerator,
        BuiltinAcceleratorPolicy::Optional
    );
    assert_eq!(
        BLSPRICE_CATALOG_ENTRY.extensions,
        &super::super::contract::BLSPRICE_EXTENSIONS
    );
}
