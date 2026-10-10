use super::*;
use crate::{BuiltinExampleVerification, REQUIRED_DEFAULT_BINDING};
use runmat_types::*;

fn call(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn numeric(class: NumericClass, shape: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape),
        StorageFact::Dense,
    )
}

#[test]
fn every_fft_identity_has_one_default_binding_and_runtime_descriptor() {
    for (name, entry) in [
        ("fft", &FFT_CATALOG_ENTRY),
        ("fft2", &FFT2_CATALOG_ENTRY),
        ("fftn", &FFTN_CATALOG_ENTRY),
        ("ifft", &IFFT_CATALOG_ENTRY),
        ("ifft2", &IFFT2_CATALOG_ENTRY),
        ("ifftn", &IFFTN_CATALOG_ENTRY),
        ("fftshift", &FFTSHIFT_CATALOG_ENTRY),
        ("ifftshift", &IFFTSHIFT_CATALOG_ENTRY),
    ] {
        assert_eq!(entry.identity.name, name);
        assert_eq!(entry.bindings, &REQUIRED_DEFAULT_BINDING);
        assert!(!entry.descriptor.signatures.is_empty());
        assert!(!entry.descriptor.errors.is_empty());
        assert!(!entry.documentation.examples.is_empty());
        assert!(entry.documentation.examples.iter().all(|example| matches!(
            example.verification,
            BuiltinExampleVerification::Assertions { .. }
        )));
    }
}

#[test]
fn known_fft_class_and_shape_follow_runtime_numeric_boundary() {
    let input = numeric(NumericClass::Single, vec![Some(2), Some(3), Some(4)]);
    let result = inference::infer(
        inference::Transform::Fftn,
        &call(vec![input.clone()], vec![]),
        &FFTN_CATALOG_ENTRY,
    );
    assert_eq!(result.outputs[0].shape, input.shape);
    assert_eq!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex
        })
    );
    let integer = numeric(NumericClass::Int16, vec![Some(2), Some(3)]);
    let result = inference::infer(
        inference::Transform::Fft2,
        &call(vec![integer], vec![]),
        &FFT2_CATALOG_ENTRY,
    );
    assert_eq!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex
        })
    );
}

#[test]
fn two_dimensional_transform_preserves_or_pads_known_rank() {
    for (transform, entry) in [
        (inference::Transform::Fft2, &FFT2_CATALOG_ENTRY),
        (inference::Transform::Ifft2, &IFFT2_CATALOG_ENTRY),
    ] {
        let input = numeric(NumericClass::Double, vec![Some(4)]);
        let result = inference::infer(transform, &call(vec![input], vec![]), entry);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(4), Some(1)])
        );
    }
}

#[test]
fn length_literal_refines_only_selected_fft_axis() {
    let input = numeric(NumericClass::Double, vec![Some(2), Some(3)]);
    let length = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let known = call(
        vec![input.clone(), length.clone()],
        vec![LiteralValue::Unknown, LiteralValue::Number(5.0)],
    );
    let result = inference::infer(inference::Transform::Fft, &known, &FFT_CATALOG_ENTRY);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(5), Some(3)])
    );
    let unknown = call(vec![input, length], vec![]);
    let result = inference::infer(inference::Transform::Fft, &unknown, &FFT_CATALOG_ENTRY);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![None, Some(3)])
    );
}

#[test]
fn size_vector_shape_is_not_mistaken_for_its_contents() {
    let input = numeric(NumericClass::Double, vec![Some(2), Some(3), Some(4)]);
    let size = numeric(NumericClass::Double, vec![Some(1), Some(3)]);
    let result = inference::infer(
        inference::Transform::Fftn,
        &call(vec![input, size], vec![]),
        &FFTN_CATALOG_ENTRY,
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn short_nd_size_changes_only_selected_axes_and_keeps_trailing_facts() {
    let input = numeric(NumericClass::Double, vec![Some(2), None, Some(4)]);
    let size = numeric(NumericClass::Double, vec![Some(1), Some(2)]);
    let request = call(
        vec![input, size],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Vector(vec![LiteralValue::Number(5.0), LiteralValue::Number(6.0)]),
        ],
    );
    for (transform, entry) in [
        (inference::Transform::Fftn, &FFTN_CATALOG_ENTRY),
        (inference::Transform::Ifftn, &IFFTN_CATALOG_ENTRY),
    ] {
        let result = inference::infer(transform, &request, entry);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(5), Some(6), Some(4)])
        );
    }
}

#[test]
fn typed_fft_lengths_use_represented_single_value_and_runtime_epsilon() {
    let input = numeric(NumericClass::Single, vec![Some(2), Some(3)]);
    let length = numeric(NumericClass::Single, vec![Some(1), Some(1)]);
    let request = call(
        vec![input.clone(), length],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Real {
                text: "16777217".into(),
                class: NumericClass::Single,
            },
        ],
    );
    let inferred = inference::infer(inference::Transform::Fft, &request, &FFT_CATALOG_ENTRY);
    assert!(
        inferred.diagnostics.is_empty(),
        "{:#?}",
        inferred.diagnostics
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(16_777_216), Some(3)])
    );

    let length = numeric(NumericClass::Double, vec![Some(1), Some(1)]);
    let request = call(
        vec![input, length],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(1.0 + f64::EPSILON),
        ],
    );
    let inferred = inference::infer(inference::Transform::Fft, &request, &FFT_CATALOG_ENTRY);
    assert!(
        inferred.diagnostics.is_empty(),
        "{:#?}",
        inferred.diagnostics
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(3)])
    );
}

#[test]
fn fft_dimension_and_nd_size_keep_their_distinct_runtime_tolerances() {
    let input = numeric(NumericClass::Double, vec![Some(2), Some(3)]);
    let length = numeric(NumericClass::Double, vec![Some(1), Some(1)]);
    let dimension = numeric(NumericClass::Double, vec![Some(1), Some(1)]);
    let request = call(
        vec![input.clone(), length, dimension],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(5.0),
            LiteralValue::Real {
                text: "2.0000005".into(),
                class: NumericClass::Double,
            },
        ],
    );
    let inferred = inference::infer(inference::Transform::Fft, &request, &FFT_CATALOG_ENTRY);
    assert!(
        inferred.diagnostics.is_empty(),
        "{:#?}",
        inferred.diagnostics
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(5)])
    );

    let fractional_length = call(
        vec![
            numeric(NumericClass::Double, vec![Some(2), Some(3)]),
            numeric(NumericClass::Double, vec![Some(1), Some(1)]),
        ],
        vec![LiteralValue::Unknown, LiteralValue::Number(2.0000005)],
    );
    assert!(!inference::infer(
        inference::Transform::Fft,
        &fractional_length,
        &FFT_CATALOG_ENTRY,
    )
    .diagnostics
    .is_empty());

    let sizes = numeric(NumericClass::Single, vec![Some(1), Some(2)]);
    let request = call(
        vec![input, sizes],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Vector(vec![
                LiteralValue::Real {
                    text: "16777217".into(),
                    class: NumericClass::Single,
                },
                LiteralValue::Number(4.0),
            ]),
        ],
    );
    let inferred = inference::infer(inference::Transform::Fftn, &request, &FFTN_CATALOG_ENTRY);
    assert!(
        inferred.diagnostics.is_empty(),
        "{:#?}",
        inferred.diagnostics
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(16_777_216), Some(4)])
    );

    let invalid = call(
        vec![
            numeric(NumericClass::Double, vec![Some(2), Some(3)]),
            numeric(NumericClass::Double, vec![Some(1), Some(1)]),
        ],
        vec![LiteralValue::Unknown, LiteralValue::Number(2.25)],
    );
    assert!(
        !inference::infer(inference::Transform::Fftn, &invalid, &FFTN_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}

#[test]
fn undecidable_default_axis_and_inverse_option_do_not_invent_geometry_or_domain() {
    let input = numeric(NumericClass::Double, vec![None, Some(3)]);
    let length = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let result = inference::infer(
        inference::Transform::Fft,
        &call(
            vec![input, length],
            vec![LiteralValue::Unknown, LiteralValue::Number(5.0)],
        ),
        &FFT_CATALOG_ENTRY,
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);

    let input = numeric(NumericClass::Single, vec![Some(2), Some(3)]);
    let unknown_flag = ValueFact::unknown(DynamicReason::RuntimeValue);
    let result = inference::infer(
        inference::Transform::Ifft,
        &call(vec![input, unknown_flag], vec![]),
        &IFFT_CATALOG_ENTRY,
    );
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn ifftn_size_then_final_symmetry_flag_is_a_valid_three_argument_form() {
    let input = numeric(NumericClass::Single, vec![Some(2), Some(3)]);
    let size = numeric(NumericClass::Double, vec![Some(1), Some(2)]);
    let flag = ValueFact::scalar(ValueKindFact::Character);
    let request = call(
        vec![input, size, flag],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Vector(vec![LiteralValue::Number(4.0), LiteralValue::Number(5.0)]),
            LiteralValue::Character("symmetric".into()),
        ],
    );
    let result = inference::infer(inference::Transform::Ifftn, &request, &IFFTN_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:#?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(4), Some(5)])
    );
    assert_eq!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })
    );
}

#[test]
fn misplaced_flags_complex_dimensions_and_nonvector_size_are_diagnosed() {
    let input = numeric(NumericClass::Double, vec![Some(2), Some(3)]);
    let flag = ValueFact::scalar(ValueKindFact::Character);
    let length = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let misplaced = call(
        vec![input.clone(), flag, length],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Character("symmetric".into()),
            LiteralValue::Number(5.0),
        ],
    );
    assert!(
        !inference::infer(inference::Transform::Ifft, &misplaced, &IFFT_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    let complex_dim = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    }));
    let dimension = call(vec![input.clone(), complex_dim], vec![]);
    assert!(!inference::infer(
        inference::Transform::Fftshift,
        &dimension,
        &FFTSHIFT_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
    let matrix_size = numeric(NumericClass::Double, vec![Some(2), Some(2)]);
    let size = call(vec![input, matrix_size], vec![]);
    assert!(
        !inference::infer(inference::Transform::Fftn, &size, &FFTN_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}

#[test]
fn inverse_symmetry_flag_changes_complexity_but_not_shape_or_class() {
    let input = numeric(NumericClass::Single, vec![Some(2), Some(3)]);
    let flag = ValueFact::scalar(ValueKindFact::Character);
    let request = call(
        vec![input.clone(), flag],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Character("symmetric".into()),
        ],
    );
    let result = inference::infer(inference::Transform::Ifft2, &request, &IFFT2_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].shape, input.shape);
    assert_eq!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })
    );
}

#[test]
fn known_invalid_control_is_diagnosed_without_using_argument_shape_as_value() {
    let input = numeric(NumericClass::Double, vec![Some(2), Some(3)]);
    let control = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let bad_length = call(
        vec![input.clone(), control.clone()],
        vec![LiteralValue::Unknown, LiteralValue::Number(-1.0)],
    );
    let result = inference::infer(inference::Transform::Fft, &bad_length, &FFT_CATALOG_ENTRY);
    assert!(!result.diagnostics.is_empty());
    let bad_dimension = call(
        vec![input, control],
        vec![LiteralValue::Unknown, LiteralValue::Number(0.0)],
    );
    let result = inference::infer(
        inference::Transform::Fftshift,
        &bad_dimension,
        &FFTSHIFT_CATALOG_ENTRY,
    );
    assert!(!result.diagnostics.is_empty());
}

mod shift;
