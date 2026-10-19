use super::*;
use runmat_types::*;

fn numeric(class: NumericClass, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn default_and_option_dependent_bins_preserve_double_column_facts() {
    let image = numeric(NumericClass::UInt8, ShapeFact::from(vec![Some(2), Some(3)]));
    let default = inference::infer(
        &request(vec![image.clone()], vec![LiteralValue::Unknown], 2),
        &IMHIST_CATALOG_ENTRY,
    );
    assert!(default.diagnostics.is_empty());
    assert_eq!(default.outputs.len(), 2);
    for output in &default.outputs {
        assert_eq!(output.shape, ShapeFact::from(vec![Some(256), Some(1)]));
        assert_eq!(output.numeric().unwrap().class, NumericClass::Double);
        assert_eq!(output.residency, ResidencyFact::Host);
    }

    let three = inference::infer(
        &request(
            vec![image, numeric(NumericClass::Double, ShapeFact::Scalar)],
            vec![LiteralValue::Unknown, LiteralValue::Number(3.0)],
            1,
        ),
        &IMHIST_CATALOG_ENTRY,
    );
    assert_eq!(
        three.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );

    let logical = inference::infer(
        &request(
            vec![ValueFact::proven(
                ValueKindFact::Logical,
                ShapeFact::Scalar,
                StorageFact::Scalar,
            )],
            vec![LiteralValue::Bool(true)],
            1,
        ),
        &IMHIST_CATALOG_ENTRY,
    );
    assert_eq!(
        logical.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
}

#[test]
fn unknown_image_and_indexed_map_keep_only_sound_shape_and_residency() {
    let unknown = inference::infer(
        &request(
            vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            vec![LiteralValue::Unknown],
            1,
        ),
        &IMHIST_CATALOG_ENTRY,
    );
    assert!(unknown.diagnostics.is_empty());
    assert_eq!(
        unknown.outputs[0].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
    assert_eq!(
        unknown.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(unknown.outputs[0].residency, ResidencyFact::Unknown);

    let map = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(4), Some(3)]),
    );
    let indexed = inference::infer(
        &request(
            vec![
                numeric(
                    NumericClass::UInt16,
                    ShapeFact::from(vec![Some(2), Some(2)]),
                ),
                map,
            ],
            vec![LiteralValue::Unknown; 2],
            2,
        ),
        &IMHIST_CATALOG_ENTRY,
    );
    assert!(indexed.diagnostics.is_empty());
    assert_eq!(
        indexed.outputs[0].shape,
        ShapeFact::from(vec![Some(4), Some(1)])
    );

    let mut resident = numeric(NumericClass::UInt8, ShapeFact::Scalar);
    resident.residency = ResidencyFact::Device { provider: None };
    let resident_result = inference::infer(
        &request(vec![resident], vec![LiteralValue::Unknown], 1),
        &IMHIST_CATALOG_ENTRY,
    );
    assert_eq!(resident_result.outputs[0].residency, ResidencyFact::Unknown);
}

#[test]
fn invalid_classes_control_and_output_count_do_not_claim_results() {
    for (image, control) in [
        (numeric(NumericClass::UInt64, ShapeFact::Scalar), None),
        (
            numeric(NumericClass::Double, ShapeFact::Scalar),
            Some(numeric(
                NumericClass::Double,
                ShapeFact::from(vec![Some(2), Some(2)]),
            )),
        ),
    ] {
        let mut args = vec![image];
        if let Some(control) = control {
            args.push(control);
        }
        let facts = inference::infer(
            &request(args.clone(), vec![LiteralValue::Unknown; args.len()], 1),
            &IMHIST_CATALOG_ENTRY,
        );
        assert!(!facts.diagnostics.is_empty());
        assert_eq!(facts.outputs[0].kind, ValueKindFact::Unknown);
    }
    let image = numeric(NumericClass::Single, ShapeFact::Scalar);
    let bad_bins = inference::infer(
        &request(
            vec![
                image.clone(),
                numeric(NumericClass::Double, ShapeFact::Scalar),
            ],
            vec![LiteralValue::Unknown, LiteralValue::Number(0.0)],
            1,
        ),
        &IMHIST_CATALOG_ENTRY,
    );
    assert!(!bad_bins.diagnostics.is_empty());
    let excess = inference::infer(
        &request(vec![image.clone()], vec![LiteralValue::Unknown], 3),
        &IMHIST_CATALOG_ENTRY,
    );
    assert!(!excess.diagnostics.is_empty());
    assert!(excess
        .outputs
        .iter()
        .all(|output| output.kind == ValueKindFact::Unknown));
    let statement = inference::infer(
        &request(vec![image], vec![LiteralValue::Unknown], 0),
        &IMHIST_CATALOG_ENTRY,
    );
    assert!(statement.outputs.is_empty());
    assert!(IMHIST_DESCRIPTOR
        .signatures
        .iter()
        .any(|signature| signature.label == "imhist(I)" && signature.outputs.is_empty()));
}

#[test]
fn partial_colormap_shape_is_not_rejected_as_a_bad_control() {
    let image = numeric(NumericClass::Double, ShapeFact::Scalar);
    for shape in [
        ShapeFact::from(vec![Some(2), None]),
        ShapeFact::from(vec![None, None]),
        ShapeFact::Ranked { rank: 2 },
    ] {
        let inferred = inference::infer(
            &request(
                vec![image.clone(), numeric(NumericClass::Double, shape)],
                vec![LiteralValue::Unknown; 2],
                1,
            ),
            &IMHIST_CATALOG_ENTRY,
        );
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![None, Some(1)])
        );
    }
}

#[test]
fn catalog_bin_rounding_matches_runtime_tolerance() {
    let image = numeric(NumericClass::Double, ShapeFact::Scalar);
    let control = numeric(NumericClass::Double, ShapeFact::Scalar);
    for (literal, accepted) in [(3.0 + 5.0e-10, true), (3.0 + 2.0e-9, false)] {
        let inferred = inference::infer(
            &request(
                vec![image.clone(), control.clone()],
                vec![LiteralValue::Unknown, LiteralValue::Number(literal)],
                1,
            ),
            &IMHIST_CATALOG_ENTRY,
        );
        if accepted {
            assert!(inferred.diagnostics.is_empty());
            assert_eq!(
                inferred.outputs[0].shape,
                ShapeFact::from(vec![Some(3), Some(1)])
            );
        } else {
            assert!(inferred
                .diagnostics
                .iter()
                .any(|d| d.code == "RM-CATALOG-IMHIST-BIN-VALUE"));
        }
    }
}
