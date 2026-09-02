use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    CallRequest, DimensionFact, DynamicReason, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

fn numeric(class: NumericClass, shape: ShapeFact) -> ValueFact {
    let mut fact = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    }));
    fact.shape = shape;
    fact.storage = StorageFact::Dense;
    fact
}

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn gamma_random_tracks_class_shape_and_residency() {
    let entry = builtin_catalog_entry_by_name("gamrnd").expect("gamrnd catalog entry");
    let mut shape = numeric(
        NumericClass::Single,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(1), DimensionFact::Known(3)],
        },
    );
    shape.residency = ResidencyFact::Device {
        provider: Some("test-provider".into()),
    };
    let inferred = infer_catalog_call(
        entry,
        &request(
            vec![shape, numeric(NumericClass::Double, ShapeFact::Scalar)],
            Vec::new(),
        ),
    );
    assert!(inferred.diagnostics.is_empty());
    assert!(matches!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real
        })
    ));
    assert_eq!(
        inferred.outputs[0].shape.known_dims(),
        Some(vec![Some(1), Some(3)])
    );
    assert!(matches!(
        inferred.outputs[0].residency,
        ResidencyFact::Device { .. }
    ));
}

#[test]
fn random_parameters_use_scalar_expansion_not_implicit_expansion() {
    let row = ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(1), DimensionFact::Known(3)],
    };
    let column = ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(3), DimensionFact::Known(1)],
    };
    for name in ["gamrnd", "binornd"] {
        let entry = builtin_catalog_entry_by_name(name).expect("catalog entry");
        let inferred = infer_catalog_call(
            entry,
            &request(
                vec![
                    numeric(NumericClass::Double, row.clone()),
                    numeric(NumericClass::Double, column.clone()),
                ],
                Vec::new(),
            ),
        );
        assert!(inferred
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-RANDOM-SHAPE"));
        assert!(matches!(inferred.outputs[0].shape, ShapeFact::Unknown));
    }
}

#[test]
fn binomial_random_validates_literals_and_explicit_parameter_shape() {
    let entry = builtin_catalog_entry_by_name("binornd").expect("binornd catalog entry");
    let parameter = numeric(
        NumericClass::Single,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(1), DimensionFact::Known(3)],
        },
    );
    let inferred = infer_catalog_call(
        entry,
        &request(
            vec![
                parameter,
                numeric(NumericClass::Double, ShapeFact::Scalar),
                numeric(NumericClass::Double, ShapeFact::Scalar),
            ],
            vec![
                LiteralValue::Number(0.5),
                LiteralValue::Number(1.5),
                LiteralValue::Vector(vec![LiteralValue::Number(2.0), LiteralValue::Number(4.0)]),
            ],
        ),
    );
    assert_eq!(
        inferred.outputs[0].shape.known_dims(),
        Some(vec![Some(2), Some(4)])
    );
    assert!(matches!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real
        })
    ));
    for code in [
        "RM-CATALOG-BINORND-TRIALS",
        "RM-CATALOG-BINORND-PROBABILITY",
        "RM-CATALOG-RANDOM-SIZE-MISMATCH",
    ] {
        assert!(inferred
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == code));
    }
}

#[test]
fn unknown_parameter_class_does_not_invent_double() {
    for name in ["gamrnd", "binornd"] {
        let entry = builtin_catalog_entry_by_name(name).expect("catalog entry");
        let inferred = infer_catalog_call(
            entry,
            &request(
                vec![
                    ValueFact::unknown(DynamicReason::RuntimeValue),
                    numeric(NumericClass::Double, ShapeFact::Scalar),
                ],
                Vec::new(),
            ),
        );
        assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Unknown));
    }
}
