use super::*;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

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

fn call(input: ValueFact, controls: Vec<LiteralValue>, outputs: usize) -> CallInference {
    let mut arguments = vec![input];
    arguments.extend(controls.iter().map(|literal| match literal {
        LiteralValue::String(_) | LiteralValue::Character(_) | LiteralValue::Keyword(_) => {
            ValueFact::scalar(ValueKindFact::String)
        }
        _ => ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
    }));
    let mut literals = vec![LiteralValue::Unknown];
    literals.extend(controls);
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &SUM_CATALOG_ENTRY,
    )
}

#[test]
fn descriptor_preserves_public_forms_defaults_and_errors() {
    let labels: Vec<&str> = SUM_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    for form in [
        "S = sum(A)",
        "S = sum(A, dim)",
        "S = sum(A, \"all\")",
        "S = sum(A, missingflag)",
        "S = sum(A, outtype)",
        "S = sum(A, dim, missingflag)",
        "S = sum(A, missingflag, dim)",
        "S = sum(A, vecdim)",
        "S = sum(A, options...)",
    ] {
        assert!(labels.contains(&form), "missing {form}");
    }
    assert!(crate::validate_builtin_preparation(&SUM_CATALOG_ENTRY).is_empty());
    let BuiltinOperandPreparation::Forms(forms) = SUM_CATALOG_ENTRY.placement.preparation else {
        panic!("sum uses its admitted data-only preparation form")
    };
    assert_eq!(forms.len(), 1);
    assert_eq!(forms[0].signatures, &[8]);
    assert_eq!(forms[0].data_arguments, &[0]);
    assert!(!labels
        .iter()
        .any(|label| label.contains("nanflag") || label.contains("like")));
    let missingflag = SUM_DESCRIPTOR
        .signatures
        .iter()
        .flat_map(|signature| signature.inputs)
        .find(|input| input.name == "missingflag")
        .expect("missing-data descriptor");
    assert_eq!(missingflag.default, Some("\"includemissing\""));
    assert!(missingflag.description.contains("\"omitmissing\""));
    assert_eq!(SUM_DESCRIPTOR.errors.len(), 3);
    for code in [
        SUM_ERROR_INVALID_ARGUMENT.code,
        SUM_ERROR_INVALID_INPUT.code,
        SUM_ERROR_INTERNAL.code,
    ] {
        assert!(SUM_DESCRIPTOR.errors.iter().any(|error| error.code == code));
    }
}

#[test]
fn known_floating_reduction_and_native_integer_classes_are_proven() {
    let shape = ShapeFact::from(vec![Some(3), Some(2)]);
    let reduced = call(numeric(NumericClass::Single, shape.clone()), vec![], 1);
    assert_eq!(
        reduced.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    assert_eq!(
        reduced.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    let native = call(
        numeric(NumericClass::UInt64, shape),
        vec![
            LiteralValue::String("all".into()),
            LiteralValue::String("native".into()),
        ],
        1,
    );
    assert_eq!(native.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(
        native.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );
}

#[test]
fn native_complex_single_preserves_class_after_actual_reduction() {
    let shape = ShapeFact::from(vec![Some(2), Some(2)]);
    let complex = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }),
        shape,
        StorageFact::Dense,
    );
    let native = call(complex, vec![LiteralValue::String("native".into())], 1);
    assert_eq!(
        native.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    assert_eq!(
        native.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
}

#[test]
fn far_axis_noop_retains_single_and_unknown_rank_does_not_guess() {
    let shape = ShapeFact::from(vec![Some(3), Some(2)]);
    let zero_axis = call(
        numeric(NumericClass::Single, shape.clone()),
        vec![LiteralValue::Number(0.0)],
        1,
    );
    assert!(zero_axis
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-REDUCTION-DIMENSION"));

    let far = call(
        numeric(NumericClass::Single, shape.clone()),
        vec![LiteralValue::Integer {
            text: usize::MAX.to_string(),
            class: NumericClass::UInt64,
        }],
        1,
    );
    assert_eq!(far.outputs[0].shape, shape);
    assert_eq!(
        far.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );

    let explicit_double = call(
        numeric(NumericClass::Single, shape.clone()),
        vec![
            LiteralValue::Number(5.0),
            LiteralValue::String("double".into()),
        ],
        1,
    );
    assert_eq!(explicit_double.outputs[0].shape, shape);
    assert_eq!(
        explicit_double.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );

    let ranked = ShapeFact::Ranked { rank: usize::MAX };
    let unknown = call(numeric(NumericClass::Single, ranked.clone()), vec![], 1);
    assert_eq!(unknown.outputs[0].shape, ranked);
    assert_eq!(
        unknown.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );

    let dynamic = call(
        numeric(NumericClass::Single, ShapeFact::Unknown),
        vec![LiteralValue::Number(3.0)],
        1,
    );
    assert_eq!(
        dynamic.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
}

#[test]
fn repeated_vector_axes_share_the_owning_idempotent_selection() {
    let shape = ShapeFact::from(vec![Some(3), Some(4)]);
    let inferred = call(
        numeric(NumericClass::Double, shape),
        vec![LiteralValue::Vector(vec![
            LiteralValue::Number(2.0),
            LiteralValue::Number(2.0),
        ])],
        1,
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
}

#[test]
fn full_axis_nd_single_and_complex_outputs_have_scalar_geometry() {
    let shape = ShapeFact::from(vec![Some(2), Some(2), Some(2)]);
    let single = call(
        numeric(NumericClass::Single, shape.clone()),
        vec![LiteralValue::String("all".into())],
        1,
    );
    assert_eq!(
        single.outputs[0].shape.known_dims(),
        Some(vec![Some(1), Some(1)])
    );
    assert_eq!(
        single.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );

    let mut complex = numeric(NumericClass::Single, shape);
    complex.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Complex,
    });
    let result = call(complex, vec![LiteralValue::String("all".into())], 1);
    assert_eq!(
        result.outputs[0].shape.known_dims(),
        Some(vec![Some(1), Some(1)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
}

#[test]
fn conflicting_options_and_resident_storage_do_not_become_proven() {
    let shape = ShapeFact::from(vec![Some(3), Some(2)]);
    for controls in [
        vec![
            LiteralValue::String("all".into()),
            LiteralValue::Number(2.0),
        ],
        vec![
            LiteralValue::String("native".into()),
            LiteralValue::String("double".into()),
        ],
        vec![LiteralValue::Unknown],
    ] {
        let result = call(numeric(NumericClass::Double, shape.clone()), controls, 1);
        assert!(matches!(
            result.outputs[0].certainty,
            runmat_types::CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
        ));
    }
    let mut resident = numeric(NumericClass::Double, shape);
    resident.residency = ResidencyFact::Device { provider: None };
    let result = call(resident, vec![], 2);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
    assert_eq!(result.outputs[0].storage, StorageFact::Unknown);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-TYPE-CALL-ARITY"));
}
