use super::*;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

#[test]
fn mean_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = MEAN_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert_eq!(
        labels,
        vec![
            "M = mean(A)",
            "M = mean(A, dim)",
            "M = mean(A, \"all\")",
            "M = mean(A, missingflag)",
            "M = mean(A, outtype)",
            "M = mean(A, dim, missingflag)",
            "M = mean(A, missingflag, dim)",
            "M = mean(A, vecdim)",
            "M = mean(___, Weights=W)",
        ]
    );
    assert_eq!(
        MEAN_DESCRIPTOR.signatures[4].inputs[1].default,
        Some("\"default\"")
    );
    assert_eq!(
        MEAN_DESCRIPTOR.signatures[3].inputs[1].default,
        Some("\"includemissing\"")
    );
}

#[test]
fn mean_descriptor_errors_have_stable_codes() {
    for expected in [
        MEAN_ERROR_INVALID_ARGUMENT,
        MEAN_ERROR_INVALID_INPUT,
        MEAN_ERROR_INTERNAL,
    ] {
        assert!(MEAN_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == expected.code));
    }
}

fn numeric(class: NumericClass) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(3), Some(2)]),
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
        &MEAN_CATALOG_ENTRY,
    )
}

#[test]
fn mean_proves_default_dim_all_and_native_integer() {
    let default = call(numeric(NumericClass::Double), vec![], 1);
    assert_eq!(
        default.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    let along_rows = call(
        numeric(NumericClass::Single),
        vec![LiteralValue::Number(2.0)],
        1,
    );
    assert_eq!(
        along_rows.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    assert_eq!(
        along_rows.outputs[0].numeric().expect("known class").class,
        NumericClass::Single
    );
    let native = call(
        numeric(NumericClass::UInt64),
        vec![
            LiteralValue::String("all".into()),
            LiteralValue::String("native".into()),
        ],
        1,
    );
    assert_eq!(native.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(
        native.outputs[0]
            .numeric()
            .expect("known native class")
            .class,
        NumericClass::UInt64
    );
}

#[test]
fn explicit_double_overrides_single_and_unknown_controls_remain_dynamic() {
    let explicit_double = call(
        numeric(NumericClass::Single),
        vec![LiteralValue::String("double".into())],
        1,
    );
    assert_eq!(
        explicit_double.outputs[0]
            .numeric()
            .expect("known class")
            .class,
        NumericClass::Double
    );
    let weighted = call(
        numeric(NumericClass::Double),
        vec![
            LiteralValue::Number(1.0),
            LiteralValue::String("Weights".into()),
            LiteralValue::Vector(vec![
                LiteralValue::Number(1.0),
                LiteralValue::Number(3.0),
                LiteralValue::Number(2.0),
            ]),
        ],
        1,
    );
    assert_eq!(
        weighted.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    assert!(weighted.diagnostics.is_empty());
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
        vec![
            LiteralValue::String("Weights".into()),
            LiteralValue::Unknown,
        ],
    ] {
        let inferred = call(numeric(NumericClass::Double), controls, 1);
        assert!(matches!(
            inferred.outputs[0].certainty,
            runmat_types::CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
        ));
    }
}

#[test]
fn output_arity_and_residency_are_not_overclaimed() {
    let mut resident = numeric(NumericClass::Single);
    resident.residency = ResidencyFact::Device { provider: None };
    let inferred = call(resident, vec![], 2);
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Unknown);
    assert_eq!(inferred.outputs[0].storage, StorageFact::Unknown);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}

#[test]
fn abstract_rank_and_far_axis_remain_bounded_and_sound() {
    let mut ranked = numeric(NumericClass::Double);
    ranked.shape = ShapeFact::Ranked { rank: usize::MAX };
    assert_eq!(
        call(ranked.clone(), vec![], 1).outputs[0].shape,
        ranked.shape
    );
    let shaped = numeric(NumericClass::Double);
    let far_axis = call(
        shaped.clone(),
        vec![LiteralValue::Integer {
            text: usize::MAX.to_string(),
            class: NumericClass::UInt64,
        }],
        1,
    );
    assert_eq!(far_axis.outputs[0].shape, shaped.shape);
}

#[test]
fn known_complex_integer_is_rejected_without_a_proven_output() {
    let mut input = numeric(NumericClass::Int64);
    input.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Int64,
        domain: NumericDomain::Complex,
    });
    let inferred = call(input, vec![], 1);
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM.MEAN.INVALID_INPUT"));
}
