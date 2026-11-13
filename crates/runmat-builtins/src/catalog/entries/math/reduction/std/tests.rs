use super::*;
use runmat_types::{
    CallInference, CallRequest, CertaintyFact, LiteralContext, LiteralValue, NumericClass,
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
        _ => numeric(NumericClass::Double, ShapeFact::Scalar),
    }));
    let mut literals = vec![LiteralValue::Unknown];
    literals.extend(controls);
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &STD_CATALOG_ENTRY,
    )
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn descriptor_preserves_public_forms_defaults_errors_and_integer_policy() {
    let labels: Vec<_> = STD_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    for form in [
        "S = std(A)",
        "S = std(A, w)",
        "S = std(A, w, dim)",
        "S = std(A, w, vecdim)",
        "S = std(A, w, \"all\")",
        "S = std(A, nanflag)",
        "S = std(A, w, nanflag)",
        "S = std(A, w, dim, nanflag)",
        "S = std(A, w, nanflag, dim)",
        "S = std(A, w, [], nanflag)",
    ] {
        assert!(labels.contains(&form), "missing {form}");
    }
    let normalization = STD_DESCRIPTOR
        .signatures
        .iter()
        .flat_map(|signature| signature.inputs)
        .find(|input| input.name == "w")
        .expect("normalization descriptor");
    assert_eq!(normalization.default, Some("0"));
    let missing = STD_DESCRIPTOR
        .signatures
        .iter()
        .flat_map(|signature| signature.inputs)
        .find(|input| input.name == "nanflag")
        .expect("missing-data descriptor");
    assert_eq!(missing.default, Some("\"includemissing\""));
    assert_eq!(STD_DESCRIPTOR.errors.len(), 4);
    for code in [
        STD_ERROR_INVALID_ARGUMENT.code,
        STD_ERROR_INVALID_INPUT.code,
        STD_ERROR_COMPLEX_UNSUPPORTED.code,
        STD_ERROR_INTERNAL.code,
    ] {
        assert!(STD_DESCRIPTOR.errors.iter().any(|error| error.code == code));
    }
    assert_eq!(STD_INTEGER_CAPABILITIES.len(), 2);
    assert_eq!(STD_EXTENSIONS.len(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn proven_double_shape_and_invalid_controls_remain_conservative() {
    let shape = ShapeFact::from(vec![Some(3), Some(4)]);
    let second = call(
        numeric(NumericClass::Double, shape.clone()),
        vec![LiteralValue::Number(0.0), LiteralValue::Number(2.0)],
        1,
    );
    assert_eq!(
        second.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    assert_eq!(
        second.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );

    for controls in [
        vec![
            LiteralValue::String("all".into()),
            LiteralValue::Number(2.0),
        ],
        vec![
            LiteralValue::Number(0.0),
            LiteralValue::Number(2.0),
            LiteralValue::Number(2.0),
        ],
        vec![
            LiteralValue::Number(0.0),
            LiteralValue::Vector(vec![LiteralValue::Number(2.0), LiteralValue::Number(0.0)]),
        ],
    ] {
        let result = call(numeric(NumericClass::Double, shape.clone()), controls, 1);
        assert!(matches!(
            result.outputs[0].certainty,
            CertaintyFact::Dynamic(_)
        ));
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn repeated_axes_in_one_vector_reduce_the_axis_once() {
    let result = call(
        numeric(
            NumericClass::Double,
            ShapeFact::from(vec![Some(3), Some(4)]),
        ),
        vec![
            LiteralValue::Number(0.0),
            LiteralValue::Vector(vec![LiteralValue::Number(2.0), LiteralValue::Number(2.0)]),
        ],
        1,
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    assert_eq!(result.outputs[0].certainty, CertaintyFact::Proven);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn single_output_class_is_proven_without_claiming_residency() {
    let shape = ShapeFact::from(vec![Some(3), Some(2)]);
    let single = call(numeric(NumericClass::Single, shape.clone()), vec![], 1);
    assert_eq!(
        single.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    assert_eq!(single.outputs[0].certainty, CertaintyFact::Proven);
    assert_eq!(
        single.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    let singleton = call(numeric(NumericClass::Single, ShapeFact::Scalar), vec![], 1);
    assert_eq!(
        singleton.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(1)])
    );
    assert_eq!(singleton.outputs[0].storage, StorageFact::Dense);
    let all = call(
        numeric(
            NumericClass::Single,
            ShapeFact::from(vec![Some(3), Some(2)]),
        ),
        vec![LiteralValue::String("all".into())],
        1,
    );
    assert_eq!(all.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(
        all.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(all.outputs[0].storage, StorageFact::Dense);

    let mut resident = numeric(NumericClass::Single, shape);
    resident.residency = ResidencyFact::Device { provider: None };
    let result = call(resident, vec![], 2);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
    assert_eq!(result.outputs[0].storage, StorageFact::Unknown);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn normalization_literal_keeps_single_rounding_and_all_spelling() {
    let shape = ShapeFact::from(vec![Some(3), Some(4)]);
    let axis_without_flag = call(
        numeric(NumericClass::Double, shape.clone()),
        vec![LiteralValue::Number(2.0)],
        1,
    );
    assert_eq!(
        axis_without_flag.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    let single_flag = call(
        numeric(NumericClass::Double, shape.clone()),
        vec![
            LiteralValue::Real {
                text: "1.00000001".into(),
                class: NumericClass::Single,
            },
            LiteralValue::Number(2.0),
        ],
        1,
    );
    assert_eq!(
        single_flag.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    let double_flag = call(
        numeric(NumericClass::Double, shape.clone()),
        vec![
            LiteralValue::Real {
                text: "1.00000001".into(),
                class: NumericClass::Double,
            },
            LiteralValue::Number(2.0),
        ],
        1,
    );
    assert!(matches!(
        double_flag.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));
    let all = call(
        numeric(NumericClass::Double, shape),
        vec![LiteralValue::String(" all ".into())],
        1,
    );
    assert_eq!(all.outputs[0].shape, ShapeFact::Scalar);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn platform_dimension_and_duplicate_vector_follow_owner_admission() {
    let shape = ShapeFact::from(vec![Some(3), Some(4)]);
    let duplicate = call(
        numeric(NumericClass::Double, shape.clone()),
        vec![
            LiteralValue::Empty,
            LiteralValue::Vector(vec![LiteralValue::Number(2.0), LiteralValue::Number(2.0)]),
        ],
        1,
    );
    assert_eq!(
        duplicate.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    let far = call(
        numeric(NumericClass::Double, shape.clone()),
        vec![LiteralValue::Integer {
            text: u64::MAX.to_string(),
            class: NumericClass::UInt64,
        }],
        1,
    );
    if usize::BITS == 64 {
        assert_eq!(far.outputs[0].shape, shape);
    } else {
        assert!(matches!(
            far.outputs[0].certainty,
            CertaintyFact::Dynamic(_)
        ));
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn integer_and_complex_inputs_report_owning_diagnostics() {
    let integer = call(numeric(NumericClass::UInt64, ShapeFact::Scalar), vec![], 1);
    assert!(integer
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == STD_ERROR_INVALID_INPUT.code));
    let complex = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let result = call(complex, vec![], 1);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == STD_ERROR_COMPLEX_UNSUPPORTED.code));
}
