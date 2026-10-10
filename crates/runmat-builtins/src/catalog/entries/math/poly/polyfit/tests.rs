use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn numeric(class: NumericClass, domain: NumericDomain, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        shape,
        StorageFact::Dense,
    )
}

fn request(x: ValueFact, y: ValueFact, degree: ValueFact, outputs: usize) -> CallRequest {
    CallRequest {
        arguments: vec![x, y, degree],
        literals: LiteralContext::new(vec![LiteralValue::Unknown; 3]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn named_outputs_have_host_double_contract_and_structure_fields() {
    let samples = ShapeFact::from(vec![Some(5), Some(1)]);
    let mut call = request(
        numeric(NumericClass::Single, NumericDomain::Real, samples.clone()),
        numeric(NumericClass::Double, NumericDomain::Real, samples),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        4,
    );
    call.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::Number(2.0),
    ]);
    let result = crate::infer_catalog_call(&POLYFIT_CATALOG_ENTRY, &call);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(3)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    let ValueKindFact::Struct(stats) = &result.outputs[1].kind else {
        panic!("S should be a struct")
    };
    assert_eq!(
        stats.fields["R"].shape,
        ShapeFact::from(vec![Some(3), Some(3)])
    );
    assert_eq!(stats.fields["df"].shape, ShapeFact::Scalar);
    assert_eq!(
        result.outputs[2].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    assert_eq!(result.outputs[3].shape, ShapeFact::Scalar);
}

#[test]
fn unknown_degree_does_not_imply_coefficient_count() {
    let sample = numeric(
        NumericClass::Double,
        NumericDomain::Real,
        ShapeFact::from(vec![Some(4), Some(1)]),
    );
    let result = crate::infer_catalog_call(
        &POLYFIT_CATALOG_ENTRY,
        &request(
            sample.clone(),
            sample,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            1,
        ),
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn wide_typed_degree_literal_is_not_rounded_through_double() {
    let sample = numeric(
        NumericClass::Double,
        NumericDomain::Real,
        ShapeFact::from(vec![Some(4), Some(1)]),
    );
    let mut call = request(
        sample.clone(),
        sample,
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        })),
        1,
    );
    call.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::Integer {
            text: "9007199254740993".into(),
            class: NumericClass::UInt64,
        },
    ]);
    let result = crate::infer_catalog_call(&POLYFIT_CATALOG_ENTRY, &call);
    let width = usize::try_from(9_007_199_254_740_993_u64)
        .ok()
        .and_then(|degree| degree.checked_add(1));
    match width {
        Some(exact_width) => {
            assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
            assert_eq!(
                result.outputs[0].shape,
                ShapeFact::from(vec![Some(1), Some(exact_width)])
            );
        }
        None => {
            assert!(result
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "RM-CATALOG-POLYFIT-DEGREE"));
            assert_eq!(
                result.outputs[0].shape,
                ShapeFact::from(vec![Some(1), None])
            );
        }
    }
}

#[test]
fn known_invalid_shape_length_complex_x_and_weight_class_are_diagnosed() {
    let matrix = numeric(
        NumericClass::Double,
        NumericDomain::Real,
        ShapeFact::from(vec![Some(2), Some(3)]),
    );
    let complex_x = numeric(
        NumericClass::Double,
        NumericDomain::Complex,
        ShapeFact::from(vec![Some(1), Some(4)]),
    );
    let y = numeric(
        NumericClass::Double,
        NumericDomain::Real,
        ShapeFact::from(vec![Some(1), Some(5)]),
    );
    let degree = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut call = request(matrix, y.clone(), degree.clone(), 1);
    let result = crate::infer_catalog_call(&POLYFIT_CATALOG_ENTRY, &call);
    assert!(result.diagnostics.len() >= 2);
    call.arguments[0] = complex_x;
    call.arguments
        .push(ValueFact::scalar(ValueKindFact::String));
    let result = crate::infer_catalog_call(&POLYFIT_CATALOG_ENTRY, &call);
    assert!(result.diagnostics.len() >= 2);
}
