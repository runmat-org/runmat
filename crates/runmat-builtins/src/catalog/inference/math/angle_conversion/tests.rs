use crate::{
    builtin_catalog_entry_by_name, AngleConversionInferenceRule, BuiltinInferenceRule,
    MathInferenceRule,
};
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn degree_conversion_preserves_float_facts_and_promotes_extensions() {
    let entry = builtin_catalog_entry_by_name("deg2rad").expect("deg2rad entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::AngleConversion(
            AngleConversionInferenceRule::DegreesToRadians,
        ))
    );
    assert_eq!(entry.descriptor.signatures[0].label, "R = deg2rad(D)");
    assert!(entry
        .descriptor
        .errors
        .iter()
        .any(|error| error.code == "RM.DEG2RAD.TOO_MANY_OUTPUTS"));

    let single = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let inferred = crate::infer_catalog_call(entry, &request(vec![single]));
    let output = &inferred.outputs[0];
    assert_eq!(output.shape, ShapeFact::from(vec![Some(2), Some(3)]));
    assert_eq!(
        output.kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        })
    );

    let integer = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt16,
        domain: NumericDomain::Real,
    }));
    let inferred = crate::infer_catalog_call(entry, &request(vec![integer]));
    assert_eq!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
}

#[test]
fn degree_conversion_rejects_invalid_representation_and_excess_inputs() {
    let entry = builtin_catalog_entry_by_name("deg2rad").expect("deg2rad entry");
    let invalid = ValueFact::scalar(ValueKindFact::String);
    let inferred = crate::infer_catalog_call(entry, &request(vec![invalid]));
    assert!(!inferred.diagnostics.is_empty());

    let input = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let inferred = crate::infer_catalog_call(entry, &request(vec![input.clone(), input]));
    assert!(!inferred.diagnostics.is_empty());
}

#[test]
fn radian_conversion_uses_the_same_typed_family_contract() {
    let entry = builtin_catalog_entry_by_name("rad2deg").expect("rad2deg entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::AngleConversion(
            AngleConversionInferenceRule::RadiansToDegrees,
        ))
    );
    assert_eq!(entry.descriptor.signatures[0].label, "D = rad2deg(R)");

    let single = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(4), Some(1)]),
        StorageFact::Dense,
    );
    let inferred = crate::infer_catalog_call(entry, &request(vec![single]));
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(4), Some(1)])
    );
    assert_eq!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })
    );
}
