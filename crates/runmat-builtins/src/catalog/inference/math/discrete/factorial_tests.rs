use super::infer;
use crate::builtin_catalog_entry_by_name;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn factorial_preserves_numeric_class_and_shape() {
    let entry = builtin_catalog_entry_by_name("factorial").expect("factorial catalog entry");
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let inferred = infer(&request(vec![input], Vec::new()), entry);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
}

#[test]
fn factorial_like_changes_residency_without_changing_class() {
    let entry = builtin_catalog_entry_by_name("factorial").expect("factorial catalog entry");
    let input = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt16,
        domain: NumericDomain::Real,
    }));
    let mut prototype = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    prototype.residency = runmat_types::ResidencyFact::Device {
        provider: Some("test".into()),
    };
    let keyword = ValueFact::scalar(ValueKindFact::String);
    let inferred = infer(
        &request(
            vec![input, keyword, prototype],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("like".into()),
                LiteralValue::Unknown,
            ],
        ),
        entry,
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt16,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(
        inferred.outputs[0].residency,
        runmat_types::ResidencyFact::Device {
            provider: Some("test".into())
        }
    );
}
