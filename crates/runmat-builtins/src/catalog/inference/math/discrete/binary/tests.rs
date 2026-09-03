use crate::builtin_catalog_entry_by_name;
use crate::BinaryNumberTheoryRule;
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

use super::infer;

fn numeric(class: NumericClass, shape: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape.iter().copied().map(Some).collect::<Vec<_>>()),
        if shape.iter().product::<usize>() == 1 {
            StorageFact::Scalar
        } else {
            StorageFact::Dense
        },
    )
}

fn request(arguments: Vec<ValueFact>, outputs: RequestedOutputCount) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(Vec::new()),
        outputs: OutputSelection::new(outputs),
    }
}

#[test]
fn binary_rules_preserve_class_and_apply_only_scalar_expansion() {
    let entry = builtin_catalog_entry_by_name("lcm").expect("lcm entry");
    let inferred = infer(
        BinaryNumberTheoryRule::Lcm,
        &request(
            vec![
                numeric(NumericClass::Int32, &[2, 3]),
                numeric(NumericClass::Double, &[1, 1]),
            ],
            RequestedOutputCount::One,
        ),
        entry,
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(
        inferred.outputs[0].numeric().unwrap().class,
        NumericClass::Int32
    );

    let mismatch = infer(
        BinaryNumberTheoryRule::Lcm,
        &request(
            vec![
                numeric(NumericClass::Double, &[2, 1]),
                numeric(NumericClass::Double, &[1, 2]),
            ],
            RequestedOutputCount::One,
        ),
        entry,
    );
    assert!(!mismatch.diagnostics.is_empty());
}

#[test]
fn gcd_models_requested_outputs_and_rejects_unsigned_coefficients() {
    let entry = builtin_catalog_entry_by_name("gcd").expect("gcd entry");
    let inferred = infer(
        BinaryNumberTheoryRule::Gcd,
        &request(
            vec![
                numeric(NumericClass::Int16, &[2, 2]),
                numeric(NumericClass::Int16, &[2, 2]),
            ],
            RequestedOutputCount::Exactly(3),
        ),
        entry,
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs.len(), 3);
    assert!(inferred
        .outputs
        .iter()
        .all(|output| output.numeric().unwrap().class == NumericClass::Int16));

    let unsigned = infer(
        BinaryNumberTheoryRule::Gcd,
        &request(
            vec![
                numeric(NumericClass::UInt16, &[1, 1]),
                numeric(NumericClass::UInt16, &[1, 1]),
            ],
            RequestedOutputCount::Exactly(3),
        ),
        entry,
    );
    assert!(!unsigned.diagnostics.is_empty());
}
