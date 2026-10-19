use crate::{
    builtin_catalog_entry_by_name, infer_catalog_call, BuiltinInferenceRule, MathInferenceRule,
};
use runmat_types::{
    standard, AliasFact, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    ViewFact,
};

use super::super::test_support::{numeric, object, request};

mod tabular;

#[test]
fn admitted_atan2_preserves_numeric_facts_without_inventing_placement() {
    let entry = builtin_catalog_entry_by_name("atan2").unwrap();
    for class in [NumericClass::Single, NumericClass::Double] {
        let result = infer_catalog_call(
            entry,
            &request(vec![
                numeric(class, NumericDomain::Real, vec![Some(2), Some(1)]),
                numeric(
                    NumericClass::Double,
                    NumericDomain::Real,
                    vec![Some(1), Some(3)],
                ),
            ]),
        );
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
        assert_eq!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real
            })
        );
    }
    let crate::BuiltinOperandPreparation::Forms(forms) = entry.placement.preparation else {
        panic!("canonical preparation")
    };
    assert_eq!(forms.len(), 1);
    assert_eq!(forms[0].data_arguments, &[0, 1]);
    assert_eq!(
        forms[0].operation,
        crate::BuiltinPreparationOperation::ElementwiseBinary
    );
}

#[test]
fn tracks_broadcast_class_residency_and_materialization() {
    let entry = builtin_catalog_entry_by_name("atan2").expect("atan2 catalog entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::Atan2)
    );
    let mut y = numeric(
        NumericClass::UInt64,
        NumericDomain::Real,
        vec![Some(2), Some(1)],
    );
    y.residency = ResidencyFact::Device {
        provider: Some("provider-a".into()),
    };
    let inferred = infer_catalog_call(
        entry,
        &request(vec![
            y,
            numeric(
                NumericClass::Double,
                NumericDomain::Real,
                vec![Some(1), Some(3)],
            ),
        ]),
    );
    assert!(inferred.diagnostics.is_empty());
    let output = &inferred.outputs[0];
    assert_eq!(
        output.kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(output.shape, ShapeFact::from(vec![Some(2), Some(3)]));
    assert!(matches!(output.residency, ResidencyFact::Device { .. }));
    assert_eq!(output.view, ViewFact::Materialized);
    assert_eq!(output.alias, AliasFact::Unique);

    let single = infer_catalog_call(
        entry,
        &request(vec![
            numeric(
                NumericClass::UInt16,
                NumericDomain::Real,
                vec![Some(2), Some(2)],
            ),
            numeric(
                NumericClass::Single,
                NumericDomain::Real,
                vec![Some(2), Some(2)],
            ),
        ]),
    );
    assert!(matches!(
        single.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real
        })
    ));
}

#[test]
fn rejects_complex_sparse_and_excess_outputs() {
    let entry = builtin_catalog_entry_by_name("atan2").expect("atan2 catalog entry");
    for (operand, code) in [
        (
            numeric(
                NumericClass::Double,
                NumericDomain::Complex,
                vec![Some(2), Some(2)],
            ),
            "RM-CATALOG-ATAN2-COMPLEX",
        ),
        (
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(2), Some(2)]),
                StorageFact::Sparse,
            ),
            "RM-CATALOG-ATAN2-SPARSE",
        ),
    ] {
        let inferred = infer_catalog_call(
            entry,
            &request(vec![
                operand,
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ]),
        );
        assert!(inferred
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == code));
    }
    let mut too_many = request(vec![
        numeric(
            NumericClass::Double,
            NumericDomain::Real,
            vec![Some(1), Some(1)],
        ),
        numeric(
            NumericClass::Double,
            NumericDomain::Real,
            vec![Some(1), Some(1)],
        ),
    ]);
    too_many.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    assert!(!infer_catalog_call(entry, &too_many).diagnostics.is_empty());
}
