use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact, ViewFact,
};

#[test]
fn unresolved_inverse_domain_retains_materialized_geometry_not_invented_numeric_kind() {
    for name in ["asin", "acos", "acosh", "atanh"] {
        let entry = builtin_catalog_entry_by_name(name).unwrap();
        for shape in [ShapeFact::Scalar, ShapeFact::from(vec![Some(2), Some(3)])] {
            for class in [NumericClass::Single, NumericClass::Double] {
                let input = ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class,
                        domain: NumericDomain::Real,
                    }),
                    shape.clone(),
                    StorageFact::Dense,
                );
                let inference = infer_catalog_call(
                    entry,
                    &CallRequest {
                        arguments: vec![input],
                        literals: Default::default(),
                        outputs: Default::default(),
                    },
                );
                assert!(inference.diagnostics.is_empty(), "{name}");
                let [output] = inference.outputs.as_slice() else {
                    panic!("one inverse result required")
                };
                assert_eq!(output.kind, ValueKindFact::Unknown, "{name}");
                assert_eq!(output.shape, shape, "{name}");
                assert_eq!(output.view, ViewFact::Materialized, "{name}");
                assert_eq!(
                    output.storage,
                    if shape.element_count() == Some(1) {
                        StorageFact::Scalar
                    } else {
                        StorageFact::Dense
                    },
                    "{name}"
                );
            }
        }
        let unknown = infer_catalog_call(
            entry,
            &CallRequest {
                arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
                literals: Default::default(),
                outputs: Default::default(),
            },
        );
        assert_eq!(unknown.outputs[0].storage, StorageFact::Unknown);
    }
}
