use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[test]
fn logarithm_and_root_singletons_match_their_floating_carriers() {
    for name in ["log", "log10", "log1p", "log2", "sqrt", "realsqrt"] {
        for class in [NumericClass::Single, NumericClass::Double] {
            for domain in [NumericDomain::Real, NumericDomain::Complex] {
                if name == "realsqrt" && domain == NumericDomain::Complex {
                    continue;
                }
                let input = ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact { class, domain }),
                    ShapeFact::from(vec![Some(1), Some(1)]),
                    StorageFact::Dense,
                );
                let inference = infer_catalog_call(
                    builtin_catalog_entry_by_name(name).unwrap(),
                    &CallRequest {
                        arguments: vec![input],
                        literals: LiteralContext::new(vec![LiteralValue::Number(4.0)]),
                        outputs: OutputSelection::new(RequestedOutputCount::One),
                    },
                );
                assert!(inference.diagnostics.is_empty(), "{name}");
                assert_eq!(
                    inference.outputs[0].storage,
                    if class == NumericClass::Single {
                        StorageFact::Dense
                    } else {
                        StorageFact::Scalar
                    },
                    "{name}"
                );
            }
        }
    }
}

#[test]
fn realsqrt_does_not_erase_sparse_or_unknown_sparse_representation() {
    for storage in [StorageFact::Sparse, StorageFact::Unknown] {
        let input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::Scalar,
            storage,
        );
        let inference = infer_catalog_call(
            builtin_catalog_entry_by_name("realsqrt").unwrap(),
            &CallRequest {
                arguments: vec![input],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        );
        assert_eq!(inference.outputs[0].storage, storage);
    }
}

#[test]
fn log2_single_dissection_keeps_dense_outputs() {
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("log2").unwrap(),
        &CallRequest {
            arguments: vec![ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Single,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::Scalar,
                StorageFact::Dense,
            )],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        },
    );
    assert_eq!(inference.outputs.len(), 2);
    assert!(inference
        .outputs
        .iter()
        .all(|output| output.storage == StorageFact::Dense));
}
