use super::*;

#[test]
fn sparse_lu_native_four_facts_have_rectangular_factors_and_real_scaling() {
    for class in [NumericClass::Single, NumericClass::Double] {
        for domain in [NumericDomain::Real, NumericDomain::Complex] {
            for vector in [false, true] {
                let mut call = factor_request_with_options(
                    3,
                    2,
                    class,
                    5,
                    if vector { &["vector"] } else { &[] },
                );
                call.arguments[0].storage = StorageFact::Sparse;
                call.arguments[0].kind = ValueKindFact::Numeric(NumericFact { class, domain });
                let inferred = infer_catalog_call(&LU_CATALOG_ENTRY, &call);
                assert!(
                    inferred.diagnostics.is_empty(),
                    "{:?}",
                    inferred.diagnostics
                );
                assert_eq!(
                    inferred.outputs[0].shape,
                    ShapeFact::from(vec![Some(3), Some(2)])
                );
                assert_eq!(
                    inferred.outputs[1].shape,
                    ShapeFact::from(vec![Some(2), Some(2)])
                );
                for index in [0, 1] {
                    assert_eq!(inferred.outputs[index].storage, StorageFact::Sparse);
                    assert_eq!(inferred.outputs[index].kind, call.arguments[0].kind);
                }
                for (index, size) in [(2, 3), (3, 2)] {
                    assert_eq!(
                        inferred.outputs[index].shape,
                        ShapeFact::from(vec![Some(size), Some(if vector { 1 } else { size })])
                    );
                    assert_eq!(
                        inferred.outputs[index].storage,
                        if vector {
                            StorageFact::Dense
                        } else {
                            StorageFact::Sparse
                        }
                    );
                    assert_eq!(
                        inferred.outputs[index].kind,
                        ValueKindFact::Numeric(NumericFact {
                            class: NumericClass::Double,
                            domain: NumericDomain::Real
                        })
                    );
                }
                assert_eq!(
                    inferred.outputs[4].shape,
                    ShapeFact::from(vec![Some(3), Some(3)])
                );
                assert_eq!(
                    inferred.outputs[4].kind,
                    ValueKindFact::Numeric(NumericFact {
                        class,
                        domain: NumericDomain::Real
                    })
                );
            }
        }
    }
    let dense = factor_request_with_options(3, 2, NumericClass::Double, 4, &[]);
    assert!(infer_catalog_call(&LU_CATALOG_ENTRY, &dense)
        .diagnostics
        .iter()
        .any(|d| d.code == LU_ERROR_INVALID_ARGUMENT.code));
}

#[test]
fn sparse_qr_facts_preserve_native_complex_and_transformed_empty_rhs() {
    for class in [NumericClass::Single, NumericClass::Double] {
        for domain in [NumericDomain::Real, NumericDomain::Complex] {
            let mut call = factor_request_with_options(3, 2, class, 3, &["econ", "vector"]);
            call.arguments[0].storage = StorageFact::Sparse;
            call.arguments[0].kind = ValueKindFact::Numeric(NumericFact { class, domain });
            let inferred = infer_catalog_call(&QR_CATALOG_ENTRY, &call);
            assert!(
                inferred.diagnostics.is_empty(),
                "{:?}",
                inferred.diagnostics
            );
            assert_eq!(
                inferred.outputs[0].shape,
                ShapeFact::from(vec![Some(3), Some(2)])
            );
            assert_eq!(
                inferred.outputs[1].shape,
                ShapeFact::from(vec![Some(2), Some(2)])
            );
            assert_eq!(inferred.outputs[1].kind, call.arguments[0].kind);
            assert_eq!(inferred.outputs[0].kind, call.arguments[0].kind);
            assert_eq!(inferred.outputs[1].storage, StorageFact::Sparse);
            assert_eq!(inferred.outputs[2].storage, StorageFact::Dense);
            let mut b = matrix(3, 0);
            b.kind = ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Complex,
            });
            call.arguments.insert(1, b);
            call.literals.literal_args.insert(1, LiteralValue::Unknown);
            let inferred = infer_catalog_call(&QR_CATALOG_ENTRY, &call);
            assert!(
                inferred.diagnostics.is_empty(),
                "{:?}",
                inferred.diagnostics
            );
            assert_eq!(
                inferred.outputs[0].shape,
                ShapeFact::from(vec![Some(2), Some(0)])
            );
            assert_eq!(
                inferred.outputs[0].kind,
                ValueKindFact::Numeric(NumericFact {
                    class,
                    domain: NumericDomain::Complex
                })
            );
            call.arguments.truncate(2);
            call.literals.literal_args.truncate(2);
            let inferred = infer_catalog_call(&QR_CATALOG_ENTRY, &call);
            assert_eq!(inferred.outputs[2].storage, StorageFact::Sparse);
        }
    }
}
