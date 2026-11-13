use super::*;
use runmat_execution::value::{
    DurableDecompositionPayload, DurableNumericFactor, DurableStructuralDecomposition, InlineValue,
    ValuePayload,
};
use runmat_value::{
    DenseOrthogonalFactorData, ExactHostSparseClass, FactorPermutation, HostIndexBuffer,
    HostNumericBuffer, NativeDenseFactorMatrix, NumericDType, NumericFactorCandidate,
    NumericFactorData, NumericFactorKind, NumericFactorValidation, NumericStorage,
    OrthogonalFactorData, SparseTensor, Tensor, Value,
};
use std::num::NonZeroUsize;

fn identity() -> FactorPermutation {
    FactorPermutation {
        forward: HostIndexBuffer::new(vec![0, 1]),
        inverse: HostIndexBuffer::new(vec![0, 1]),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn empty_factor_data_roundtrips_without_virtual_axis_buffers() {
    for kind in [NumericFactorKind::Qr, NumericFactorKind::Cod] {
        let data = sealed(NumericFactorCandidate::Empty {
            kind,
            rows: 0,
            columns: 7,
            class: ExactHostSparseClass::Complex(NumericDType::F32),
        });
        assert_eq!(data.dimensions(), (0, 7));
        assert_roundtrip(data, kind);
    }
    for kind in [NumericFactorKind::Lu, NumericFactorKind::Cholesky] {
        assert_roundtrip(
            sealed(NumericFactorCandidate::Empty {
                kind,
                rows: 0,
                columns: 0,
                class: ExactHostSparseClass::Numeric(NumericDType::F64),
            }),
            kind,
        );
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn empty_factor_wire_rejects_nonempty_geometry_and_nonfloating_class() {
    use runmat_execution::value::ElementType;
    let source = sealed(NumericFactorCandidate::Empty {
        kind: NumericFactorKind::Qr,
        rows: 0,
        columns: 7,
        class: ExactHostSparseClass::Numeric(NumericDType::F32),
    });
    for defect in 0..2 {
        let mut payload = payload_with_factor(&source);
        let ValuePayload::Inline(encoded) = &mut payload else {
            unreachable!()
        };
        let InlineValue::DurablePlainObject(encoded) = encoded.as_mut() else {
            unreachable!()
        };
        let Some(DurableDecompositionPayload::Data(data)) = &mut encoded.decomposition else {
            unreachable!()
        };
        let DurableNumericFactor::Empty { rows, class, .. } = data.as_mut() else {
            unreachable!()
        };
        if defect == 0 {
            *rows = 1;
        } else {
            *class = ElementType::U8;
        }
        assert!(
            validate_durable_payload(&payload).is_err(),
            "wire defect {defect}"
        );
        assert!(
            decode_durable_value(&payload).is_err(),
            "native import defect {defect}"
        );
    }
}

fn identity_one() -> FactorPermutation {
    FactorPermutation {
        forward: HostIndexBuffer::new(vec![0]),
        inverse: HostIndexBuffer::new(vec![0]),
    }
}

fn block(values: [f64; 2]) -> SparseTensor {
    SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], values.to_vec()).unwrap()
}

fn dense(values: Vec<f64>, rows: usize, columns: usize) -> NativeDenseFactorMatrix {
    NativeDenseFactorMatrix::Real(Tensor::new(values, vec![rows, columns]).unwrap())
}

fn dense_qr(rank: usize) -> DenseOrthogonalFactorData {
    DenseOrthogonalFactorData {
        packed: dense(vec![2.0, 1.0, 3.0, 4.0], 2, 2),
        coefficients: dense(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
        permutation: identity(),
        rank,
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn durable_packed_dense_factors_roundtrip_without_csc_conversion() {
    assert_roundtrip(
        sealed(NumericFactorCandidate::DenseLu {
            packed: dense(vec![2.0, 1.0, 3.0, 4.0], 2, 2),
            rows: identity(),
        }),
        NumericFactorKind::Lu,
    );
    assert_roundtrip(
        sealed(NumericFactorCandidate::DenseCholesky {
            lower: dense(vec![2.0, 3.0, 0.0, 4.0], 2, 2),
        }),
        NumericFactorKind::Cholesky,
    );
    assert_roundtrip(
        sealed(NumericFactorCandidate::DenseQr(dense_qr(2))),
        NumericFactorKind::Qr,
    );
    assert_roundtrip(
        sealed(NumericFactorCandidate::DenseCod {
            first: dense_qr(1),
            completion: Some(DenseOrthogonalFactorData {
                packed: dense(vec![2.0, 1.0], 2, 1),
                coefficients: dense(vec![1.0], 1, 1),
                permutation: identity_one(),
                rank: 1,
            }),
        }),
        NumericFactorKind::Cod,
    );
}

fn sealed(candidate: NumericFactorCandidate) -> NumericFactorData {
    let mut validation = NumericFactorValidation::new(candidate).unwrap();
    while !validation.advance(NonZeroUsize::new(1).unwrap()).unwrap() {}
    validation.finish().unwrap()
}

fn qr_first() -> OrthogonalFactorData {
    OrthogonalFactorData {
        r: SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 0], vec![1.0, 2.0]).unwrap(),
        reflectors: SparseTensor::new(2, 2, vec![0, 1, 1], vec![0], vec![1.0]).unwrap(),
        permutation: identity(),
        rank: 1,
    }
}

fn cod_completion() -> OrthogonalFactorData {
    OrthogonalFactorData {
        r: SparseTensor::new(2, 1, vec![0, 1], vec![0], vec![1.0]).unwrap(),
        reflectors: SparseTensor::new(2, 1, vec![0, 1], vec![0], vec![1.0]).unwrap(),
        permutation: identity_one(),
        rank: 1,
    }
}

fn selected_object(data: &NumericFactorData, context: &crate::context::RuntimeContext) -> Value {
    crate::builtins::wasm_registry::register_all();
    let (rows, columns) = data.dimensions();
    assert!(
        rows <= 7 && columns <= 7,
        "fixture must stay physically bounded"
    );
    let sparse = matches!(
        data.parts(),
        NumericFactorCandidate::Lu { .. }
            | NumericFactorCandidate::Cholesky { .. }
            | NumericFactorCandidate::Qr(_)
            | NumericFactorCandidate::Cod { .. }
    );
    let matrix = match (sparse, data.storage_class()) {
        (true, ExactHostSparseClass::Numeric(NumericDType::F64)) => Value::SparseTensor(
            SparseTensor::new(
                rows,
                columns,
                vec![0, 2, 4],
                vec![0, 1, 0, 1],
                vec![4.0, 1.0, 1.0, 3.0],
            )
            .unwrap(),
        ),
        (true, ExactHostSparseClass::Numeric(NumericDType::F32)) => Value::SparseTensor(
            SparseTensor::new_f32(
                rows,
                columns,
                vec![0, 2, 4],
                vec![0, 1, 0, 1],
                vec![4.0, 1.0, 1.0, 3.0],
            )
            .unwrap(),
        ),
        (true, ExactHostSparseClass::Complex(NumericDType::F64)) => Value::SparseTensor(
            SparseTensor::new_complex(
                rows,
                columns,
                vec![0, 2, 4],
                vec![0, 1, 0, 1],
                vec![(4.0, 0.0), (1.0, 0.0), (1.0, 0.0), (3.0, 0.0)],
            )
            .unwrap(),
        ),
        (true, ExactHostSparseClass::Complex(NumericDType::F32)) => Value::SparseTensor(
            SparseTensor::new_complex_f32(
                rows,
                columns,
                vec![0, 2, 4],
                vec![0, 1, 0, 1],
                vec![(4.0, 0.0), (1.0, 0.0), (1.0, 0.0), (3.0, 0.0)],
            )
            .unwrap(),
        ),
        (false, ExactHostSparseClass::Numeric(NumericDType::F64)) => Value::Tensor(
            Tensor::new(
                if rows == 0 || columns == 0 {
                    vec![]
                } else {
                    vec![4.0, 1.0, 1.0, 3.0]
                },
                vec![rows, columns],
            )
            .unwrap(),
        ),
        (false, ExactHostSparseClass::Numeric(NumericDType::F32)) => Value::Tensor(
            Tensor::from_f32(
                if rows == 0 || columns == 0 {
                    vec![]
                } else {
                    vec![4.0, 1.0, 1.0, 3.0]
                },
                vec![rows, columns],
            )
            .unwrap(),
        ),
        (false, ExactHostSparseClass::Complex(NumericDType::F64)) => Value::ComplexTensor(
            runmat_value::ComplexTensor::new(
                if rows == 0 || columns == 0 {
                    vec![]
                } else {
                    vec![(4.0, 0.0), (1.0, 0.0), (1.0, 0.0), (3.0, 0.0)]
                },
                vec![rows, columns],
            )
            .unwrap(),
        ),
        (false, ExactHostSparseClass::Complex(NumericDType::F32)) => Value::ComplexTensor(
            runmat_value::ComplexTensor::from_f32(
                if rows == 0 || columns == 0 {
                    vec![]
                } else {
                    vec![(4.0, 0.0), (1.0, 0.0), (1.0, 0.0), (3.0, 0.0)]
                },
                vec![rows, columns],
            )
            .unwrap(),
        ),
        _ => panic!("test factor requires a supported native floating class"),
    };
    let type_name = match data.kind() {
        NumericFactorKind::Lu => "lu",
        NumericFactorKind::Cholesky => "chol",
        NumericFactorKind::Qr => "qr",
        NumericFactorKind::Cod => "cod",
    };
    let original = futures::executor::block_on(crate::call_builtin_async_in_context(
        context,
        "decomposition",
        &[matrix, Value::String(type_name.into())],
    ))
    .expect("selected public decomposition constructor")
    .with_owned(|value| value);
    let Value::Object(original) = original else {
        panic!("constructor must publish an object")
    };
    assert_eq!(original.numeric_factor_data().unwrap().kind(), data.kind());
    Value::Object(original)
}

fn payload_with_factor(data: &NumericFactorData) -> ValuePayload {
    let context = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let mut payload = encode_durable_value(&selected_object(data, &context)).unwrap();
    let ValuePayload::Inline(encoded) = &mut payload else {
        panic!("selected constructor must encode inline")
    };
    let InlineValue::DurablePlainObject(encoded) = encoded.as_mut() else {
        panic!("selected constructor must encode as a durable object")
    };
    encoded.decomposition = Some(DurableDecompositionPayload::Data(Box::new(
        super::encode::encode_factor_for_test(data, "fixture").unwrap(),
    )));
    payload
}

fn assert_roundtrip(data: NumericFactorData, expected: NumericFactorKind) {
    let payload = payload_with_factor(&data);
    validate_durable_payload(&payload).expect("selected recipe and closed numerical wire DATA");
    let ValuePayload::Inline(encoded) = &payload else {
        panic!("durable inline value")
    };
    let InlineValue::DurablePlainObject(encoded) = encoded.as_ref() else {
        panic!("durable object")
    };
    assert!(matches!(
        encoded.decomposition.as_ref(),
        Some(DurableDecompositionPayload::Data(_))
    ));
    assert!(
        decode_inline_value(&payload).is_err(),
        "V2 transport must not admit DATA"
    );
    let Value::Object(restored) = decode_durable_value(&payload).unwrap() else {
        panic!("restored object")
    };
    assert_eq!(restored.numeric_factor_data().unwrap().kind(), expected);
    assert_eq!(restored.numeric_factor_data(), Some(&data));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn durable_factor_wire_variants_roundtrip_with_selected_constructor_metadata() {
    assert_roundtrip(
        sealed(NumericFactorCandidate::Lu {
            lower: block([1.0, 1.0]),
            upper: block([2.0, 3.0]),
            rows: identity(),
            columns: identity(),
            row_scale: None,
        }),
        NumericFactorKind::Lu,
    );
    assert_roundtrip(
        sealed(NumericFactorCandidate::Cholesky {
            lower: block([2.0, 3.0]),
            permutation: identity(),
        }),
        NumericFactorKind::Cholesky,
    );
    assert_roundtrip(
        sealed(NumericFactorCandidate::Qr(qr_first())),
        NumericFactorKind::Qr,
    );
    assert_roundtrip(
        sealed(NumericFactorCandidate::Cod {
            first: qr_first(),
            completion: Some(cod_completion()),
        }),
        NumericFactorKind::Cod,
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn malformed_factor_map_or_foreign_class_cannot_acquire_native_data() {
    let value = sealed(NumericFactorCandidate::Cholesky {
        lower: block([2.0, 3.0]),
        permutation: identity(),
    });
    let mut payload = payload_with_factor(&value);
    let ValuePayload::Inline(encoded) = &mut payload else {
        unreachable!()
    };
    let InlineValue::DurablePlainObject(encoded) = encoded.as_mut() else {
        unreachable!()
    };
    let Some(DurableDecompositionPayload::Data(data)) = &mut encoded.decomposition else {
        unreachable!()
    };
    let DurableNumericFactor::Cholesky { permutation, .. } = data.as_mut() else {
        unreachable!()
    };
    permutation.forward[0] = 2;
    assert!(
        validate_durable_payload(&payload).is_err(),
        "out-of-range permutation must fail before document acceptance"
    );
    assert!(
        decode_durable_value(&payload).is_err(),
        "out-of-range permutation must fail"
    );

    let mut payload = payload_with_factor(&value);
    let ValuePayload::Inline(encoded) = &mut payload else {
        unreachable!()
    };
    let InlineValue::DurablePlainObject(encoded) = encoded.as_mut() else {
        unreachable!()
    };
    encoded.class_identity = runmat_types::ClassIdentity::new("OtherClass").unwrap();
    assert!(
        validate_durable_payload(&payload).is_err(),
        "foreign class cannot enter an accepted preference document"
    );
    assert!(
        decode_durable_value(&payload).is_err(),
        "foreign class cannot acquire a factor"
    );

    let mut payload = payload_with_factor(&value);
    let ValuePayload::Inline(encoded) = &mut payload else {
        unreachable!()
    };
    let InlineValue::DurablePlainObject(encoded) = encoded.as_mut() else {
        unreachable!()
    };
    encoded.decomposition = None;
    assert!(
        validate_durable_payload(&payload).is_err(),
        "decomposition class cannot silently lose its required native DATA"
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn factor_wire_and_native_import_reject_the_same_numerical_corruption() {
    let value = sealed(NumericFactorCandidate::Cholesky {
        lower: block([2.0, 3.0]),
        permutation: identity(),
    });
    for defect in 0..3 {
        let mut payload = payload_with_factor(&value);
        let ValuePayload::Inline(encoded) = &mut payload else {
            unreachable!()
        };
        let InlineValue::DurablePlainObject(encoded) = encoded.as_mut() else {
            unreachable!()
        };
        let Some(DurableDecompositionPayload::Data(data)) = &mut encoded.decomposition else {
            unreachable!()
        };
        let DurableNumericFactor::Cholesky { lower, permutation } = data.as_mut() else {
            unreachable!()
        };
        match defect {
            0 => lower.little_endian_data[..8].copy_from_slice(&f64::NAN.to_bits().to_le_bytes()),
            1 => lower.row_indices[1] = 0,
            2 => permutation.inverse[0] = 1,
            _ => unreachable!(),
        }
        assert!(
            validate_durable_payload(&payload).is_err(),
            "wire corruption {defect}"
        );
        assert!(
            decode_durable_value(&payload).is_err(),
            "import corruption {defect}"
        );
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn complex_lu_real_scale_has_the_same_wire_and_native_precision_contract() {
    let lower = SparseTensor::new_complex(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![(1.0, 0.0), (1.0, 0.0)],
    )
    .unwrap();
    let upper = SparseTensor::new_complex(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![(2.0, 0.0), (3.0, 0.0)],
    )
    .unwrap();
    let factor = sealed(NumericFactorCandidate::Lu {
        lower,
        upper,
        rows: identity(),
        columns: identity(),
        row_scale: Some(HostNumericBuffer::from_numeric_storage(
            NumericStorage::F64(vec![1.0, 2.0]),
        )),
    });
    let payload = payload_with_factor(&factor);
    validate_durable_payload(&payload).unwrap();
    let Value::Object(decoded) = decode_durable_value(&payload).unwrap() else {
        panic!("restored factor object")
    };
    assert_eq!(decoded.numeric_factor_data(), Some(&factor));

    let lower = SparseTensor::new_complex_f32(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![(1.0, 0.0), (1.0, 0.0)],
    )
    .unwrap();
    let upper = SparseTensor::new_complex_f32(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![(2.0, 0.0), (3.0, 0.0)],
    )
    .unwrap();
    let factor = sealed(NumericFactorCandidate::Lu {
        lower,
        upper,
        rows: identity(),
        columns: identity(),
        row_scale: Some(HostNumericBuffer::from_numeric_storage(
            NumericStorage::F32(vec![1.0, 2.0]),
        )),
    });
    let payload = payload_with_factor(&factor);
    validate_durable_payload(&payload).unwrap();
    let Value::Object(decoded) = decode_durable_value(&payload).unwrap() else {
        panic!("restored factor object")
    };
    assert_eq!(decoded.numeric_factor_data(), Some(&factor));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_preference_read_reopens_factor_data_without_document_time_import() {
    use crate::context::{
        IsolatedMemoryPreferenceService, RuntimeContext, RuntimePreferenceService,
    };
    use crate::execution::RuntimeExecutionService;
    use std::rc::Rc;

    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let expected = selected_object(
        &sealed(NumericFactorCandidate::DenseCholesky {
            lower: dense(vec![2.0, 1.0, 0.0, 3.0], 2, 2),
        }),
        &runtime,
    );
    let service = IsolatedMemoryPreferenceService::new();
    futures::executor::block_on(service.set_many(
        runtime.clone(),
        "factors".into(),
        vec![("chol".into(), expected.clone())],
    ))
    .unwrap();
    let stored = futures::executor::block_on(service.document_bytes()).unwrap();
    let reopened = IsolatedMemoryPreferenceService::from_document_bytes(&stored).unwrap();
    let actual = futures::executor::block_on(reopened.group_values(runtime, "factors".into()))
        .unwrap()
        .unwrap();
    assert_eq!(actual["chol"], expected);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_structural_preference_roundtrips_without_refactor_or_native_data() {
    use crate::context::{
        IsolatedMemoryPreferenceService, RuntimeContext, RuntimePreferenceService,
    };
    use crate::execution::RuntimeExecutionService;
    use std::rc::Rc;

    crate::builtins::wasm_registry::register_all();
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let service = IsolatedMemoryPreferenceService::new();
    for (name, tag) in [
        ("diagonal", DurableStructuralDecomposition::Diagonal),
        ("triangular", DurableStructuralDecomposition::Triangular),
    ] {
        let matrix = Value::Tensor(Tensor::new(vec![2.0, 0.0, 0.0, 3.0], vec![2, 2]).unwrap());
        let original = futures::executor::block_on(crate::call_builtin_async_in_context(
            &runtime,
            "decomposition",
            &[matrix, Value::String(name.into())],
        ))
        .expect("selected public structural constructor")
        .with_owned(|value| value);
        let Value::Object(object) = &original else {
            panic!("constructor must publish an object")
        };
        assert!(object.numeric_factor_data().is_none());
        let ValuePayload::Inline(encoded) = encode_durable_value(&original).unwrap() else {
            panic!("durable inline object")
        };
        let InlineValue::DurablePlainObject(encoded) = encoded.as_ref() else {
            panic!("durable plain object")
        };
        assert_eq!(
            encoded.decomposition,
            Some(DurableDecompositionPayload::Structural(tag))
        );
        futures::executor::block_on(service.set_many(
            runtime.clone(),
            "structural".into(),
            vec![(name.into(), original.clone())],
        ))
        .expect("selected preference commit");
        let reopened = IsolatedMemoryPreferenceService::from_document_bytes(
            &futures::executor::block_on(service.document_bytes()).unwrap(),
        )
        .unwrap();
        let actual = futures::executor::block_on(
            reopened.group_values(runtime.clone(), "structural".into()),
        )
        .unwrap()
        .unwrap();
        assert_eq!(actual[name], original);
    }
}
