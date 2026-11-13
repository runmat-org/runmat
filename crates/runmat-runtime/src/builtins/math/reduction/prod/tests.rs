use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::{
    HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView, IntegerElementType,
};
use runmat_value::{IntValue, IntegerStorage};

fn prod_builtin(value: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    block_on(super::prod_builtin(None, value, rest))
}

fn values(tensor: &Tensor) -> Vec<f64> {
    tensor.materialize_f64()
}

fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
    error.identifier()
}

#[test]
fn prod_descriptor_signatures_and_errors() {
    let labels: Vec<&str> = PROD_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"B = prod(A)"));
    assert!(labels.contains(&"B = prod(A, dim)"));
    assert!(labels.contains(&"B = prod(A, \"all\")"));
    assert!(labels.contains(&"B = prod(A, missingflag)"));
    assert!(labels.contains(&"B = prod(A, outtype)"));
    assert!(labels.contains(&"B = prod(A, dim, missingflag)"));
    assert!(labels.contains(&"B = prod(A, missingflag, dim)"));
    assert!(!labels.iter().any(|label| label.contains("nanflag")));
    assert!(!labels.iter().any(|label| label.contains("like")));
    let missingflag = PROD_DESCRIPTOR
        .signatures
        .iter()
        .flat_map(|signature| signature.inputs)
        .find(|input| input.name == "missingflag")
        .expect("missingflag descriptor");
    assert_eq!(missingflag.default, Some("\"includemissing\""));
    assert!(missingflag.description.contains("\"omitmissing\""));
    assert!(PROD_DESCRIPTOR
        .errors
        .iter()
        .any(|err| err.code == PROD_ERROR_INVALID_ARGUMENT.code));
    assert!(PROD_DESCRIPTOR
        .errors
        .iter()
        .any(|err| err.code == PROD_ERROR_INVALID_INPUT.code));
    assert!(PROD_DESCRIPTOR
        .errors
        .iter()
        .any(|err| err.code == PROD_ERROR_COMPLEX_UNSUPPORTED.code));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prod_scalar_num() {
    let result = prod_builtin(
        Value::Num(5.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("prod");
    assert_eq!(result, Value::Num(5.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prod_matrix_default_dimension() {
    let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
    let result = prod_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("prod");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [1, 3]);
            assert_eq!(values(&out), vec![4.0, 10.0, 18.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prod_matrix_dimension_two() {
    let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
    let result = prod_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::I32(
            2,
        ))]),
    )
    .expect("prod");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [2, 1]);
            assert_eq!(values(&out), vec![6.0, 120.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn prod_double_path_reads_typed_integer_storage_exactly() {
    let tensor = Tensor::new_integer(IntegerStorage::I16(vec![1, 4, 2, 5, 3, 6]), vec![2, 3])
        .expect("tensor");

    let result = prod_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("prod");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [1, 3]);
            assert_eq!(values(&out), vec![4.0, 10.0, 18.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prod_all_dimension() {
    let tensor = Tensor::new((1..=6).map(|v| v as f64).collect::<Vec<_>>(), vec![2, 3]).unwrap();
    let result = prod_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("all")]),
    )
    .expect("prod");
    assert_eq!(result, Value::Num(720.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prod_vecdim_multiple_axes() {
    let tensor = Tensor::new(
        (1..=24).map(|v| v as f64).collect::<Vec<_>>(),
        vec![3, 4, 2],
    )
    .unwrap();
    let dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
    let result = prod_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(dims)]),
    )
    .expect("prod");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [1, 4, 1]);
            assert_eq!(values(&out), vec![16380.0, 587520.0, 4021920.0, 16030080.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prod_with_omit_nan_default_dimension() {
    let tensor = Tensor::new(vec![2.0, f64::NAN, 4.0], vec![3, 1]).unwrap();
    let result = prod_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("omitnan")]),
    )
    .expect("prod");
    assert_eq!(result, Value::Num(8.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prod_generic_missing_flags_match_legacy_aliases() {
    let tensor = || Tensor::new(vec![2.0, f64::NAN, 4.0], vec![3, 1]).unwrap();
    let omitted = prod_builtin(
        Value::Tensor(tensor()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("omitmissing")]),
    )
    .expect("prod omitmissing");
    assert_eq!(omitted, Value::Num(8.0));
    let included = prod_builtin(
        Value::Tensor(tensor()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
            "includemissing",
        )]),
    )
    .expect("prod includemissing");
    match included {
        Value::Num(value) => assert!(value.is_nan()),
        other => panic!("expected scalar NaN, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prod_with_include_nan_propagates() {
    let tensor = Tensor::new(vec![2.0, f64::NAN, 4.0], vec![3, 1]).unwrap();
    let result = prod_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("prod");
    match result {
        Value::Num(v) => assert!(v.is_nan()),
        other => panic!("expected scalar NaN, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prod_dimension_greater_than_ndims_returns_input() {
    let tensor = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
    let original = tensor.clone();
    let result = prod_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::I32(
            5,
        ))]),
    )
    .expect("prod");
    match result {
        Value::Tensor(out) => assert_eq!(out, original),
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn prod_integer_dimension_beyond_rank_still_honors_output_class() {
    let storages = vec![
        IntegerStorage::I8(vec![2, 3]),
        IntegerStorage::I16(vec![2, 3]),
        IntegerStorage::I32(vec![2, 3]),
        IntegerStorage::I64(vec![2, 3]),
        IntegerStorage::U8(vec![2, 3]),
        IntegerStorage::U16(vec![2, 3]),
        IntegerStorage::U32(vec![2, 3]),
        IntegerStorage::U64(vec![2, 3]),
    ];
    for storage in storages {
        let tensor = Tensor::new_integer(storage.clone(), vec![1, 2]).expect("integer tensor");
        let default = prod_builtin(
            Value::Tensor(tensor.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::I32(4),
            )]),
        )
        .expect("default prod");
        let explicit_double = prod_builtin(
            Value::Tensor(tensor.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(IntValue::I32(4)),
                Value::from("double"),
            ]),
        )
        .expect("double prod");
        let explicit_default = prod_builtin(
            Value::Tensor(tensor.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(IntValue::I32(4)),
                Value::from("default"),
            ]),
        )
        .expect("explicit default prod");
        for result in [default, explicit_default, explicit_double] {
            let Value::Tensor(output) = result else {
                panic!("expected tensor output");
            };
            assert_eq!(output.shape(), [1, 2]);
            assert_eq!(output.numeric_dtype(), NumericDType::F64);
            assert_eq!(values(&output), vec![2.0, 3.0]);
        }
        let native = prod_builtin(
            Value::Tensor(tensor),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(IntValue::I32(4)),
                Value::from("native"),
            ]),
        )
        .expect("native prod");
        let Value::Tensor(output) = native else {
            panic!("expected native tensor output");
        };
        assert_eq!(output.integer_storage(), Some(&storage));
    }
}

#[test]
fn prod_empty_integer_identities_preserve_shape_and_output_class() {
    let storages = vec![
        IntegerStorage::I8(Vec::new()),
        IntegerStorage::I16(Vec::new()),
        IntegerStorage::I32(Vec::new()),
        IntegerStorage::I64(Vec::new()),
        IntegerStorage::U8(Vec::new()),
        IntegerStorage::U16(Vec::new()),
        IntegerStorage::U32(Vec::new()),
        IntegerStorage::U64(Vec::new()),
    ];
    for storage in storages {
        let tensor =
            Tensor::new_integer(storage.clone(), vec![0, 3]).expect("empty integer tensor");
        let default = prod_builtin(
            Value::Tensor(tensor.clone()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .expect("default empty prod");
        let Value::Tensor(default) = default else {
            panic!("expected default tensor output");
        };
        assert_eq!(default.shape(), [1, 3]);
        assert_eq!(default.numeric_dtype(), NumericDType::F64);
        assert_eq!(values(&default), vec![1.0; 3]);

        let native = prod_builtin(
            Value::Tensor(tensor.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
        )
        .expect("native empty prod");
        let Value::Tensor(native) = native else {
            panic!("expected native tensor output");
        };
        assert_eq!(native.shape(), [1, 3]);
        assert_eq!(native.integer_storage(), Some(&storage.ones_like(3)));

        let default_dim_two = prod_builtin(
            Value::Tensor(tensor.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::I32(2),
            )]),
        )
        .expect("default empty prod along dimension two");
        let Value::Tensor(default_dim_two) = default_dim_two else {
            panic!("expected default empty tensor output");
        };
        assert_eq!(default_dim_two.shape(), [0, 1]);
        assert_eq!(default_dim_two.numeric_dtype(), NumericDType::F64);
        assert!(default_dim_two.is_empty());

        let native_dim_two = prod_builtin(
            Value::Tensor(tensor.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(IntValue::I32(2)),
                Value::from("native"),
            ]),
        )
        .expect("native empty prod along dimension two");
        let Value::Tensor(native_dim_two) = native_dim_two else {
            panic!("expected native empty tensor output");
        };
        assert_eq!(native_dim_two.shape(), [0, 1]);
        assert_eq!(native_dim_two.integer_storage(), Some(&storage));

        let default_all = prod_builtin(
            Value::Tensor(tensor.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("all")]),
        )
        .expect("default empty prod across all dimensions");
        assert_eq!(default_all, Value::Num(1.0));

        let native_all = prod_builtin(
            Value::Tensor(tensor),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("all"),
                Value::from("native"),
            ]),
        )
        .expect("native empty prod across all dimensions");
        assert_eq!(
            native_all,
            Value::Int(storage.ones_like(1).value_at(0).unwrap())
        );

        let zero_by_zero =
            Tensor::new_integer(storage.clone(), vec![0, 0]).expect("zero-by-zero tensor");
        let zero_by_zero = prod_builtin(
            Value::Tensor(zero_by_zero),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
        )
        .expect("zero-by-zero prod");
        let Value::Tensor(zero_by_zero) = zero_by_zero else {
            panic!("expected zero-by-zero tensor output");
        };
        assert_eq!(zero_by_zero.shape(), [1, 0]);
        assert_eq!(zero_by_zero.integer_storage(), Some(&storage));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prod_native_integer_scalar() {
    let value = Value::Int(IntValue::I16(4));
    let result = prod_builtin(
        value,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
    )
    .expect("prod");
    assert_eq!(result, Value::Int(IntValue::I16(4)));
}

#[test]
fn prod_native_integer_tensor_forms_read_typed_storage_exactly() {
    let tensor = Tensor::new_integer(IntegerStorage::I16(vec![1, 4, 2, 5, 3, 6]), vec![2, 3])
        .expect("tensor");

    let result = prod_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
    )
    .expect("prod");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [1, 3]);
            assert_eq!(
                out.integer_storage(),
                Some(&IntegerStorage::I16(vec![4, 10, 18]))
            );
        }
        other => panic!("expected tensor result, got {other:?}"),
    }

    let tensor =
        Tensor::new_integer(IntegerStorage::I16(vec![2, 3, 4]), vec![3, 1]).expect("tensor");
    let result = prod_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("all"),
            Value::from("native"),
        ]),
    )
    .expect("prod all");
    assert_eq!(result, Value::Int(IntValue::I16(24)));

    let tensor = Tensor::new_integer(
        IntegerStorage::I32((1..=12).collect::<Vec<i32>>()),
        vec![2, 3, 2],
    )
    .expect("tensor");
    let dims = Tensor::new_integer(IntegerStorage::U8(vec![1, 3]), vec![1, 2]).expect("dims");

    let result = prod_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(dims),
            Value::from("native"),
        ]),
    )
    .expect("prod vecdim");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [1, 3, 1]);
            assert_eq!(
                out.integer_storage(),
                Some(&IntegerStorage::I32(vec![112, 1080, 3960]))
            );
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn prod_native_template_reads_typed_integer_scalar_storage_exactly() {
    let tensor = Tensor::new_integer(IntegerStorage::U32(vec![77]), vec![1, 1]).unwrap();
    let result = prod_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
    )
    .unwrap();
    assert_eq!(result, Value::Int(IntValue::U32(77)));
    let result = prod_builtin(
        Value::Bool(true),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
    )
    .unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prod_rejects_undocumented_like_option() {
    let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let err = prod_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("like"),
            Value::Num(0.0),
        ]),
    )
    .unwrap_err();
    assert_eq!(err.identifier(), PROD_ERROR_INVALID_ARGUMENT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prod_rejects_complex_input() {
    let err = prod_builtin(
        Value::Complex(1.0, 2.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .unwrap_err();
    assert_eq!(
        error_identifier(&err),
        PROD_ERROR_COMPLEX_UNSUPPORTED.identifier
    );
    assert!(err.message().contains("complex inputs"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prod_too_many_output_class_specifiers_is_invalid_argument() {
    let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let err = prod_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("native"),
            Value::from("double"),
        ]),
    )
    .unwrap_err();
    assert_eq!(
        error_identifier(&err),
        PROD_ERROR_INVALID_ARGUMENT.identifier
    );
    assert!(err.message().contains(PROD_ERROR_INVALID_ARGUMENT.message));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prod_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let result = prod_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .expect("prod");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape(), [1, 3]);
        assert_eq!(values(&gathered), vec![4.0, 10.0, 18.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prod_gpu_all_reduction() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.5, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let result = prod_builtin(
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("all")]),
        )
        .expect("prod");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered, Tensor::new(vec![36.0], vec![1, 1]).unwrap());
    });
}

#[test]
fn prod_integer_gpu_dimension_beyond_rank_returns_resident_double() {
    test_support::with_test_provider(|provider| {
        let handle = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[9_007_199_254_740_992, 5]),
                shape: &[1, 2],
            })
            .expect("upload integer gpuArray");
        let result = prod_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::I32(4),
            )]),
        )
        .expect("prod");
        let Value::GpuTensor(output) = &result else {
            panic!("expected resident gpuArray output");
        };
        assert_eq!(output.shape(), vec![1, 2]);
        assert_eq!(runmat_accelerate_api::handle_integer_type(output), None);
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.numeric_dtype(), NumericDType::F64);
        assert_eq!(values(&gathered), vec![9_007_199_254_740_992.0, 5.0]);
    });
}

#[test]
fn prod_empty_integer_gpu_identities_preserve_class_shape_and_residency() {
    test_support::with_test_provider(|provider| {
        let upload = || {
            provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U16(&[]),
                    shape: &[0, 3],
                })
                .expect("upload empty integer gpuArray")
        };
        let default = prod_builtin(
            Value::GpuTensor(upload()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .expect("default prod");
        let Value::GpuTensor(default_handle) = &default else {
            panic!("expected resident default output");
        };
        assert_eq!(default_handle.shape(), vec![1, 3]);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(default_handle),
            None
        );
        let default = test_support::gather(default).expect("gather default prod");
        assert_eq!(default.numeric_dtype(), NumericDType::F64);
        assert_eq!(values(&default), vec![1.0; 3]);

        let native = prod_builtin(
            Value::GpuTensor(upload()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
        )
        .expect("native prod");
        let Value::GpuTensor(native) = native else {
            panic!("expected resident native output");
        };
        assert_eq!(native.shape(), vec![1, 3]);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&native),
            Some(IntegerElementType::U16)
        );
        assert_eq!(
            block_on(provider.download_integer(&native))
                .expect("download native prod")
                .data,
            HostIntegerDataOwned::U16(vec![1; 3])
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn prod_native_integer_gpu_stays_resident() {
    match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) {
        Ok(_) => {}
        Err(error) if error.to_string() == "wgpu: no compatible adapter found" => return,
        Err(error) => panic!("register wgpu provider failed: {error:?}"),
    }
    let provider = runmat_accelerate_api::provider().expect("provider");
    let handle = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U8(&[20, 30, 15, 10]),
            shape: &[2, 2],
        })
        .expect("upload native integer gpuArray");
    let result = prod_builtin(
        Value::GpuTensor(handle),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Int(IntValue::I32(1)),
            Value::from("native"),
        ]),
    )
    .expect("prod native integer gpuArray");
    let Value::GpuTensor(out) = result else {
        panic!("expected resident gpuArray result");
    };
    assert_eq!(
        runmat_accelerate_api::handle_integer_type(&out),
        Some(IntegerElementType::U8)
    );
    assert_eq!(
        block_on(provider.download_integer(&out))
            .expect("download native prod")
            .data,
        HostIntegerDataOwned::U8(vec![255, 150])
    );
}

#[test]
fn prod_native_integer_gpu_vecdim_and_all_nd_stay_resident() {
    test_support::with_test_provider(|provider| {
        let values: Vec<u16> = (1..=12).collect();
        let handle = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U16(&values),
                shape: &[2, 3, 2],
            })
            .expect("upload native integer gpuArray");
        let dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        let vecdim = prod_builtin(
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(dims),
                Value::from("native"),
            ]),
        )
        .expect("prod native integer gpuArray vecdim");
        let all = prod_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("all"),
                Value::from("native"),
            ]),
        )
        .expect("prod native integer gpuArray all");
        let Value::GpuTensor(vecdim_out) = vecdim else {
            panic!("expected resident vecdim gpuArray result");
        };
        let Value::GpuTensor(all_out) = all else {
            panic!("expected resident all gpuArray result");
        };
        assert_eq!(vecdim_out.shape(), vec![1, 3, 1]);
        assert_eq!(all_out.shape(), vec![1, 1, 1]);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&vecdim_out),
            Some(IntegerElementType::U16)
        );
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&all_out),
            Some(IntegerElementType::U16)
        );
        assert_eq!(
            block_on(provider.download_integer(&vecdim_out))
                .expect("download native vecdim prod")
                .data,
            HostIntegerDataOwned::U16(vec![112, 1080, 3960])
        );
        assert_eq!(
            block_on(provider.download_integer(&all_out))
                .expect("download native all prod")
                .data,
            HostIntegerDataOwned::U16(vec![u16::MAX])
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prod_gpu_omit_nan_falls_back_to_host() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![f64::NAN, 2.0, f64::NAN, 4.0], vec![2, 2]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let result = prod_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("omitnan")]),
        )
        .expect("prod");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape(), [1, 2]);
        assert_eq!(values(&gathered), vec![2.0, 4.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn prod_wgpu_dim1_matches_cpu() {
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
    let cpu = host::execute(
        Value::Tensor(tensor.clone()),
        &ParsedArguments {
            selection: DimSelection::Dim(1),
            nan_mode: ReductionNaN::Include,
            output: OutputTemplate::Double,
        },
    )
    .unwrap();
    let h =
        gpu_helpers::upload_tensor(runmat_accelerate_api::provider().unwrap(), &tensor).unwrap();
    let gpu = block_on(provider::execute(
        &h,
        &ParsedArguments {
            selection: DimSelection::Dim(1),
            nan_mode: ReductionNaN::Include,
            output: OutputTemplate::Double,
        },
        &super::super::numerical_execution::NumericalCall::new(
            runmat_builtins::NumericalReductionKind::Product,
            None,
        )
        .unwrap(),
    ))
    .unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    match (cpu, gathered) {
        (Value::Tensor(ct), gt) => {
            assert_eq!(gt.shape(), ct.shape());
            for (a, b) in values(&gt).iter().zip(values(&ct).iter()) {
                assert!((a - b).abs() < 1e-8);
            }
        }
        _ => panic!("unexpected shapes"),
    }
}

#[test]
fn default_integer_product_converts_before_native_saturation_and_single_stays_single() {
    let value =
        Value::Tensor(Tensor::new_integer(IntegerStorage::I8(vec![100, 2]), vec![1, 2]).unwrap());
    let ordinary = prod_builtin(value.clone(), test_support::builtin_arguments(vec![])).unwrap();
    let native = prod_builtin(
        value,
        test_support::builtin_arguments(vec![Value::from("native")]),
    )
    .unwrap();
    assert_eq!(ordinary, Value::Num(200.0));
    assert_eq!(native, Value::Int(IntValue::I8(127)));
    let value = Value::Tensor(Tensor::from_f32(vec![2., 3.], vec![1, 2]).unwrap());
    let Value::Tensor(single) =
        prod_builtin(value.clone(), test_support::builtin_arguments(vec![])).unwrap()
    else {
        panic!("single tensor")
    };
    assert_eq!(single.as_f32_slice().unwrap(), &[6.]);
    assert_eq!(
        prod_builtin(
            value,
            test_support::builtin_arguments(vec![Value::from("double")])
        )
        .unwrap(),
        Value::Num(6.)
    );
}
#[test]
fn only_vector_shaped_empty_product_selectors_choose_default_axis() {
    let input = Value::Tensor(Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]).unwrap());
    for shape in [vec![1, 0], vec![0, 1]] {
        let selector = Value::Tensor(Tensor::new(vec![], shape).unwrap());
        let result = prod_builtin(
            input.clone(),
            test_support::builtin_arguments(vec![selector]),
        )
        .unwrap();
        let Value::Tensor(output) = result else {
            panic!("tensor")
        };
        assert_eq!(output.materialize_f64(), vec![2., 12.]);
    }
    let selector = Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap());
    assert_eq!(
        prod_builtin(input, test_support::builtin_arguments(vec![selector]))
            .unwrap_err()
            .identifier(),
        Some("RunMat:prod:InvalidArgument")
    );
}
