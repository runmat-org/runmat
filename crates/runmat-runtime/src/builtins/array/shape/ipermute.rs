//! MATLAB-compatible `ipermute` builtin with GPU-aware semantics for RunMat.
//!
//! This module implements the inverse permutation primitive that undoes the action of
//! `permute`. It shares the same validation rules and GPU plumbing as `permute`, but
//! automatically computes the inverse order vector before delegating to the shared
//! permutation helpers.

use crate::aggregate::structure::operations as struct_ops;

use crate::builtins::array::shape::permute::{
    parse_order_argument, permute_char_array, permute_complex_tensor, permute_gpu,
    permute_logical_array, permute_string_array, permute_tensor, validate_rank,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
pub use runmat_builtins::IPERMUTE_DESCRIPTOR;
use runmat_builtins::*;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

fn ipermute_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("ipermute")
        .build()
}

#[runtime_builtin(
    name = "ipermute",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::shape::ipermute"
)]
async fn ipermute_builtin(value: Value, order: Value) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &IPERMUTE_DESCRIPTOR,
        &IPERMUTE_ERROR_INVALID_ORDER,
        "ipermute",
    )?;

    let order_vec = parse_order_argument("ipermute", order)
        .map_err(|error| super::error::declared(error, &IPERMUTE_ERROR_INVALID_ORDER))?;
    let inverse = inverse_permutation(&order_vec);

    match value {
        Value::Tensor(t) => {
            validate_rank("ipermute", &order_vec, t.shape().len()).map_err(|error| super::error::declared(error, &IPERMUTE_ERROR_RANK_MISMATCH))?;
            Ok(permute_tensor("ipermute", t, &inverse)
                .map(tensor::tensor_into_value)?)
        }
        Value::LogicalArray(la) => {
            validate_rank("ipermute", &order_vec, la.shape.len()).map_err(|error| super::error::declared(error, &IPERMUTE_ERROR_RANK_MISMATCH))?;
            Ok(permute_logical_array("ipermute", la, &inverse)
                .map(Value::LogicalArray)?)
        }
        Value::ComplexTensor(ct) => {
            validate_rank("ipermute", &order_vec, ct.shape().len()).map_err(|error| super::error::declared(error, &IPERMUTE_ERROR_RANK_MISMATCH))?;
            Ok(permute_complex_tensor("ipermute", ct, &inverse)
                .map(Value::ComplexTensor)?)
        }
        Value::StringArray(sa) => {
            validate_rank("ipermute", &order_vec, sa.shape.len()).map_err(|error| super::error::declared(error, &IPERMUTE_ERROR_RANK_MISMATCH))?;
            Ok(permute_string_array("ipermute", sa, &inverse)
                .map(Value::StringArray)?)
        }
        Value::CharArray(ca) => {
            validate_rank("ipermute", &order_vec, ca.shape.len()).map_err(|error| super::error::declared(error, &IPERMUTE_ERROR_RANK_MISMATCH))?;
            Ok(permute_char_array("ipermute", ca, &inverse)
                .map(Value::CharArray)?)
        }
        Value::Struct(structure) => {
            validate_rank("ipermute", &order_vec, 2).map_err(|error| super::error::declared(error, &IPERMUTE_ERROR_RANK_MISMATCH))?;
            Ok(Value::Struct(structure))
        }
        Value::StructArray(array) => {
            validate_rank(
                "ipermute",
                &order_vec,
                crate::builtins::common::shape::effective_rank(array.shape()),
            ).map_err(|error| super::error::declared(error, &IPERMUTE_ERROR_RANK_MISMATCH))?;
            let zero_based = inverse.iter().map(|dimension| dimension - 1).collect::<Vec<_>>();
            struct_ops::permute(array,&zero_based).map(Value::StructArray)
                .map_err(|error|struct_ops::remap(error,|message|ipermute_error(message)))
        }
        Value::GpuTensor(handle) => {
            validate_rank("ipermute", &order_vec, handle.shape().len()).map_err(|error| super::error::declared(error, &IPERMUTE_ERROR_RANK_MISMATCH))?;
            Ok(ipermute_gpu(handle, &inverse).await?)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor = tensor::value_into_tensor_for("ipermute", value)
                .map_err(|e| ipermute_error(e))?;
            validate_rank("ipermute", &order_vec, tensor.shape().len()).map_err(|error| super::error::declared(error, &IPERMUTE_ERROR_RANK_MISMATCH))?;
            Ok(permute_tensor("ipermute", tensor, &inverse)
                .map(tensor::tensor_into_value)?)
        }
        other => Err(super::error::declared(ipermute_error(format!(
            "ipermute: unsupported input type {:?}; expected numeric, logical, complex, string, or gpuArray values",
            other
        )), &IPERMUTE_ERROR_UNSUPPORTED_INPUT)),
    }
}

async fn ipermute_gpu(
    handle: GpuTensorHandle,
    inverse_order: &[usize],
) -> crate::BuiltinResult<Value> {
    permute_gpu("ipermute", handle, inverse_order).await
}

fn inverse_permutation(order: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0usize; order.len()];
    for (pos, &idx) in order.iter().enumerate() {
        let place = idx
            .checked_sub(1)
            .expect("parse_order_argument guarantees indices are >= 1");
        inverse[place] = pos + 1;
    }
    inverse
}

#[cfg(test)]
pub(crate) mod tests {
    use futures::executor::block_on;
    use runmat_builtins::IPERMUTE_ERROR_INVALID_ORDER;

    fn ipermute_builtin(value: Value, order: Value) -> crate::BuiltinResult<Value> {
        block_on(super::ipermute_builtin(value, order))
    }
    use crate::aggregate::structure::operations as struct_ops;
    use crate::builtins::array::shape::permute::{
        parse_order_argument, permute_char_array, permute_complex_tensor, permute_gpu,
        permute_logical_array, permute_string_array, permute_tensor,
    };
    use crate::builtins::common::{tensor, test_support};
    use runmat_value::{
        CharArray, ComplexTensor, IntegerComplexStorage, IntegerStorage, LogicalArray, StringArray,
        StructArray, Tensor, Value,
    };

    fn make_tensor(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::new(data.to_vec(), shape.to_vec()).unwrap()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ipermute_inverts_permute() {
        let data: Vec<f64> = (1..=24).map(|n| n as f64).collect();
        let order = make_tensor(&[3.0, 1.0, 2.0], &[1, 3]);
        let order_vec =
            parse_order_argument("ipermute", Value::Tensor(order.clone())).expect("parse order");
        let original_tensor = make_tensor(&data, &[2, 3, 4]);
        let permuted_tensor =
            permute_tensor("ipermute", original_tensor.clone(), &order_vec).expect("permute");
        let permuted = tensor::tensor_into_value(permuted_tensor);
        let restored = ipermute_builtin(permuted, Value::Tensor(order)).expect("ipermute");
        match (Value::Tensor(original_tensor), restored) {
            (Value::Tensor(orig), Value::Tensor(rest)) => {
                assert_eq!(orig.shape(), rest.shape());
                assert_eq!(orig.materialize_f64(), rest.materialize_f64());
            }
            _ => panic!("expected tensor pair"),
        }
    }

    #[test]
    fn ipermute_restores_typed_complex_integer_storage_exactly() {
        let original = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![1, 2, 9_223_372_036_854_775_808, u64::MAX]),
                IntegerStorage::U64(vec![5, 6, 7, 8]),
            )
            .expect("storage"),
            vec![2, 2],
        )
        .expect("complex");
        let order = make_tensor(&[2.0, 1.0], &[1, 2]);
        let order_vec =
            parse_order_argument("ipermute", Value::Tensor(order.clone())).expect("parse order");
        let permuted = permute_complex_tensor("ipermute", original, &order_vec).expect("permute");
        let value = ipermute_builtin(Value::ComplexTensor(permuted), Value::Tensor(order))
            .expect("ipermute");

        let Value::ComplexTensor(output) = value else {
            panic!("expected complex tensor");
        };
        assert_eq!(
            output
                .integer_storage()
                .as_ref()
                .map(|storage| (storage.real(), storage.imag())),
            Some((
                &IntegerStorage::U64(vec![1, 2, 9_223_372_036_854_775_808, u64::MAX]),
                &IntegerStorage::U64(vec![5, 6, 7, 8]),
            ))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ipermute_rejects_invalid_order() {
        let order = make_tensor(&[1.0, 1.0], &[1, 2]);
        let err = ipermute_builtin(
            Value::Tensor(make_tensor(&[1.0], &[1, 1])),
            Value::Tensor(order),
        )
        .expect_err("should fail");
        assert_eq!(
            err.identifier.as_deref(),
            IPERMUTE_ERROR_INVALID_ORDER.identifier
        );
        assert!(
            err.to_string().contains("duplicate"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ipermute_requires_vector_order() {
        let order = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = ipermute_builtin(
            Value::Tensor(make_tensor(&[1.0, 2.0, 3.0, 4.0], &[4, 1])),
            Value::Tensor(order),
        )
        .expect_err("should fail");
        assert!(
            err.to_string().contains("row or column vector"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ipermute_char_array_roundtrip() {
        let chars = CharArray::from_unicode_scalars("runmat".chars().collect(), 2, 3).unwrap();
        let order = make_tensor(&[2.0, 1.0], &[1, 2]);
        let order_vec =
            parse_order_argument("ipermute", Value::Tensor(order.clone())).expect("parse order");
        let permuted =
            permute_char_array("ipermute", chars.clone(), &order_vec).expect("permute chars");
        let restored = ipermute_builtin(Value::CharArray(permuted), Value::Tensor(order))
            .expect("ipermute chars");
        match restored {
            Value::CharArray(out) => {
                assert_eq!(out.rows, chars.rows);
                assert_eq!(out.cols, chars.cols);
                assert_eq!(out.data(), chars.data());
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ipermute_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let host = make_tensor(&(0..24).map(|n| n as f64).collect::<Vec<_>>(), &[2, 3, 4]);
            let order = make_tensor(&[3.0, 1.0, 2.0], &[1, 3]);
            let view = runmat_accelerate_api::HostTensorView {
                data: &host.materialize_f64(),
                shape: host.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let order_vec = parse_order_argument("ipermute", Value::Tensor(order.clone()))
                .expect("parse order");
            let permuted = futures::executor::block_on(permute_gpu("ipermute", handle, &order_vec))
                .expect("permute gpu");
            let restored = ipermute_builtin(permuted, Value::Tensor(order)).expect("ipermute gpu");
            let gathered = test_support::gather(restored).expect("gather");
            assert_eq!(gathered.shape(), host.shape());
            assert_eq!(gathered.materialize_f64(), host.materialize_f64());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ipermute_numeric_scalar() {
        let value = Value::Num(42.0);
        let order = make_tensor(&[1.0, 2.0], &[1, 2]);
        let result = ipermute_builtin(value.clone(), Value::Tensor(order)).expect("ipermute");
        assert_eq!(result, value);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ipermute_logical_array_roundtrip() {
        let logical = LogicalArray::new(vec![0, 1, 0, 1], vec![2, 2]).unwrap();
        let order = make_tensor(&[2.0, 1.0], &[1, 2]);
        let order_vec =
            parse_order_argument("ipermute", Value::Tensor(order.clone())).expect("parse order");
        let permuted = permute_logical_array("ipermute", logical.clone(), &order_vec)
            .expect("permute logical");
        let restored = ipermute_builtin(Value::LogicalArray(permuted), Value::Tensor(order))
            .expect("ipermute logical");
        match restored {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, logical.shape);
                assert_eq!(out.data, logical.data);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ipermute_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let host = make_tensor(&(0..24).map(|n| n as f64).collect::<Vec<_>>(), &[2, 3, 4]);
        let order = make_tensor(&[3.0, 1.0, 2.0], &[1, 3]);
        let order_vec =
            parse_order_argument("ipermute", Value::Tensor(order.clone())).expect("parse order");

        let permuted_tensor =
            permute_tensor("ipermute", host.clone(), &order_vec).expect("permute host");
        let permuted = tensor::tensor_into_value(permuted_tensor);
        let cpu = ipermute_builtin(permuted, Value::Tensor(order.clone())).expect("cpu ipermute");

        let view = runmat_accelerate_api::HostTensorView {
            data: &host.materialize_f64(),
            shape: host.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let permuted_gpu =
            block_on(permute_gpu("ipermute", handle, &order_vec)).expect("permute gpu");
        let gpu = ipermute_builtin(permuted_gpu, Value::Tensor(order)).expect("gpu ipermute");
        let gathered = test_support::gather(gpu).expect("gather");

        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(ct.shape(), gathered.shape());
                assert_eq!(ct.materialize_f64(), gathered.materialize_f64());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ipermute_string_array_roundtrip() {
        let data = vec![
            "run".to_string(),
            "mat".to_string(),
            "fast".to_string(),
            "gpu".to_string(),
        ];
        let strings = StringArray::new(data.clone(), vec![2, 2]).unwrap();
        let order = make_tensor(&[2.0, 1.0], &[1, 2]);
        let order_vec =
            parse_order_argument("ipermute", Value::Tensor(order.clone())).expect("parse order");
        let permuted = permute_string_array("ipermute", strings.clone(), &order_vec)
            .expect("permute string array");
        let restored =
            ipermute_builtin(Value::StringArray(permuted), Value::Tensor(order)).expect("ipermute");
        match restored {
            Value::StringArray(out) => {
                assert_eq!(out.shape, strings.shape);
                assert_eq!(out.data(), data);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ipermute_extends_missing_dimensions() {
        let row = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5]);
        let order = make_tensor(&[2.0, 1.0, 3.0], &[1, 3]);
        let order_vec =
            parse_order_argument("ipermute", Value::Tensor(order.clone())).expect("parse order");
        let permuted = permute_tensor("ipermute", row.clone(), &order_vec).expect("permute");
        let restored = ipermute_builtin(tensor::tensor_into_value(permuted), Value::Tensor(order))
            .expect("ipermute");
        match restored {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), &[1, 5, 1]);
                assert_eq!(out.materialize_f64(), row.materialize_f64());
            }
            Value::Num(n) => {
                panic!("expected tensor result, got scalar {n}");
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn ipermute_struct_array_uses_effective_rank_and_padding() {
        let mut first = runmat_value::UnboundedStructValueBuilder::new();
        first.insert("value", Value::Num(1.0));
        let mut second = runmat_value::UnboundedStructValueBuilder::new();
        second.insert("value", Value::Num(2.0));
        let second = second.finish();
        let first = first.finish();
        let array = StructArray::new(vec![first, second], vec![2, 1, 1]).unwrap();
        let short = make_tensor(&[2.0, 1.0], &[1, 2]);
        let Value::StructArray(direct) = ipermute_builtin(
            Value::StructArray(array.clone()),
            Value::Tensor(short.clone()),
        )
        .unwrap() else {
            panic!("expected structure array");
        };
        assert_eq!(direct.shape(), &[1, 2]);
        let short_permuted = struct_ops::permute(array.clone(), &[1, 0]).unwrap();
        let Value::StructArray(permuted) =
            ipermute_builtin(Value::StructArray(short_permuted), Value::Tensor(short)).unwrap()
        else {
            panic!("expected structure array");
        };
        assert_eq!(permuted.shape(), &[2, 1]);
        assert_eq!(permuted.field_values("value"), array.field_values("value"));

        let padded = make_tensor(&[2.0, 1.0, 3.0, 4.0], &[1, 4]);
        let padded_permuted = struct_ops::permute(array.clone(), &[1, 0, 2, 3]).unwrap();
        let Value::StructArray(permuted) =
            ipermute_builtin(Value::StructArray(padded_permuted), Value::Tensor(padded)).unwrap()
        else {
            panic!("expected structure array");
        };
        assert_eq!(permuted.shape(), &[2, 1, 1, 1]);
        assert_eq!(permuted.field_values("value"), array.field_values("value"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ipermute_errors_when_order_too_short() {
        let matrix = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let order = make_tensor(&[1.0], &[1, 1]);
        let err = ipermute_builtin(Value::Tensor(matrix), Value::Tensor(order)).unwrap_err();
        assert!(
            err.to_string().contains("order length"),
            "expected rank error, got {err}"
        );
    }
}
