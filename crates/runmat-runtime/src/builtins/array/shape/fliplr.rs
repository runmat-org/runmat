//! MATLAB-compatible `fliplr` builtin with GPU-aware semantics for RunMat.

use crate::aggregate::structure::operations as struct_ops;

use crate::builtins::array::shape::flip::{
    complex_tensor_into_value, flip_char_array_with, flip_complex_tensor_with, flip_gpu_with,
    flip_logical_array_with, flip_string_array_with, flip_tensor_with,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::*;
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, Value};

const LR_DIM: [usize; 1] = [2];
const BUILTIN_NAME: &str = "fliplr";

fn fliplr_error(message: impl Into<String>) -> RuntimeError {
    fliplr_error_with_message(message, &FLIPLR_ERROR_INVALID_INPUT)
}

fn fliplr_error_descriptor(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    fliplr_error_with_message(error.message, error)
}

fn fliplr_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_fliplr_error(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(err.message().to_string())
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FLIPLR_ERROR_INVALID_INPUT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "fliplr",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::shape::fliplr"
)]
async fn fliplr_builtin(value: Value) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &FLIPLR_DESCRIPTOR,
        &FLIPLR_ERROR_INVALID_INPUT,
        BUILTIN_NAME,
    )?;
    match value {
        Value::Tensor(tensor) => Ok(flip_tensor_with("fliplr", tensor, &LR_DIM)
            .map(tensor::tensor_into_value)
            .map_err(remap_fliplr_error)?),
        Value::LogicalArray(array) => Ok(flip_logical_array_with("fliplr", array, &LR_DIM)
            .map(Value::LogicalArray)
            .map_err(remap_fliplr_error)?),
        Value::ComplexTensor(ct) => Ok(flip_complex_tensor_with("fliplr", ct, &LR_DIM)
            .map(Value::ComplexTensor)
            .map_err(remap_fliplr_error)?),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| fliplr_error(format!("fliplr: {e}")))?;
            Ok(flip_complex_tensor_with("fliplr", tensor, &LR_DIM)
                .map(complex_tensor_into_value)
                .map_err(remap_fliplr_error)?)
        }
        Value::StringArray(strings) => Ok(flip_string_array_with("fliplr", strings, &LR_DIM)
            .map(Value::StringArray)
            .map_err(remap_fliplr_error)?),
        Value::CharArray(chars) => Ok(flip_char_array_with("fliplr", chars, &LR_DIM)
            .map(Value::CharArray)
            .map_err(remap_fliplr_error)?),
        Value::String(scalar) => Ok(Value::String(scalar.into())),
        Value::Num(n) => {
            let tensor = tensor::value_into_tensor_for("fliplr", Value::Num(n))
                .map_err(|e| fliplr_error(e))?;
            Ok(flip_tensor_with("fliplr", tensor, &LR_DIM)
                .map(tensor::tensor_into_value)
                .map_err(remap_fliplr_error)?)
        }
        Value::Int(i) => {
            let tensor = tensor::value_into_tensor_for("fliplr", Value::Int(i))
                .map_err(|e| fliplr_error(e))?;
            Ok(flip_tensor_with("fliplr", tensor, &LR_DIM)
                .map(tensor::tensor_into_value)
                .map_err(remap_fliplr_error)?)
        }
        Value::Bool(flag) => {
            let tensor = tensor::value_into_tensor_for("fliplr", Value::Bool(flag))
                .map_err(|e| fliplr_error(e))?;
            Ok(flip_tensor_with("fliplr", tensor, &LR_DIM)
                .map(tensor::tensor_into_value)
                .map_err(remap_fliplr_error)?)
        }
        Value::GpuTensor(handle) => Ok(flip_gpu_with("fliplr", handle, &LR_DIM)
            .await
            .map_err(remap_fliplr_error)?),
        Value::Cell(_) => Err(fliplr_error_with_message(
            "fliplr: cell arrays are not yet supported",
            &FLIPLR_ERROR_UNSUPPORTED_INPUT,
        )),
        Value::Struct(structure) => Ok(Value::Struct(structure)),
        Value::StructArray(array) => struct_ops::flip(array, 1)
            .map(Value::StructArray)
            .map_err(|error| struct_ops::remap(error, |message| fliplr_error(message))),
        Value::ScopedFunctionHandle(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_)
        | Value::SparseTensor(_)
        | Value::ObjectArray(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_)
        | Value::Foreign(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_)
        | Value::OutputList(_) => Err(fliplr_error_descriptor(&FLIPLR_ERROR_UNSUPPORTED_INPUT)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn fliplr_builtin(value: Value) -> crate::BuiltinResult<Value> {
        block_on(super::fliplr_builtin(value))
    }
    use crate::builtins::array::shape::flip::{flip_logical_array, flip_tensor};
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{
        CharArray, IntegerComplexStorage, IntegerStorage, LogicalArray, StringArray, StructArray,
        Tensor, Value,
    };

    #[test]
    fn fliplr_integer_contract_preserves_storage() {
        assert_eq!(FLIPLR_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
        assert_eq!(
            FLIPLR_INTEGER_CAPABILITIES[0].output_class,
            BuiltinIntegerOutputClassRule::PreserveInput
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_matrix_reverses_columns() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2]).expect("tensor");
        let expected = flip_tensor(tensor.clone(), &LR_DIM).expect("expected");
        let result = fliplr_builtin(Value::Tensor(tensor)).expect("fliplr");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), expected.shape());
                assert_eq!(out.materialize_f64(), expected.materialize_f64());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_row_vector_reverses_order() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let expected = flip_tensor(tensor.clone(), &LR_DIM).expect("expected");
        let result = fliplr_builtin(Value::Tensor(tensor)).expect("fliplr");
        match result {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), expected.materialize_f64()),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_column_vector_noop() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let expected = tensor.clone();
        let result = fliplr_builtin(Value::Tensor(tensor)).expect("fliplr");
        match result {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), expected.materialize_f64()),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_nd_tensor_flips_second_dim_only() {
        let tensor = Tensor::new((1..=24).map(|v| v as f64).collect(), vec![3, 4, 2]).unwrap();
        let expected = flip_tensor(tensor.clone(), &LR_DIM).expect("expected");
        let result = fliplr_builtin(Value::Tensor(tensor)).expect("fliplr");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), expected.shape());
                assert_eq!(out.materialize_f64(), expected.materialize_f64());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_char_array() {
        let chars = CharArray::from_unicode_scalars("runmat".chars().collect(), 2, 3).unwrap();
        let result = fliplr_builtin(Value::CharArray(chars)).expect("fliplr");
        match result {
            Value::CharArray(out) => {
                let collected: String = runmat_value::decode_utf16_code_units(out.data())
                    .expect("valid UTF-16 fixture text");
                assert_eq!(collected, "nurtam");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_string_array() {
        let strings =
            StringArray::new(vec!["left".into(), "right".into()], vec![1, 2]).expect("strings");
        let result = fliplr_builtin(Value::StringArray(strings)).expect("fliplr");
        match result {
            Value::StringArray(out) => assert_eq!(out.data(), vec!["right", "left"]),
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_logical_array_preserves_bits() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let expected = flip_logical_array(logical.clone(), &LR_DIM).expect("expected");
        let result = fliplr_builtin(Value::LogicalArray(logical)).expect("fliplr");
        match result {
            Value::LogicalArray(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_preserves_exact_integer_storage_without_f64_mirror() {
        let exact = IntegerStorage::U64(vec![u64::MAX, 0, 9_007_199_254_740_993, 7]);
        let tensor = Tensor::new_integer(exact, vec![2, 2]).expect("integer tensor");

        let result = fliplr_builtin(Value::Tensor(tensor)).expect("fliplr");

        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), vec![2, 2]);
                assert_eq!(
                    out.integer_storage(),
                    Some(&IntegerStorage::U64(vec![
                        9_007_199_254_740_993,
                        7,
                        u64::MAX,
                        0,
                    ]))
                );
            }
            other => panic!("expected typed integer tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_preserves_typed_complex_integer_storage_without_f64_mirror() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::I16(vec![1, 2, 3, 4]),
            IntegerStorage::I16(vec![5, 6, 7, 8]),
        )
        .expect("complex integer storage");
        let tensor =
            ComplexTensor::new_integer(storage, vec![2, 2]).expect("complex integer tensor");

        let result = fliplr_builtin(Value::ComplexTensor(tensor)).expect("fliplr");

        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape(), vec![2, 2]);
                let storage = out.integer_storage().expect("typed complex integer output");
                assert_eq!(storage.real(), &IntegerStorage::I16(vec![3, 4, 1, 2]));
                assert_eq!(storage.imag(), &IntegerStorage::I16(vec![7, 8, 5, 6]));
            }
            other => panic!("expected typed complex integer tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_scalar_numeric_noop() {
        let result = fliplr_builtin(Value::Num(42.0)).expect("fliplr");
        match result {
            Value::Num(v) => assert_eq!(v, 42.0),
            other => panic!("expected numeric scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_string_scalar_noop() {
        let result = fliplr_builtin(Value::String("runmat".into())).expect("fliplr");
        match result {
            Value::String(s) => assert_eq!(s, "runmat"),
            other => panic!("expected string scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new((1..=12).map(|v| v as f64).collect(), vec![3, 4]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = fliplr_builtin(Value::GpuTensor(handle)).expect("fliplr gpu");
            let gathered = test_support::gather(result).expect("gather");
            let expected = flip_tensor(tensor, &LR_DIM).expect("expected");
            assert_eq!(gathered.shape(), expected.shape());
            assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_preserves_scalar_and_reorders_struct_arrays() {
        let mut first = runmat_value::UnboundedStructValueBuilder::new();
        first.insert("value", Value::Num(1.0));
        let first = first.finish();
        assert_eq!(
            fliplr_builtin(Value::Struct(first.clone())).unwrap(),
            Value::Struct(first.clone())
        );
        let mut second = runmat_value::UnboundedStructValueBuilder::new();
        second.insert("value", Value::Num(2.0));
        let second = second.finish();
        let array = StructArray::new(vec![first, second], vec![1, 2]).unwrap();
        let Value::StructArray(output) = fliplr_builtin(Value::StructArray(array)).unwrap() else {
            panic!("expected structure array");
        };
        assert_eq!(
            output.get_linear(0).unwrap().fields["value"],
            Value::Num(2.0)
        );
        assert_eq!(
            output.get_linear(1).unwrap().fields["value"],
            Value::Num(1.0)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fliplr_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new((1..=16).map(|v| v as f64).collect(), vec![4, 4]).unwrap();
        let expected = flip_tensor(tensor.clone(), &LR_DIM).expect("expected");
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&view).expect("upload");
        let value = fliplr_builtin(Value::GpuTensor(handle)).expect("fliplr gpu");
        let gathered = test_support::gather(value).expect("gather");
        assert_eq!(gathered.shape(), expected.shape());
        assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
    }
}
