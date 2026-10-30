//! MATLAB-compatible `flipud` builtin with GPU-aware semantics for RunMat.

use crate::aggregate::structure::operations as struct_ops;

use crate::builtins::array::shape::flip::{
    complex_tensor_into_value, flip_char_array_with, flip_complex_tensor_with, flip_gpu_with,
    flip_logical_array_with, flip_string_array_with, flip_tensor_with,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::*;
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, ComplexTensor, Value};

const UD_DIM: [usize; 1] = [1];
const BUILTIN_NAME: &str = "flipud";

fn flipud_error(message: impl Into<String>) -> RuntimeError {
    flipud_error_with_message(message, &FLIPUD_ERROR_INVALID_INPUT)
}

fn flipud_error_descriptor(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    flipud_error_with_message(error.message, error)
}

fn flipud_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_flipud_error(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(err.message().to_string())
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FLIPUD_ERROR_INVALID_INPUT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "flipud",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::shape::flipud"
)]
async fn flipud_builtin(value: Value) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &FLIPUD_DESCRIPTOR,
        &FLIPUD_ERROR_INVALID_INPUT,
        BUILTIN_NAME,
    )?;
    match value {
        Value::Tensor(tensor) => Ok(flip_tensor_with("flipud", tensor, &UD_DIM)
            .map(tensor::tensor_into_value)
            .map_err(remap_flipud_error)?),
        Value::LogicalArray(array) => Ok(flip_logical_array_with("flipud", array, &UD_DIM)
            .map(Value::LogicalArray)
            .map_err(remap_flipud_error)?),
        Value::ComplexTensor(ct) => Ok(flip_complex_tensor_with("flipud", ct, &UD_DIM)
            .map(Value::ComplexTensor)
            .map_err(remap_flipud_error)?),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| flipud_error(format!("flipud: {e}")))?;
            Ok(flip_complex_tensor_with("flipud", tensor, &UD_DIM)
                .map(complex_tensor_into_value)
                .map_err(remap_flipud_error)?)
        }
        Value::StringArray(strings) => Ok(flip_string_array_with("flipud", strings, &UD_DIM)
            .map(Value::StringArray)
            .map_err(remap_flipud_error)?),
        Value::CharArray(chars) => Ok(flip_char_array_with("flipud", chars, &UD_DIM)
            .map(Value::CharArray)
            .map_err(remap_flipud_error)?),
        Value::String(scalar) => Ok(Value::String(scalar.into())),
        Value::Cell(cell) => flip_cell_array_rows(cell).map_err(remap_flipud_error),
        Value::Struct(structure) => Ok(Value::Struct(structure)),
        Value::StructArray(array) => {
            struct_ops::flip(array, 0)
                .map(Value::StructArray)
                .map_err(|error| {
                    struct_ops::remap(error, |message| flipud_error(format!("flipud: {message}")))
                })
        }
        Value::Num(n) => {
            let tensor = tensor::value_into_tensor_for("flipud", Value::Num(n))
                .map_err(|e| flipud_error(e))?;
            Ok(flip_tensor_with("flipud", tensor, &UD_DIM)
                .map(tensor::tensor_into_value)
                .map_err(remap_flipud_error)?)
        }
        Value::Int(i) => {
            let tensor = tensor::value_into_tensor_for("flipud", Value::Int(i))
                .map_err(|e| flipud_error(e))?;
            Ok(flip_tensor_with("flipud", tensor, &UD_DIM)
                .map(tensor::tensor_into_value)
                .map_err(remap_flipud_error)?)
        }
        Value::Bool(flag) => {
            let tensor = tensor::value_into_tensor_for("flipud", Value::Bool(flag))
                .map_err(|e| flipud_error(e))?;
            Ok(flip_tensor_with("flipud", tensor, &UD_DIM)
                .map(tensor::tensor_into_value)
                .map_err(remap_flipud_error)?)
        }
        Value::GpuTensor(handle) => Ok(flip_gpu_with("flipud", handle, &UD_DIM)
            .await
            .map_err(remap_flipud_error)?),
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
        | Value::OutputList(_) => Err(flipud_error_descriptor(&FLIPUD_ERROR_UNSUPPORTED_INPUT)),
    }
}

fn flip_cell_array_rows(cell: CellArray) -> crate::BuiltinResult<Value> {
    if cell.rows <= 1 || cell.data().is_empty() {
        return Ok(Value::Cell(cell));
    }
    let rows = cell.rows;
    let cols = cell.cols;
    let data = cell.data();
    let mut flipped = Vec::with_capacity(data.len());
    for row in (0..rows).rev() {
        let base = row * cols;
        for col in 0..cols {
            flipped.push(data[base + col].clone());
        }
    }
    CellArray::new(flipped, rows, cols)
        .map(Value::Cell)
        .map_err(|e| flipud_error(format!("flipud: {e}")))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn flipud_builtin(value: Value) -> crate::BuiltinResult<Value> {
        block_on(super::flipud_builtin(value))
    }
    use crate::builtins::array::shape::flip::{
        flip_complex_tensor, flip_logical_array, flip_tensor,
    };
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{
        CellArray, CharArray, IntegerComplexStorage, IntegerStorage, LogicalArray, StringArray,
        Tensor, Value,
    };

    #[test]
    fn flipud_integer_contract_preserves_storage() {
        assert_eq!(FLIPUD_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
        assert_eq!(
            FLIPUD_INTEGER_CAPABILITIES[0].output_class,
            BuiltinIntegerOutputClassRule::PreserveInput
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_matrix_reverses_rows() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).expect("tensor");
        let expected = flip_tensor(tensor.clone(), &UD_DIM).expect("expected");
        let result = flipud_builtin(Value::Tensor(tensor)).expect("flipud");
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
    fn flipud_column_vector_reverses_order() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let expected = flip_tensor(tensor.clone(), &UD_DIM).expect("expected");
        let result = flipud_builtin(Value::Tensor(tensor)).expect("flipud");
        match result {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), expected.materialize_f64()),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_row_vector_noop() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let expected = tensor.clone();
        let result = flipud_builtin(Value::Tensor(tensor)).expect("flipud");
        match result {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), expected.materialize_f64()),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_nd_tensor_flips_first_dim_only() {
        let tensor = Tensor::new((1..=24).map(|v| v as f64).collect(), vec![3, 4, 2]).unwrap();
        let expected = flip_tensor(tensor.clone(), &UD_DIM).expect("expected");
        let result = flipud_builtin(Value::Tensor(tensor)).expect("flipud");
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
    fn flipud_char_array() {
        let chars = CharArray::from_unicode_scalars("runmat".chars().collect(), 2, 3).unwrap();
        let result = flipud_builtin(Value::CharArray(chars)).expect("flipud");
        match result {
            Value::CharArray(out) => {
                let collected: String = runmat_value::decode_utf16_code_units(out.data())
                    .expect("valid UTF-16 fixture text");
                assert_eq!(collected, "matrun");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_string_array() {
        let strings =
            StringArray::new(vec!["top".into(), "bottom".into()], vec![2, 1]).expect("strings");
        let result = flipud_builtin(Value::StringArray(strings)).expect("flipud");
        match result {
            Value::StringArray(out) => assert_eq!(out.data(), vec!["bottom", "top"]),
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_cell_array_reverses_rows() {
        let cell = CellArray::new(
            vec![
                Value::from("r1c1"),
                Value::from("r1c2"),
                Value::from("r2c1"),
                Value::from("r2c2"),
            ],
            2,
            2,
        )
        .expect("cell");
        let result = flipud_builtin(Value::Cell(cell)).expect("flipud");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 2);
                assert_eq!(out.get(0, 0).unwrap(), Value::from("r2c1"));
                assert_eq!(out.get(0, 1).unwrap(), Value::from("r2c2"));
                assert_eq!(out.get(1, 0).unwrap(), Value::from("r1c1"));
                assert_eq!(out.get(1, 1).unwrap(), Value::from("r1c2"));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_logical_array_preserves_bits() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let expected = flip_logical_array(logical.clone(), &UD_DIM).expect("expected");
        let result = flipud_builtin(Value::LogicalArray(logical)).expect("flipud");
        match result {
            Value::LogicalArray(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_preserves_exact_integer_storage_without_f64_mirror() {
        let exact = IntegerStorage::U64(vec![u64::MAX, 0, 9_007_199_254_740_993, 7]);
        let tensor = Tensor::new_integer(exact, vec![2, 2]).expect("integer tensor");

        let result = flipud_builtin(Value::Tensor(tensor)).expect("flipud");

        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), vec![2, 2]);
                assert_eq!(
                    out.integer_storage(),
                    Some(&IntegerStorage::U64(vec![
                        0,
                        u64::MAX,
                        7,
                        9_007_199_254_740_993,
                    ]))
                );
            }
            other => panic!("expected typed integer tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_preserves_typed_complex_integer_storage_without_f64_mirror() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::I16(vec![1, 2, 3, 4]),
            IntegerStorage::I16(vec![5, 6, 7, 8]),
        )
        .expect("complex integer storage");
        let tensor =
            ComplexTensor::new_integer(storage, vec![2, 2]).expect("complex integer tensor");

        let result = flipud_builtin(Value::ComplexTensor(tensor)).expect("flipud");

        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape(), vec![2, 2]);
                let storage = out.integer_storage().expect("typed complex integer output");
                assert_eq!(storage.real(), &IntegerStorage::I16(vec![2, 1, 4, 3]));
                assert_eq!(storage.imag(), &IntegerStorage::I16(vec![6, 5, 8, 7]));
            }
            other => panic!("expected typed complex integer tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_scalar_numeric_noop() {
        let result = flipud_builtin(Value::Num(42.0)).expect("flipud");
        match result {
            Value::Num(v) => assert_eq!(v, 42.0),
            other => panic!("expected numeric scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_complex_tensor_defaults_to_first_dim() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 1.0), (2.0, -1.0), (3.0, 0.5), (4.0, -0.25)],
            vec![2, 2],
        )
        .unwrap();
        let expected = flip_complex_tensor(tensor.clone(), &UD_DIM).expect("expected");
        let result = flipud_builtin(Value::ComplexTensor(tensor)).expect("flipud");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.materialize_f64(), expected.materialize_f64())
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_preserves_scalar_structure() {
        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        st.fields.insert("field".into(), Value::Num(1.0));
        let st = st.finish();
        let result = flipud_builtin(Value::Struct(st.clone())).expect("scalar structure");
        assert_eq!(result, Value::Struct(st));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = flipud_builtin(Value::GpuTensor(handle)).expect("flipud");
            let gathered = test_support::gather(result).expect("gather");
            let expected = flip_tensor(tensor, &UD_DIM).expect("expected");
            assert_eq!(gathered.shape(), expected.shape());
            assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_gpu_preserves_row_vector() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = flipud_builtin(Value::GpuTensor(handle)).expect("flipud");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), tensor.shape());
            assert_eq!(gathered.materialize_f64(), tensor.materialize_f64());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_gpu_falls_back_when_axis_missing() {
        // The simple provider does not expose flip, so this exercises gather→flip→upload.
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let result = flipud_builtin(Value::Tensor(tensor.clone())).expect("flipud");
        match result {
            Value::Tensor(out) => assert_eq!(
                out.materialize_f64(),
                flip_tensor(tensor, &UD_DIM).unwrap().materialize_f64()
            ),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_gpu_with_registered_provider_preserves_gpu_type() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = flipud_builtin(Value::GpuTensor(handle)).expect("flipud");
            assert!(matches!(result, Value::GpuTensor(_)));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn flipud_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let cpu = flip_tensor(tensor.clone(), &UD_DIM).expect("cpu flip");
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = flipud_builtin(Value::GpuTensor(handle)).expect("flipud");
        let gathered = test_support::gather(gpu).expect("gather");
        assert_eq!(gathered.shape(), cpu.shape());
        assert_eq!(gathered.materialize_f64(), cpu.materialize_f64());
    }
}
