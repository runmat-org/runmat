//! MATLAB-compatible `squeeze` builtin with GPU-aware semantics for RunMat.

use crate::aggregate::structure::operations as struct_ops;

use crate::{build_runtime_error, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
pub use runmat_builtins::SQUEEZE_DESCRIPTOR;
use runmat_builtins::*;
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};

fn squeeze_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("squeeze").build()
}

#[runtime_builtin(
    name = "squeeze",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::shape::squeeze"
)]
async fn squeeze_builtin(value: Value) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &SQUEEZE_DESCRIPTOR,
        &SQUEEZE_ERROR_INVALID_INPUT,
        "squeeze",
    )?;

    squeeze_value(value).await
}

async fn squeeze_value(value: Value) -> crate::BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => squeeze_numeric_tensor(tensor).map(Value::Tensor),
        Value::ComplexTensor(ct) => squeeze_complex_tensor(ct).map(Value::ComplexTensor),
        Value::LogicalArray(logical) => squeeze_logical_array(logical).map(Value::LogicalArray),
        Value::StringArray(strings) => squeeze_string_array(strings).map(Value::StringArray),
        Value::CharArray(chars) => squeeze_char_array(chars).map(Value::CharArray),
        Value::Cell(cell) => squeeze_cell_array(cell).map(Value::Cell),
        Value::StructArray(array) => {
            let shape = squeeze_shape(array.shape());
            struct_ops::reshape(array,shape).map(Value::StructArray).map_err(|error|struct_ops::remap(error,|message|squeeze_error(message)))
        }
        Value::GpuTensor(handle) => squeeze_gpu(handle).await,
        Value::String(_) | Value::Struct(_) => Ok(value),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(value),
        other => Err(super::error::declared(squeeze_error(format!(
            "squeeze: unsupported input type {}; expected numeric, logical, string, char, cell, or gpu array",
            value_kind(&other)
        )), &SQUEEZE_ERROR_INVALID_INPUT)),
    }
}

fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Tensor(_) => "tensor",
        Value::SparseTensor(_) => "sparse tensor",
        Value::ComplexTensor(_) => "complex tensor",
        Value::LogicalArray(_) => "logical array",
        Value::StringArray(_) => "string array",
        Value::CharArray(_) => "char array",
        Value::Cell(_) => "cell array",
        Value::GpuTensor(_) => "gpu array",
        Value::Num(_) => "double scalar",
        Value::Int(_) => "integer scalar",
        Value::Bool(_) => "logical scalar",
        Value::Complex(_, _) => "complex scalar",
        Value::String(_) => "string scalar",
        Value::Symbolic(_) => "symbolic scalar",
        Value::SymbolicArray(_) => "symbolic array",
        Value::ObjectArray(_) | Value::Object(_) => "object",
        Value::HandleObject(_) => "handle object",
        Value::Listener(_) => "listener",
        Value::Struct(_) | Value::StructArray(_) => "struct",
        Value::ScopedFunctionHandle(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_) => "function handle",
        Value::ClassRef(_) => "class reference",
        Value::MException(_) => "exception",
        Value::OutputList(_) => "output list",
        Value::Future(_) => "future",
        Value::Task(_) => "task",
        Value::Pool(_) => "pool",
        Value::Job(_) => "job",
        Value::Distributed(_) => "distributed",
        Value::Composite(_) => "Composite",
        Value::Foreign(_) => "foreign reference",
    }
}

fn squeeze_numeric_tensor(tensor: Tensor) -> crate::BuiltinResult<Tensor> {
    let shape = squeeze_shape(tensor.shape());
    if shape == tensor.shape() {
        return Ok(tensor);
    }
    tensor
        .reshape(shape)
        .map_err(|e| squeeze_error(format!("squeeze: {e}")))
}

fn squeeze_complex_tensor(ct: ComplexTensor) -> crate::BuiltinResult<ComplexTensor> {
    let shape = squeeze_shape(ct.shape());
    if shape == ct.shape() {
        return Ok(ct);
    }
    ComplexTensor::from_complex_storage(ct.into_complex_storage(), shape)
        .map_err(|e| squeeze_error(format!("squeeze: {e}")))
}

fn squeeze_logical_array(logical: LogicalArray) -> crate::BuiltinResult<LogicalArray> {
    let shape = squeeze_shape(&logical.shape);
    if shape == logical.shape {
        return Ok(logical);
    }
    LogicalArray::from_host_buffer(logical.data, shape)
        .map_err(|e| squeeze_error(format!("squeeze: {e}")))
}

fn squeeze_string_array(strings: StringArray) -> crate::BuiltinResult<StringArray> {
    let shape = squeeze_shape(&strings.shape);
    if shape == strings.shape {
        return Ok(strings);
    }
    StringArray::from_shared_storage(strings.into_parts().0, shape.into())
        .map_err(|e| squeeze_error(format!("squeeze: {e}")))
}

fn squeeze_char_array(chars: CharArray) -> crate::BuiltinResult<CharArray> {
    let shape = squeeze_shape(&chars.shape);
    if shape == chars.shape {
        return Ok(chars);
    }
    CharArray::from_column_major(chars.to_column_major(), shape)
        .map_err(|e| squeeze_error(format!("squeeze: {e}")))
}

fn squeeze_cell_array(
    cell: runmat_value::CellArray,
) -> crate::BuiltinResult<runmat_value::CellArray> {
    let shape = squeeze_shape(&cell.shape);
    if shape == cell.shape {
        return Ok(cell);
    }
    runmat_value::CellArray::from_column_major(cell.to_column_major(), shape)
        .map_err(|e| squeeze_error(format!("squeeze: {e}")))
}

async fn squeeze_gpu(handle: GpuTensorHandle) -> crate::BuiltinResult<Value> {
    let squeezed = squeeze_shape(handle.shape());
    super::gpu_reshape::reshape(handle, &squeezed, "squeeze").map(Value::GpuTensor)
}

fn squeeze_shape(shape: &[usize]) -> Vec<usize> {
    if shape.len() <= 2 {
        return shape.to_vec();
    }
    let mut squeezed: Vec<usize> = shape.iter().copied().filter(|&d| d != 1).collect();
    if squeezed.is_empty() {
        squeezed.push(1);
        squeezed.push(1);
    } else if squeezed.len() == 1 {
        let first = squeezed[0];
        squeezed = vec![first, 1];
    }
    squeezed
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn squeeze_builtin(value: Value) -> crate::BuiltinResult<Value> {
        block_on(super::squeeze_builtin(value))
    }
    use crate::builtins::common::test_support;
    use runmat_value::{IntValue, IntegerComplexStorage, IntegerStorage, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn squeeze_removes_middle_singletons() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]).unwrap();
        let result = squeeze_builtin(Value::Tensor(tensor)).expect("squeeze ok");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape(), &[2, 2]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn squeeze_preserves_exact_integer_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![0, u64::MAX]), vec![1, 2, 1])
            .expect("integer tensor");
        let result = squeeze_builtin(Value::Tensor(tensor)).expect("squeeze ok");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[2, 1]);
                assert_eq!(
                    tensor.integer_storage(),
                    Some(&IntegerStorage::U64(vec![0, u64::MAX]))
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn squeeze_preserves_exact_typed_complex_integer_storage() {
        let complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                IntegerStorage::I64(vec![7, -8]),
            )
            .expect("storage"),
            vec![1, 2, 1],
        )
        .expect("complex");
        let value = squeeze_builtin(Value::ComplexTensor(complex)).expect("squeeze");

        let Value::ComplexTensor(output) = value else {
            panic!("expected complex tensor");
        };
        assert_eq!(output.shape(), &[2, 1]);
        assert_eq!(
            output
                .integer_storage()
                .as_ref()
                .map(|storage| (storage.real(), storage.imag())),
            Some((
                &IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                &IntegerStorage::I64(vec![7, -8]),
            ))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn squeeze_preserves_row_vector() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let result = squeeze_builtin(Value::Tensor(tensor)).expect("squeeze ok");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape(), &[1, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn squeeze_single_dimension_becomes_column_vector() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4, 1]).unwrap();
        let result = squeeze_builtin(Value::Tensor(tensor)).expect("squeeze ok");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape(), &[4, 1]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn squeeze_on_logical_array_respects_zero_dims() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![1, 4, 1]).unwrap();
        let result = squeeze_builtin(Value::LogicalArray(logical)).expect("squeeze ok");
        match result {
            Value::LogicalArray(arr) => assert_eq!(arr.shape, vec![4, 1]),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn squeeze_on_string_array() {
        let strings = StringArray::new(vec!["a".into(), "b".into()], vec![1, 1, 2]).unwrap();
        let result = squeeze_builtin(Value::StringArray(strings)).expect("squeeze ok");
        match result {
            Value::StringArray(sa) => assert_eq!(sa.shape, vec![2, 1]),
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn squeeze_preserves_zero_length_dimensions() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![1, 0, 3]).unwrap();
        let result = squeeze_builtin(Value::Tensor(tensor)).expect("squeeze ok");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape(), &[0, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn squeeze_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let value = squeeze_builtin(Value::GpuTensor(handle)).expect("squeeze ok");
            let gathered = test_support::gather(value).expect("gather");
            assert_eq!(gathered.shape(), &[2, 2]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn squeeze_scalar_inputs_passthrough() {
        let result = squeeze_builtin(Value::Int(IntValue::I32(42))).expect("squeeze ok for scalar");
        assert_eq!(result, Value::Int(IntValue::I32(42)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn squeeze_wgpu_updates_shape_metadata() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let tensor = Tensor::new(
            (0..12).map(|v| v as f64).collect::<Vec<_>>(),
            vec![1, 3, 4, 1],
        )
        .unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload source tensor");

        let squeezed =
            squeeze_builtin(Value::GpuTensor(handle.clone())).expect("squeeze gpu tensor");
        let gpu_handle = match squeezed {
            Value::GpuTensor(h) => h,
            other => panic!("expected gpu tensor, got {other:?}"),
        };
        assert_eq!(gpu_handle.shape(), vec![3, 4]);

        let downloaded = test_support::gather(Value::GpuTensor(gpu_handle.clone()))
            .expect("gather squeezed tensor");
        assert_eq!(downloaded.shape(), &[3, 4]);
        assert_eq!(
            downloaded.materialize_f64().as_slice(),
            tensor.materialize_f64().as_slice()
        );
        drop(gpu_handle);
    }
}
