//! Runtime binding for canonical character-array predicate.
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "ischar",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::ischar"
)]
fn ischar_builtin(value: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Bool(matches!(value, Value::CharArray(_))))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{CellArray, CharArray, LogicalArray, StringArray, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn character_vector_reports_true() {
        let chars = CharArray::new_row("RunMat");
        let result = ischar_builtin(Value::CharArray(chars)).expect("ischar");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn character_matrix_reports_true() {
        let chars = CharArray::from_unicode_scalars(vec!['a', 'b', 'c', 'd', 'e', 'f'], 2, 3)
            .expect("char array");
        let result = ischar_builtin(Value::CharArray(chars)).expect("ischar");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_values_report_false() {
        let scalar = Value::String("RunMat".to_string().into());
        let array = Value::StringArray(
            StringArray::new(vec!["a".to_string(), "b".to_string()], vec![1, 2]).expect("strings"),
        );
        assert_eq!(ischar_builtin(scalar).expect("ischar"), Value::Bool(false));
        assert_eq!(ischar_builtin(array).expect("ischar"), Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_and_logical_values_report_false() {
        let numeric = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).expect("tensor"));
        let logical = Value::Bool(true);
        let logical_array =
            Value::LogicalArray(LogicalArray::new(vec![1u8], vec![1, 1]).expect("logical array"));
        assert_eq!(ischar_builtin(numeric).expect("ischar"), Value::Bool(false));
        assert_eq!(ischar_builtin(logical).expect("ischar"), Value::Bool(false));
        assert_eq!(
            ischar_builtin(logical_array).expect("ischar"),
            Value::Bool(false)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_array_reports_false() {
        let cell = CellArray::new(vec![Value::Num(1.0), Value::from("text")], 1, 2).expect("cell");
        let result = ischar_builtin(Value::Cell(cell)).expect("ischar");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_value_reports_false() {
        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        st.insert("field", Value::Num(1.0));
        let result = ischar_builtin(Value::Struct(st.finish())).expect("ischar");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn function_handle_reports_false() {
        let fh = Value::FunctionHandle("sin".into());
        let result = ischar_builtin(fh).expect("ischar");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ischar_gpu_inputs_return_false() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let value = Value::GpuTensor(handle);
            let result = ischar_builtin(value).expect("ischar");
            assert_eq!(result, Value::Bool(false));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ischar_wgpu_numeric_returns_false() {
        let _ = register_wgpu_provider(WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let data = vec![0.0, 1.0];
        let shape = vec![2, 1];
        let view = HostTensorView {
            data: &data,
            shape: &shape,
        };
        let handle = provider.upload(&view).expect("upload to GPU");
        let result = ischar_builtin(Value::GpuTensor(handle)).expect("ischar");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_character_array_reports_true() {
        let chars = CharArray::from_code_units(Vec::new(), 0, 0).expect("empty char array");
        let result = ischar_builtin(Value::CharArray(chars)).expect("ischar");
        assert_eq!(result, Value::Bool(true));
    }
}
