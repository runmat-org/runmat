//! MATLAB-compatible `isstring` builtin with GPU-aware semantics for RunMat.
//!
//! Determines whether a value is a MATLAB string array (including string scalars) without moving
//! data between host and device memory.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::introspection::type_resolvers::isstring_type;
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::introspection::isstring")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isstring",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Metadata-only predicate; gpuArray inputs stay on device while the result is returned on the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::introspection::isstring")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isstring",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Type-check predicate that does not participate in fusion planning.",
};

#[runtime_builtin(
    name = "isstring",
    category = "introspection",
    summary = "Return true when a value is a MATLAB string array.",
    keywords = "isstring,string array,string scalar,type checking,introspection",
    accel = "metadata",
    type_resolver(isstring_type),
    builtin_path = "crate::builtins::introspection::isstring"
)]
fn isstring_builtin(value: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Bool(matches!(
        value,
        Value::String(_) | Value::StringArray(_)
    )))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CellArray, CharArray, Closure, ComplexTensor, LogicalArray, MException, ObjectInstance,
        StringArray, StructValue, Tensor,
    };

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_scalar_reports_true() {
        let value = Value::String("RunMat".to_string());
        let result = isstring_builtin(value).expect("isstring");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_object_and_handle_variants_report_false() {
        let complex_scalar = Value::Complex(1.25, -3.5);
        let complex_tensor = Value::ComplexTensor(
            ComplexTensor::new(vec![(0.0, 1.0), (2.0, -2.5)], vec![2, 1]).expect("complex tensor"),
        );
        let object = Value::Object(ObjectInstance::new("ExampleClass".to_string()));
        let closure = Value::Closure(Closure {
            function_name: "some_func".to_string(),
            captures: Vec::new(),
        });
        let class_ref = Value::ClassRef("pkg.Type".to_string());
        let exception = Value::MException(MException::new(
            "RunMat:Test".to_string(),
            "example".to_string(),
        ));
        let function_handle = Value::FunctionHandle("sin".to_string());

        for candidate in vec![
            complex_scalar,
            complex_tensor,
            object,
            closure,
            class_ref,
            exception,
            function_handle,
        ] {
            assert_eq!(
                isstring_builtin(candidate).expect("isstring"),
                Value::Bool(false)
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_array_reports_true() {
        let array = StringArray::new(vec!["one".to_string(), "two".to_string()], vec![1, 2])
            .expect("string array");
        let result = isstring_builtin(Value::StringArray(array)).expect("isstring");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_string_array_reports_true() {
        let array = StringArray::new(vec![], vec![0, 0]).expect("empty string array");
        let result = isstring_builtin(Value::StringArray(array)).expect("isstring");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn character_arrays_report_false() {
        let chars = CharArray::new_row("RunMat");
        let result = isstring_builtin(Value::CharArray(chars)).expect("isstring");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_and_logical_values_report_false() {
        let tensor = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("tensor"));
        let logical = Value::Bool(true);
        let logical_array = Value::LogicalArray(
            LogicalArray::new(vec![1u8, 0u8], vec![2, 1]).expect("logical array"),
        );

        assert_eq!(
            isstring_builtin(tensor).expect("isstring"),
            Value::Bool(false)
        );
        assert_eq!(
            isstring_builtin(logical).expect("isstring"),
            Value::Bool(false)
        );
        assert_eq!(
            isstring_builtin(logical_array).expect("isstring"),
            Value::Bool(false)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_and_struct_values_report_false() {
        let empty_cell = CellArray::new(Vec::<Value>::new(), 0, 0).expect("cell");
        let empty_struct = StructValue::new();
        assert_eq!(
            isstring_builtin(Value::Cell(empty_cell)).expect("isstring"),
            Value::Bool(false)
        );
        assert_eq!(
            isstring_builtin(Value::Struct(empty_struct)).expect("isstring"),
            Value::Bool(false)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isstring_gpu_inputs_return_false() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = isstring_builtin(Value::GpuTensor(handle)).expect("isstring");
            assert_eq!(result, Value::Bool(false));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn isstring_wgpu_numeric_returns_false() {
        register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu provider");
        let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).expect("tensor");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let result = isstring_builtin(Value::GpuTensor(handle)).expect("isstring");
        assert_eq!(result, Value::Bool(false));
    }
}
