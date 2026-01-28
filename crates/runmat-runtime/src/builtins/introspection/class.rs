//! MATLAB-compatible `class` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::introspection::class")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "class",
    op_kind: GpuOpKind::Custom("introspection"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Introspection-only builtin; providers do not need to implement hooks. RunMat reads residency metadata and returns a host string.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::introspection::class")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "class",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; class executes on the host and returns a string scalar.",
};

#[runtime_builtin(
    name = "class",
    category = "introspection",
    summary = "Return the MATLAB class name for scalars, arrays, and objects.",
    keywords = "class,type inspection,type name,gpuArray class",
    builtin_path = "crate::builtins::introspection::class"
)]
fn class_builtin(value: Value) -> crate::BuiltinResult<String> {
    Ok(class_name_for_value(&value))
}

/// Return the canonical MATLAB class name for a runtime value.
pub(crate) fn class_name_for_value(value: &Value) -> String {
    match value {
        Value::Num(_) | Value::Tensor(_) | Value::ComplexTensor(_) | Value::Complex(_, _) => {
            "double".to_string()
        }
        Value::Int(iv) => iv.class_name().to_string(),
        Value::Bool(_) | Value::LogicalArray(_) => "logical".to_string(),
        Value::String(_) | Value::StringArray(_) => "string".to_string(),
        Value::CharArray(_) => "char".to_string(),
        Value::Cell(_) => "cell".to_string(),
        Value::Struct(_) => "struct".to_string(),
        Value::GpuTensor(_) => "gpuArray".to_string(),
        Value::FunctionHandle(_) | Value::Closure(_) => "function_handle".to_string(),
        Value::HandleObject(handle) => {
            if handle.class_name.is_empty() {
                "handle".to_string()
            } else {
                handle.class_name.clone()
            }
        }
        Value::Listener(_) => "event.listener".to_string(),
        Value::Object(obj) => obj.class_name.clone(),
        Value::ClassRef(_) => "meta.class".to_string(),
        Value::MException(_) => "MException".to_string(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CellArray, CharArray, Closure, ComplexTensor, HandleRef, IntValue, Listener, LogicalArray,
        MException, ObjectInstance, StringArray, StructValue, Tensor,
    };
    use runmat_gc_api::GcPtr;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_double_for_numeric_scalars() {
        let name = class_builtin(Value::Num(std::f64::consts::PI)).expect("class");
        assert_eq!(name, "double");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_integer_type_names() {
        let name = class_builtin(Value::Int(IntValue::I32(12))).expect("class");
        assert_eq!(name, "int32");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_expected_names_for_core_types() {
        let logical_scalar = Value::Bool(true);
        assert_eq!(class_name_for_value(&logical_scalar), "logical");

        let logical_array = Value::LogicalArray(
            LogicalArray::new(vec![1u8, 0u8, 1u8, 1u8], vec![2, 2]).expect("logical array"),
        );
        assert_eq!(class_name_for_value(&logical_array), "logical");

        let string_scalar = Value::String("hello".to_string());
        assert_eq!(class_name_for_value(&string_scalar), "string");

        let string_array = Value::StringArray(
            StringArray::new(vec!["Ada".into(), "Grace".into()], vec![1, 2]).expect("string array"),
        );
        assert_eq!(class_name_for_value(&string_array), "string");

        let char_array = Value::CharArray(CharArray::new_row("abc"));
        assert_eq!(class_name_for_value(&char_array), "char");

        let real_tensor = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
        assert_eq!(class_name_for_value(&real_tensor), "double");

        let complex_scalar = Value::Complex(1.0, -1.0);
        assert_eq!(class_name_for_value(&complex_scalar), "double");

        let complex_tensor = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 1.0), (2.0, -3.0)], vec![2, 1]).expect("complex tensor"),
        );
        assert_eq!(class_name_for_value(&complex_tensor), "double");

        let cell =
            Value::Cell(CellArray::new(vec![Value::Num(1.0), Value::Bool(false)], 1, 2).unwrap());
        assert_eq!(class_name_for_value(&cell), "cell");

        let mut st = StructValue::new();
        st.fields.insert("field".into(), Value::Num(42.0));
        let struct_value = Value::Struct(st);
        assert_eq!(class_name_for_value(&struct_value), "struct");

        let func_handle = Value::FunctionHandle("sin".into());
        assert_eq!(class_name_for_value(&func_handle), "function_handle");

        let closure = Value::Closure(Closure {
            function_name: "anon".into(),
            captures: vec![],
        });
        assert_eq!(class_name_for_value(&closure), "function_handle");

        let class_ref = Value::ClassRef("pkg.Point".into());
        assert_eq!(class_name_for_value(&class_ref), "meta.class");

        let exception = Value::MException(MException::new("id:err".into(), "fail".into()));
        assert_eq!(class_name_for_value(&exception), "MException");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_gpuarray_without_gather() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let name = class_builtin(Value::GpuTensor(handle)).expect("class");
            assert_eq!(name, "gpuArray");
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_handle_class_names() {
        let fallback = HandleRef {
            class_name: String::new(),
            target: GcPtr::null(),
            valid: false,
        };
        let fallback_name = class_builtin(Value::HandleObject(fallback)).expect("class");
        assert_eq!(fallback_name, "handle");

        let handle = HandleRef {
            class_name: "MockHandle".into(),
            target: GcPtr::null(),
            valid: true,
        };
        let name = class_builtin(Value::HandleObject(handle)).expect("class");
        assert_eq!(name, "MockHandle");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_object_and_listener_classes() {
        let object = ObjectInstance::new("pkg.Point".into());
        let obj_name = class_builtin(Value::Object(object)).expect("class object");
        assert_eq!(obj_name, "pkg.Point");

        let listener = Listener {
            id: 1,
            target: GcPtr::null(),
            event_name: "changed".into(),
            callback: GcPtr::null(),
            enabled: true,
            valid: true,
        };
        let listener_name = class_builtin(Value::Listener(listener)).expect("class listener");
        assert_eq!(listener_name, "event.listener");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn class_reports_gpuarray_with_wgpu_provider() {
        use runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider;
        use runmat_accelerate_api::AccelProvider;

        // Attempt to register a WGPU provider; skip if the environment lacks a compatible adapter.
        let provider = match ensure_wgpu_provider() {
            Ok(Some(p)) => p,
            _ => return,
        };

        let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("wgpu upload");
        let name = class_builtin(Value::GpuTensor(handle)).expect("class");
        assert_eq!(name, "gpuArray");
    }
}
