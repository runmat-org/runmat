//! MATLAB-compatible `isreal` builtin with GPU-aware semantics for RunMat.
//!
//! This predicate reports whether a value is stored without an imaginary
//! component. Unlike `isfinite`/`isnan`, it returns a single logical scalar.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Type, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::isreal")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isreal",
    op_kind: GpuOpKind::Custom("storage-check"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("logical_isreal")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Queries provider metadata when `logical_isreal` is available; otherwise gathers once and inspects host storage.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::isreal")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isreal",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Scalar metadata predicate that remains outside fusion graphs.",
};

const BUILTIN_NAME: &str = "isreal";
const IDENTIFIER_INTERNAL: &str = "RunMat:isreal:InternalError";

#[runtime_builtin(
    name = "isreal",
    category = "logical/tests",
    summary = "Return true when a value uses real storage without an imaginary component.",
    keywords = "isreal,real,complex,gpu,logical",
    accel = "metadata",
    type_resolver(bool_scalar_type),
    builtin_path = "crate::builtins::logical::tests::isreal"
)]
async fn isreal_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => isreal_gpu(handle).await,
        other => isreal_host(other),
    }
}

fn bool_scalar_type(_: &[Type]) -> Type {
    Type::Bool
}

async fn isreal_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(flag) = provider.logical_isreal(&handle) {
            return Ok(Value::Bool(flag));
        }
    }

    let gpu_value = Value::GpuTensor(handle);
    let gathered = gpu_helpers::gather_value_async(&gpu_value)
        .await
        .map_err(|err| internal_error(format!("isreal: {err}")))?;
    isreal_host(gathered)
}

fn isreal_host(value: Value) -> BuiltinResult<Value> {
    let flag = match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(_) => true,
        Value::LogicalArray(_) => true,
        Value::CharArray(_) => true,
        Value::Complex(_, _) => false,
        Value::ComplexTensor(_) => false,
        Value::String(_) => false,
        Value::StringArray(_) => false,
        Value::Struct(_) => false,
        Value::Cell(_) => false,
        Value::Object(_) => false,
        Value::HandleObject(_) => false,
        Value::Listener(_) => false,
        Value::FunctionHandle(_) => false,
        Value::Closure(_) => false,
        Value::ClassRef(_) => false,
        Value::MException(_) => false,
        Value::GpuTensor(_) => {
            return Err(internal_error(
                "isreal: internal error, GPU value reached host path",
            ));
        }
    };
    Ok(Value::Bool(flag))
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(IDENTIFIER_INTERNAL)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{
        CellArray, CharArray, Closure, ComplexTensor, HandleRef, Listener, LogicalArray,
        MException, ObjectInstance, StructValue, Tensor,
    };
    use runmat_gc_api::GcPtr;

    fn run_isreal(value: Value) -> BuiltinResult<Value> {
        block_on(super::isreal_builtin(value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isreal_reports_true_for_real_scalars() {
        let real = run_isreal(Value::Num(42.0)).expect("isreal");
        let integer = run_isreal(Value::from(5_i32)).expect("isreal");
        let boolean = run_isreal(Value::Bool(false)).expect("isreal");
        assert_eq!(real, Value::Bool(true));
        assert_eq!(integer, Value::Bool(true));
        assert_eq!(boolean, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isreal_rejects_complex_storage_even_with_zero_imaginary_part() {
        let complex = run_isreal(Value::Complex(3.0, 4.0)).expect("isreal");
        let complex_zero_imag = run_isreal(Value::Complex(12.0, 0.0)).expect("isreal");
        let complex_tensor = ComplexTensor::new(vec![(1.0, 0.0), (2.0, -1.0)], vec![2, 1]).unwrap();
        let tensor_flag = run_isreal(Value::ComplexTensor(complex_tensor)).expect("isreal");
        assert_eq!(complex, Value::Bool(false));
        assert_eq!(complex_zero_imag, Value::Bool(false));
        assert_eq!(tensor_flag, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isreal_handles_array_and_container_types() {
        let tensor = Tensor::new(vec![1.0, -2.0, 3.5], vec![3, 1]).unwrap();
        let logical = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
        let chars = CharArray::new_row("RunMat");
        let string_flag = run_isreal(Value::from("RunMat")).expect("isreal");
        let string_array =
            runmat_builtins::StringArray::new(vec!["a".into(), "b".into()], vec![2]).unwrap();
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let mut fields = StructValue::new();
        fields.fields.insert("name".into(), Value::from("Ada"));
        let object = ObjectInstance::new("RunMat.Object".into());

        let tensor_flag = run_isreal(Value::Tensor(tensor)).expect("isreal");
        let logical_flag = run_isreal(Value::LogicalArray(logical)).expect("isreal");
        let char_flag = run_isreal(Value::CharArray(chars)).expect("isreal");
        let string_array_flag =
            run_isreal(Value::StringArray(string_array)).expect("isreal string array");
        let cell_flag = run_isreal(Value::Cell(cell)).expect("isreal cell");
        let struct_flag = run_isreal(Value::Struct(fields)).expect("isreal struct");
        let object_flag = run_isreal(Value::Object(object)).expect("isreal object");

        assert_eq!(tensor_flag, Value::Bool(true));
        assert_eq!(logical_flag, Value::Bool(true));
        assert_eq!(char_flag, Value::Bool(true));
        assert_eq!(string_flag, Value::Bool(false));
        assert_eq!(string_array_flag, Value::Bool(false));
        assert_eq!(cell_flag, Value::Bool(false));
        assert_eq!(struct_flag, Value::Bool(false));
        assert_eq!(object_flag, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isreal_handles_function_and_handle_like_types() {
        let function_flag =
            run_isreal(Value::FunctionHandle("runmat_builtin".into())).expect("isreal fn");
        let closure_flag = run_isreal(Value::Closure(Closure {
            function_name: "anon".into(),
            captures: vec![Value::Num(1.0)],
        }))
        .expect("isreal closure");
        let handle_flag = run_isreal(Value::HandleObject(HandleRef {
            class_name: "MockHandle".into(),
            target: GcPtr::null(),
            valid: true,
        }))
        .expect("isreal handle");
        let listener_flag = run_isreal(Value::Listener(Listener {
            id: 42,
            target: GcPtr::null(),
            event_name: "changed".into(),
            callback: GcPtr::null(),
            enabled: true,
            valid: true,
        }))
        .expect("isreal listener");
        let class_ref_flag =
            run_isreal(Value::ClassRef("pkg.Class".into())).expect("isreal classref");
        let mex_flag = run_isreal(Value::MException(MException::new(
            "MATLAB:mock".into(),
            "message".into(),
        )))
        .expect("isreal mexception");

        assert_eq!(function_flag, Value::Bool(false));
        assert_eq!(closure_flag, Value::Bool(false));
        assert_eq!(handle_flag, Value::Bool(false));
        assert_eq!(listener_flag, Value::Bool(false));
        assert_eq!(class_ref_flag, Value::Bool(false));
        assert_eq!(mex_flag, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isreal_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = run_isreal(Value::GpuTensor(handle)).expect("isreal gpu");
            assert_eq!(result, Value::Bool(true));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn isreal_wgpu_provider_reports_true() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let result = run_isreal(Value::GpuTensor(handle)).expect("isreal gpu");
        assert_eq!(result, Value::Bool(true));
    }
}
