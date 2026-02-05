//! MATLAB-compatible `isnumeric` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{ResolveContext, Type, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::isnumeric")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isnumeric",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("logical_islogical")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Uses provider metadata to distinguish logical gpuArrays from numeric ones; otherwise falls back to runtime residency tracking.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::isnumeric")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isnumeric",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Type check executed outside fusion; planners treat it as a scalar metadata query.",
};

const BUILTIN_NAME: &str = "isnumeric";
const IDENTIFIER_INTERNAL: &str = "RunMat:isnumeric:InternalError";

#[runtime_builtin(
    name = "isnumeric",
    category = "logical/tests",
    summary = "Return true when a value is stored as numeric data.",
    keywords = "isnumeric,numeric,type,gpu",
    accel = "metadata",
    type_resolver(bool_scalar_type),
    builtin_path = "crate::builtins::logical::tests::isnumeric"
)]
async fn isnumeric_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => isnumeric_gpu(handle).await,
        other => Ok(Value::Bool(isnumeric_value(&other))),
    }
}

fn bool_scalar_type(_: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

async fn isnumeric_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(flag) = provider.logical_islogical(&handle) {
            return Ok(Value::Bool(!flag));
        }
    }

    if runmat_accelerate_api::handle_is_logical(&handle) {
        return Ok(Value::Bool(false));
    }

    // Fall back to gathering only when metadata is unavailable.
    let gpu_value = Value::GpuTensor(handle.clone());
    let gathered = gpu_helpers::gather_value_async(&gpu_value)
        .await
        .map_err(|err| internal_error(format!("isnumeric: {err}")))?;
    isnumeric_host(gathered)
}

fn isnumeric_host(value: Value) -> BuiltinResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        return Err(internal_error(
            "isnumeric: internal error, GPU value reached host path",
        ));
    }
    Ok(Value::Bool(isnumeric_value(&value)))
}

fn isnumeric_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Complex(_, _)
            | Value::Tensor(_)
            | Value::ComplexTensor(_)
    )
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
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CellArray, CharArray, Closure, ComplexTensor, HandleRef, IntValue, Listener, LogicalArray,
        MException, ObjectInstance, ResolveContext, StringArray, StructValue, Tensor, Type,
    };
    use runmat_gc_api::GcPtr;

    fn run_isnumeric(value: Value) -> BuiltinResult<Value> {
        block_on(super::isnumeric_builtin(value))
    }

    #[test]
    fn isnumeric_type_returns_bool() {
        assert_eq!(
            bool_scalar_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::Bool
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_scalars_return_true() {
        assert_eq!(run_isnumeric(Value::Num(3.5)).unwrap(), Value::Bool(true));
        assert_eq!(
            run_isnumeric(Value::Int(IntValue::I16(7))).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            run_isnumeric(Value::Complex(1.0, -2.0)).unwrap(),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_tensors_return_true() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        assert_eq!(
            run_isnumeric(Value::Tensor(tensor)).unwrap(),
            Value::Bool(true)
        );

        let complex = ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![2, 1]).unwrap();
        assert_eq!(
            run_isnumeric(Value::ComplexTensor(complex)).unwrap(),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn non_numeric_types_return_false() {
        assert_eq!(
            run_isnumeric(Value::Bool(true)).unwrap(),
            Value::Bool(false)
        );

        let logical = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        assert_eq!(
            run_isnumeric(Value::LogicalArray(logical)).unwrap(),
            Value::Bool(false)
        );

        let chars = CharArray::new("rm".chars().collect(), 1, 2).unwrap();
        assert_eq!(
            run_isnumeric(Value::CharArray(chars)).unwrap(),
            Value::Bool(false)
        );

        assert_eq!(
            run_isnumeric(Value::String("runmat".into())).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_isnumeric(Value::Struct(StructValue::new())).unwrap(),
            Value::Bool(false)
        );
        let string_array =
            StringArray::new(vec!["foo".into(), "bar".into()], vec![1, 2]).expect("string array");
        assert_eq!(
            run_isnumeric(Value::StringArray(string_array)).unwrap(),
            Value::Bool(false)
        );
        let cell =
            CellArray::new(vec![Value::Num(1.0), Value::Bool(false)], 1, 2).expect("cell array");
        assert_eq!(
            run_isnumeric(Value::Cell(cell)).unwrap(),
            Value::Bool(false)
        );
        let object = ObjectInstance::new("runmat.MockObject".into());
        assert_eq!(
            run_isnumeric(Value::Object(object)).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_isnumeric(Value::FunctionHandle("runmat_fun".into())).unwrap(),
            Value::Bool(false)
        );
        let closure = Closure {
            function_name: "anon".into(),
            captures: vec![Value::Num(1.0)],
        };
        assert_eq!(
            run_isnumeric(Value::Closure(closure)).unwrap(),
            Value::Bool(false)
        );
        let handle = HandleRef {
            class_name: "runmat.Handle".into(),
            target: GcPtr::null(),
            valid: true,
        };
        assert_eq!(
            run_isnumeric(Value::HandleObject(handle)).unwrap(),
            Value::Bool(false)
        );
        let listener = Listener {
            id: 1,
            target: GcPtr::null(),
            event_name: "changed".into(),
            callback: GcPtr::null(),
            enabled: true,
            valid: true,
        };
        assert_eq!(
            run_isnumeric(Value::Listener(listener)).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_isnumeric(Value::ClassRef("pkg.Class".into())).unwrap(),
            Value::Bool(false)
        );
        let mex = MException::new("MATLAB:mock".into(), "message".into());
        assert_eq!(
            run_isnumeric(Value::MException(mex)).unwrap(),
            Value::Bool(false)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_numeric_and_logical_handles() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let numeric_handle = provider.upload(&view).expect("upload");
            let numeric = run_isnumeric(Value::GpuTensor(numeric_handle.clone())).unwrap();
            assert_eq!(numeric, Value::Bool(true));

            let logical_value = gpu_helpers::logical_gpu_value(numeric_handle.clone());
            let logical = run_isnumeric(logical_value).unwrap();
            assert_eq!(logical, Value::Bool(false));

            runmat_accelerate_api::clear_handle_logical(&numeric_handle);
            provider.free(&numeric_handle).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn isnumeric_wgpu_handles_respect_metadata() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![4, 1];
        let view = HostTensorView {
            data: &data,
            shape: &shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let numeric = run_isnumeric(Value::GpuTensor(handle.clone())).unwrap();
        assert_eq!(numeric, Value::Bool(true));

        let logical_value = gpu_helpers::logical_gpu_value(handle.clone());
        let logical = run_isnumeric(logical_value).unwrap();
        assert_eq!(logical, Value::Bool(false));

        runmat_accelerate_api::clear_handle_logical(&handle);
        provider.free(&handle).ok();
    }
}
