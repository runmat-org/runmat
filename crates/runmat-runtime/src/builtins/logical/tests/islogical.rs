//! MATLAB-compatible `islogical` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::islogical")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "islogical",
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
        "Reads provider metadata when `logical_islogical` is implemented; otherwise consults runtime residency tracking and, as a last resort, gathers once to the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::islogical")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "islogical",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query that executes outside of fusion pipelines.",
};

const BUILTIN_NAME: &str = "islogical";

const ISLOGICAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when input uses logical storage.",
}];

const ISLOGICAL_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to test.",
}];

const ISLOGICAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = islogical(A)",
    inputs: &ISLOGICAL_INPUTS,
    outputs: &ISLOGICAL_OUTPUT,
}];

const ISLOGICAL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISLOGICAL.INTERNAL",
    identifier: Some("RunMat:islogical:InternalError"),
    when: "Internal gather/dispatch path fails.",
    message: "islogical: internal error",
};

const ISLOGICAL_ERRORS: [BuiltinErrorDescriptor; 1] = [ISLOGICAL_ERROR_INTERNAL];

pub const ISLOGICAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISLOGICAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISLOGICAL_ERRORS,
};

fn islogical_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "islogical",
    category = "logical/tests",
    summary = "Return true when a value uses logical storage.",
    keywords = "islogical,logical,bool,gpu",
    accel = "metadata",
    type_resolver(bool_scalar_type),
    descriptor(crate::builtins::logical::tests::islogical::ISLOGICAL_DESCRIPTOR),
    builtin_path = "crate::builtins::logical::tests::islogical"
)]
async fn islogical_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => islogical_gpu(handle).await,
        other => islogical_host(other),
    }
}

fn bool_scalar_type(_: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

async fn islogical_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(flag) = provider.logical_islogical(&handle) {
            return Ok(Value::Bool(flag));
        }
    }

    if runmat_accelerate_api::handle_is_logical(&handle) {
        return Ok(Value::Bool(true));
    }

    let gpu_value = Value::GpuTensor(handle.clone());
    let gathered = gpu_helpers::gather_value_async(&gpu_value)
        .await
        .map_err(|err| {
            islogical_error_with_message(format!("islogical: {err}"), &ISLOGICAL_ERROR_INTERNAL)
        })?;
    islogical_host(gathered)
}

fn islogical_host(value: Value) -> BuiltinResult<Value> {
    let flag = matches!(value, Value::Bool(_) | Value::LogicalArray(_));
    match value {
        Value::GpuTensor(_) => Err(internal_error(
            "islogical: internal error, GPU value reached host path",
        )),
        _ => Ok(Value::Bool(flag)),
    }
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    islogical_error_with_message(message, &ISLOGICAL_ERROR_INTERNAL)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_backend;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CharArray, LogicalArray, Tensor, Value};

    fn run_islogical(value: Value) -> BuiltinResult<Value> {
        block_on(super::islogical_builtin(value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_scalars_report_true() {
        assert_eq!(run_islogical(Value::Bool(true)).unwrap(), Value::Bool(true));
        assert_eq!(
            run_islogical(Value::Bool(false)).unwrap(),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_arrays_report_true() {
        let array = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
        assert_eq!(
            run_islogical(Value::LogicalArray(array)).unwrap(),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_values_report_false() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        assert_eq!(
            run_islogical(Value::Tensor(tensor)).unwrap(),
            Value::Bool(false)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_values_report_false() {
        assert_eq!(
            run_islogical(Value::String("runmat".to_string())).unwrap(),
            Value::Bool(false)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_arrays_report_false() {
        let chars = CharArray::new("rm".chars().collect(), 1, 2).unwrap();
        assert_eq!(
            run_islogical(Value::CharArray(chars)).unwrap(),
            Value::Bool(false)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn structures_and_cells_report_false() {
        let struct_value = runmat_builtins::StructValue::new();
        let cell = runmat_builtins::CellArray::new(vec![Value::Bool(true)], 1, 1).unwrap();
        assert_eq!(
            run_islogical(Value::Struct(struct_value)).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_islogical(Value::Cell(cell)).unwrap(),
            Value::Bool(false)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_handles_use_metadata_when_available() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 0.0, 1.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let logical_value = gpu_helpers::logical_gpu_value(handle.clone());
            let result = run_islogical(logical_value).expect("islogical");
            assert_eq!(result, Value::Bool(true));
            provider.free(&handle).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_numeric_handles_report_false() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = run_islogical(Value::GpuTensor(handle.clone())).expect("islogical");
            assert_eq!(result, Value::Bool(false));
            provider.free(&handle).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn islogical_wgpu_elem_eq_sets_metadata() {
        let _ = wgpu_backend::register_wgpu_provider(wgpu_backend::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let lhs = vec![1.0, 0.0, 1.0, 0.0];
        let rhs = vec![1.0, 0.0, 0.0, 0.0];
        let shape = vec![4, 1];
        let lhs_view = HostTensorView {
            data: &lhs,
            shape: &shape,
        };
        let rhs_view = HostTensorView {
            data: &rhs,
            shape: &shape,
        };
        let lhs_handle = provider.upload(&lhs_view).expect("upload lhs");
        let rhs_handle = provider.upload(&rhs_view).expect("upload rhs");
        let mask_handle = block_on(provider.elem_eq(&lhs_handle, &rhs_handle)).expect("elem_eq");

        let result = run_islogical(Value::GpuTensor(mask_handle.clone())).expect("islogical");
        assert_eq!(result, Value::Bool(true));

        provider.free(&mask_handle).ok();
        provider.free(&lhs_handle).ok();
        provider.free(&rhs_handle).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn islogical_wgpu_numeric_stays_false() {
        let _ = wgpu_backend::register_wgpu_provider(wgpu_backend::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let data = vec![0.25, 0.5, 0.75];
        let shape = vec![3, 1];
        let view = HostTensorView {
            data: &data,
            shape: &shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = run_islogical(Value::GpuTensor(handle.clone())).expect("islogical");
        assert_eq!(result, Value::Bool(false));
        provider.free(&handle).ok();
    }
}
