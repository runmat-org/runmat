//! MATLAB-compatible `islogical` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, ResolveContext, Type,
    Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::islogical")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "islogical",
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
    notes: "Reads coherent owning-provider and handle class metadata and returns a host logical scalar without gathering payload data.",
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
pub const ISLOGICAL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "islogical is a universal type predicate; all eight integer classes return scalar false, including resident integer arrays, without reading payload data.",
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
    integer_audit(crate::builtins::logical::tests::islogical::ISLOGICAL_INTEGER_AUDIT),
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
    Ok(Value::Bool(checked_resident_is_logical(&handle)?))
}

fn checked_resident_is_logical(handle: &GpuTensorHandle) -> BuiltinResult<bool> {
    let owner = gpu_helpers::exact_provider_for_handle(handle).ok_or_else(|| {
        internal_error("islogical: no acceleration provider owns the input handle")
    })?;
    let logical = runmat_accelerate_api::handle_is_logical(handle);
    let integer = runmat_accelerate_api::handle_integer_type(handle);
    let precision = runmat_accelerate_api::handle_precision(handle);
    let storage = runmat_accelerate_api::handle_storage(handle);
    let structurally_valid = if logical {
        integer.is_none()
            && precision == Some(owner.precision())
            && storage == GpuTensorStorage::Real
    } else if integer.is_some() {
        precision.is_none() && storage == GpuTensorStorage::Real
    } else {
        precision == Some(owner.precision())
            && matches!(
                storage,
                GpuTensorStorage::Real | GpuTensorStorage::ComplexInterleaved
            )
    };
    if !structurally_valid
        || !gpu_helpers::gpu_class_metadata_matches(handle, precision, integer, logical)
    {
        return Err(internal_error(
            "islogical: resident class metadata contradicts physical storage metadata",
        )
        .into());
    }
    Ok(logical)
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
    use runmat_builtins::{CharArray, IntValue, IntegerStorage, LogicalArray, Tensor, Value};

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

    #[test]
    fn all_integer_classes_report_false_without_conversion() {
        let scalars = [
            IntValue::I8(-1),
            IntValue::I16(-2),
            IntValue::I32(-3),
            IntValue::I64(i64::MIN),
            IntValue::U8(1),
            IntValue::U16(2),
            IntValue::U32(3),
            IntValue::U64(u64::MAX),
        ];
        for scalar in scalars {
            assert_eq!(
                run_islogical(Value::Int(scalar)).unwrap(),
                Value::Bool(false)
            );
        }
    }

    #[test]
    fn resident_integer_classes_report_false_without_gather() {
        test_support::with_test_provider(|provider| {
            let storages = [
                IntegerStorage::I8(vec![-1]),
                IntegerStorage::I16(vec![-2]),
                IntegerStorage::I32(vec![-3]),
                IntegerStorage::I64(vec![i64::MIN]),
                IntegerStorage::U8(vec![1]),
                IntegerStorage::U16(vec![2]),
                IntegerStorage::U32(vec![3]),
                IntegerStorage::U64(vec![u64::MAX]),
            ];
            for storage in storages {
                let tensor = Tensor::new_integer(storage, vec![1, 1]).unwrap();
                let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload integer");
                assert_eq!(
                    run_islogical(Value::GpuTensor(handle.clone())).unwrap(),
                    Value::Bool(false)
                );
                provider.free(&handle).ok();
            }
        });
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
                data: &tensor.materialize_f64(),
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
                data: &tensor.materialize_f64(),
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
