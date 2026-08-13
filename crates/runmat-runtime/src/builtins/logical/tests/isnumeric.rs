//! MATLAB-compatible `isnumeric` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, ResolveContext, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::isnumeric")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isnumeric",
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

const ISNUMERIC_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when input uses numeric storage.",
}];

const ISNUMERIC_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to test.",
}];

const ISNUMERIC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isnumeric(A)",
    inputs: &ISNUMERIC_INPUTS,
    outputs: &ISNUMERIC_OUTPUT,
}];

const ISNUMERIC_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISNUMERIC.INTERNAL",
    identifier: Some("RunMat:isnumeric:InternalError"),
    when: "Internal gather/dispatch path fails.",
    message: "isnumeric: internal error",
};

const ISNUMERIC_ERRORS: [BuiltinErrorDescriptor; 1] = [ISNUMERIC_ERROR_INTERNAL];

pub const ISNUMERIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISNUMERIC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISNUMERIC_ERRORS,
};

const ISNUMERIC_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "MATLAB explicitly defines every signed and unsigned fixed-width integer class as numeric.",
    }];
pub const ISNUMERIC_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "tf = isnumeric(integer_A)",
        inputs: &ISNUMERIC_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Returns a host logical scalar from authoritative host dtype or coherent resident class metadata without reading payload values.",
    }];

fn isnumeric_error_with_message(
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
    name = "isnumeric",
    category = "logical/tests",
    summary = "Return true when a value is stored as numeric data.",
    keywords = "isnumeric,numeric,type,gpu",
    accel = "metadata",
    type_resolver(bool_scalar_type),
    descriptor(crate::builtins::logical::tests::isnumeric::ISNUMERIC_DESCRIPTOR),
    integer_capabilities(
        crate::builtins::logical::tests::isnumeric::ISNUMERIC_INTEGER_CAPABILITIES
    ),
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
    Ok(Value::Bool(checked_resident_is_numeric(&handle)?))
}

fn checked_resident_is_numeric(handle: &GpuTensorHandle) -> BuiltinResult<bool> {
    let owner = gpu_helpers::exact_provider_for_handle(handle).ok_or_else(|| {
        internal_error("isnumeric: no acceleration provider owns the input handle")
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
            "isnumeric: resident class metadata contradicts physical storage metadata",
        )
        .into());
    }
    Ok(!logical)
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
    isnumeric_error_with_message(message, &ISNUMERIC_ERROR_INTERNAL)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CellArray, CharArray, Closure, ComplexTensor, HandleRef, IntValue, IntegerStorage,
        Listener, LogicalArray, MException, ObjectInstance, ResolveContext, StringArray,
        StructValue, Tensor, Type,
    };

    fn run_isnumeric(value: Value) -> BuiltinResult<Value> {
        block_on(super::isnumeric_builtin(value))
    }

    fn test_handle_target() -> runmat_gc::GcHandle {
        runmat_gc::gc_allocate(Value::Num(0.0)).expect("gc allocation")
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

    #[test]
    fn all_integer_classes_report_true_without_conversion() {
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
                run_isnumeric(Value::Int(scalar)).unwrap(),
                Value::Bool(true)
            );
        }
    }

    #[test]
    fn resident_integer_classes_report_true_without_gather() {
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
                    run_isnumeric(Value::GpuTensor(handle.clone())).unwrap(),
                    Value::Bool(true)
                );
                provider.free(&handle).ok();
            }
        });
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
            bound_function: None,
            captures: vec![Value::Num(1.0)],
        };
        assert_eq!(
            run_isnumeric(Value::Closure(closure)).unwrap(),
            Value::Bool(false)
        );
        let handle = HandleRef {
            class_name: "runmat.Handle".into(),
            target: test_handle_target(),
            valid: true,
        };
        assert_eq!(
            run_isnumeric(Value::HandleObject(handle)).unwrap(),
            Value::Bool(false)
        );
        let listener = Listener {
            id: 1,
            target: test_handle_target(),
            target_class_name: "EventTarget".into(),
            event_name: "changed".into(),
            callback: test_handle_target(),
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
        let mex = MException::new("RunMat:mock".into(), "message".into());
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
                data: &tensor.materialize_f64(),
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
