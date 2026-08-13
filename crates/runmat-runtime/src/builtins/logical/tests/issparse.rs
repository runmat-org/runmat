//! MATLAB-compatible `issparse` builtin for RunMat sparse matrix values.

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

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::issparse")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "issparse",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Reports whether the value is a host sparse matrix; dense gpuArray handles are not sparse.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::issparse")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "issparse",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query that executes outside fusion and returns a scalar logical.",
};

const ISSPARSE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when input is a sparse matrix.",
}];

const ISSPARSE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to test.",
}];

const ISSPARSE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = issparse(A)",
    inputs: &ISSPARSE_INPUTS,
    outputs: &ISSPARSE_OUTPUT,
}];

const ISSPARSE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSPARSE.INTERNAL",
    identifier: Some("RunMat:issparse:InternalError"),
    when: "Resident handle ownership or dtype metadata is contradictory.",
    message: "issparse: invalid resident metadata",
};
const ISSPARSE_ERRORS: [BuiltinErrorDescriptor; 1] = [ISSPARSE_ERROR_INTERNAL];

pub const ISSPARSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISSPARSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISSPARSE_ERRORS,
};
const ISSPARSE_DENSE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Dense arrays of every fixed-width integer class are not sparse.",
    }];
const ISSPARSE_SPARSE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed integer CSC storage is a RunMat-only value representation; MATLAB sparse numeric storage is limited to floating and logical classes.",
    }];
pub const ISSPARSE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "tf = issparse(dense_integer_A)",
        inputs: &ISSPARSE_DENSE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Returns scalar logical false from storage metadata. Resident integer handles are validated against their exact owner without gathering.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "tf = issparse(RunMat_sparse_integer_A)",
        inputs: &ISSPARSE_SPARSE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Returns scalar logical true for an already-existing RunMat sparse integer value. Creation and propagation of that value are compatibility-gated at their owning operations; this universal storage predicate does not reinterpret or hide the value.",
    },
];

#[runtime_builtin(
    name = "issparse",
    category = "logical/tests",
    summary = "Return true when a value is a sparse matrix.",
    keywords = "issparse,sparse,matrix,type,logical",
    accel = "metadata",
    type_resolver(bool_scalar_type),
    descriptor(crate::builtins::logical::tests::issparse::ISSPARSE_DESCRIPTOR),
    integer_capabilities(crate::builtins::logical::tests::issparse::ISSPARSE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::logical::tests::issparse"
)]
async fn issparse_builtin(value: Value) -> BuiltinResult<Value> {
    if let Value::GpuTensor(handle) = &value {
        if let Some(integer) = runmat_accelerate_api::handle_integer_type(handle) {
            let storage = runmat_accelerate_api::handle_storage(handle);
            if gpu_helpers::exact_provider_for_handle(handle).is_none()
                || storage != runmat_accelerate_api::GpuTensorStorage::Real
                || runmat_accelerate_api::handle_precision(handle).is_some()
                || runmat_accelerate_api::handle_is_logical(handle)
                || !gpu_helpers::gpu_class_metadata_matches(handle, None, Some(integer), false)
            {
                return Err(issparse_internal_error(
                    "issparse: resident integer metadata is contradictory",
                ));
            }
        }
    }
    Ok(Value::Bool(matches!(value, Value::SparseTensor(_))))
}

fn issparse_internal_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("issparse")
        .with_identifier(
            ISSPARSE_ERROR_INTERNAL
                .identifier
                .expect("internal error identifier"),
        )
        .build()
}

fn bool_scalar_type(_: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CellArray, CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, MException,
        ObjectInstance, SparseTensor, StringArray, StructValue, SymbolicExpr, Tensor, Value,
    };

    fn run_issparse(value: Value) -> BuiltinResult<Value> {
        block_on(super::issparse_builtin(value))
    }

    #[test]
    fn issparse_type_returns_bool() {
        assert_eq!(
            super::bool_scalar_type(&[Type::Unknown], &ResolveContext::new(Vec::new())),
            Type::Bool
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sparse_tensors_report_true() {
        let sparse = SparseTensor::new(3, 2, vec![0, 1, 2], vec![1, 2], vec![4.0, -1.0]).unwrap();
        assert_eq!(
            run_issparse(Value::SparseTensor(sparse)).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            run_issparse(Value::SparseTensor(SparseTensor::zeros(4, 5))).unwrap(),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dense_numeric_and_logical_values_report_false() {
        assert_eq!(
            run_issparse(Value::Int(IntValue::I32(1))).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(run_issparse(Value::Num(1.0)).unwrap(), Value::Bool(false));
        assert_eq!(
            run_issparse(Value::Complex(1.0, -2.0)).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(run_issparse(Value::Bool(true)).unwrap(), Value::Bool(false));
        assert_eq!(
            run_issparse(Value::LogicalArray(
                LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap()
            ))
            .unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_issparse(Value::Tensor(
                Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()
            ))
            .unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_issparse(Value::ComplexTensor(
                ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).unwrap()
            ))
            .unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn dense_arrays_of_all_integer_classes_report_false() {
        let storages = [
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![u8::MIN, u8::MAX]),
            IntegerStorage::U16(vec![u16::MIN, u16::MAX]),
            IntegerStorage::U32(vec![u32::MIN, u32::MAX]),
            IntegerStorage::U64(vec![u64::MIN, u64::MAX]),
        ];
        for storage in storages {
            let tensor = Tensor::new_integer(storage, vec![1, 2]).expect("integer tensor");
            assert_eq!(
                run_issparse(Value::Tensor(tensor)).expect("issparse"),
                Value::Bool(false)
            );
        }
    }

    #[test]
    fn runmat_sparse_integer_storage_reports_true_without_hiding_its_storage_kind() {
        let sparse = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            IntegerStorage::U64(vec![1, u64::MAX]),
        )
        .expect("sparse integer");
        assert_eq!(
            run_issparse(Value::SparseTensor(sparse)).expect("issparse"),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn containers_text_and_objects_report_false() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let string = Value::String("abc".into());
        let chars = CharArray::new_row("abc");
        let strings = StringArray::new(vec!["abc".into()], vec![1, 1]).unwrap();
        let structure = StructValue::new();
        let object = ObjectInstance::new("Example".into());
        assert_eq!(run_issparse(Value::Cell(cell)).unwrap(), Value::Bool(false));
        assert_eq!(run_issparse(string).unwrap(), Value::Bool(false));
        assert_eq!(
            run_issparse(Value::CharArray(chars)).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_issparse(Value::StringArray(strings)).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_issparse(Value::Struct(structure)).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_issparse(Value::Object(object)).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_issparse(Value::FunctionHandle("sin".into())).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_issparse(Value::Symbolic(SymbolicExpr::variable("x"))).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_issparse(Value::ClassRef("Example".into())).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_issparse(Value::MException(MException::new(
                "RunMat:test".into(),
                "not sparse".into()
            )))
            .unwrap(),
            Value::Bool(false)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_handles_report_false() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = run_issparse(Value::GpuTensor(handle.clone())).expect("issparse");
            assert_eq!(result, Value::Bool(false));
            provider.free(&handle).ok();
        });
    }

    #[test]
    fn resident_dense_integer_reports_false_from_exact_owner_metadata() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![2, 1])
                    .expect("integer tensor");
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("upload integer");
            assert_eq!(
                run_issparse(Value::GpuTensor(handle.clone())).expect("issparse"),
                Value::Bool(false)
            );
            assert!(gpu_helpers::exact_provider_for_handle(&handle).is_some());
            provider.free(&handle).ok();
        });
    }

    #[test]
    fn resident_dense_integer_rejects_contradictory_class_metadata() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1])
                .expect("integer tensor");
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("upload integer");
            runmat_accelerate_api::set_handle_class_name(&handle, "double");
            let error = run_issparse(Value::GpuTensor(handle.clone()))
                .expect_err("contradictory class metadata must reject");
            assert!(error.message().contains("metadata is contradictory"));
            provider.free(&handle).ok();
        });
    }
}
