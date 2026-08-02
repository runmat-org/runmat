//! MATLAB-compatible `underlyingType` and `isUnderlyingType` builtins.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::introspection::class::class_name_for_value;
use crate::builtins::introspection::type_resolvers::{
    is_underlying_type_type, underlying_type_type,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::{
    handle_class_name, handle_is_logical, handle_precision, provider_for_handle, ProviderPrecision,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::introspection::underlying_type"
)]
pub const UNDERLYING_TYPE_GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "underlyingType",
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
    notes: "Metadata-only query that returns a host string scalar. gpuArray inputs stay resident while RunMat reads logical/precision handle metadata.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::introspection::underlying_type"
)]
pub const UNDERLYING_TYPE_FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "underlyingType",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; the builtin executes on the host and returns a string scalar.",
};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::introspection::underlying_type"
)]
pub const IS_UNDERLYING_TYPE_GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isUnderlyingType",
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
    notes: "Metadata-only predicate that returns a host logical scalar without gathering gpuArray buffers.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::introspection::underlying_type"
)]
pub const IS_UNDERLYING_TYPE_FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isUnderlyingType",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion planning; the predicate executes on the host from metadata.",
};

const UNDERLYING_TYPE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "typename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Name of the underlying MATLAB data type.",
}];

const UNDERLYING_TYPE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array or object to inspect.",
}];

const UNDERLYING_TYPE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "typename = underlyingType(X)",
    inputs: &UNDERLYING_TYPE_INPUTS,
    outputs: &UNDERLYING_TYPE_OUTPUT,
}];

const IS_UNDERLYING_TYPE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when X has the requested underlying type.",
}];

const IS_UNDERLYING_TYPE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array or object to inspect.",
    },
    BuiltinParamDescriptor {
        name: "typename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Underlying data type name to test.",
    },
];

const IS_UNDERLYING_TYPE_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "tf = isUnderlyingType(X, typename)",
        inputs: &IS_UNDERLYING_TYPE_INPUTS,
        outputs: &IS_UNDERLYING_TYPE_OUTPUT,
    }];

const BUILTIN_IS_UNDERLYING_TYPE: &str = "isUnderlyingType";

const UNDERLYING_TYPE_ERRORS: [BuiltinErrorDescriptor; 0] = [];

const IS_UNDERLYING_TYPE_ERROR_TYPE_NAME_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IS_UNDERLYING_TYPE.TYPE_NAME_INVALID",
    identifier: Some("RunMat:isUnderlyingType:TypeNameInvalid"),
    when: "Second argument is not a string scalar or row character vector.",
    message: "isUnderlyingType: TYPENAME must be a string scalar or character vector",
};

const IS_UNDERLYING_TYPE_ERRORS: [BuiltinErrorDescriptor; 1] =
    [IS_UNDERLYING_TYPE_ERROR_TYPE_NAME_INVALID];

pub const UNDERLYING_TYPE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UNDERLYING_TYPE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UNDERLYING_TYPE_ERRORS,
};

pub const IS_UNDERLYING_TYPE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IS_UNDERLYING_TYPE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IS_UNDERLYING_TYPE_ERRORS,
};

#[runtime_builtin(
    name = "underlyingType",
    category = "introspection",
    summary = "Return the underlying MATLAB data type for a value.",
    keywords = "underlyingType,type inspection,underlying data type,gpuArray dtype",
    accel = "metadata",
    type_resolver(underlying_type_type),
    descriptor(crate::builtins::introspection::underlying_type::UNDERLYING_TYPE_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::underlying_type"
)]
fn underlying_type_builtin(value: Value) -> BuiltinResult<String> {
    Ok(underlying_type_for_value(&value))
}

#[runtime_builtin(
    name = "isUnderlyingType",
    category = "introspection",
    summary = "Test whether a value has a specified underlying MATLAB data type.",
    keywords = "isUnderlyingType,type checking,underlying data type,gpuArray dtype",
    accel = "metadata",
    type_resolver(is_underlying_type_type),
    descriptor(crate::builtins::introspection::underlying_type::IS_UNDERLYING_TYPE_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::underlying_type"
)]
fn is_underlying_type_builtin(value: Value, typename: Value) -> BuiltinResult<Value> {
    let requested = parse_type_name(&typename)?;
    Ok(Value::Bool(underlying_type_matches(
        &value,
        requested.as_str(),
    )))
}

/// Return the canonical underlying MATLAB data type for a runtime value.
pub(crate) fn underlying_type_for_value(value: &Value) -> String {
    match value {
        Value::Tensor(tensor) => tensor.numeric_dtype().class_name().to_string(),
        Value::SparseTensor(sparse) => sparse.class_name().to_string(),
        Value::ComplexTensor(tensor) => tensor.integer_data.as_ref().map_or_else(
            || "double".to_string(),
            |storage| storage.class_name().to_string(),
        ),
        Value::Complex(_, _) | Value::Num(_) => "double".to_string(),
        Value::Int(iv) => iv.class_name().to_string(),
        Value::Bool(_) | Value::LogicalArray(_) => "logical".to_string(),
        Value::GpuTensor(handle) => {
            if handle_is_logical(handle) {
                "logical".to_string()
            } else if let Some(class_name) = handle_class_name(handle) {
                class_name
            } else {
                match handle_precision(handle)
                    .or_else(|| provider_for_handle(handle).map(|provider| provider.precision()))
                    .unwrap_or(ProviderPrecision::F64)
                {
                    ProviderPrecision::F32 => "single".to_string(),
                    ProviderPrecision::F64 => "double".to_string(),
                }
            }
        }
        _ => class_name_for_value(value),
    }
}

pub(crate) fn underlying_type_matches(value: &Value, typename: &str) -> bool {
    let requested = typename.trim();
    !requested.is_empty() && underlying_type_for_value(value) == requested
}

fn parse_type_name(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::StringArray(sa) if sa.rows == 1 && sa.cols == 1 && !sa.data.is_empty() => {
            Ok(sa.data[0].clone())
        }
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        _ => Err(is_underlying_type_error(&IS_UNDERLYING_TYPE_ERROR_TYPE_NAME_INVALID).into()),
    }
}

fn is_underlying_type_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_IS_UNDERLYING_TYPE);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CellArray, CharArray, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage,
        LogicalArray, NumericDType, ObjectInstance, StringArray, StructValue, Tensor,
    };

    #[test]
    fn underlying_type_reports_core_host_types() {
        assert_eq!(underlying_type_for_value(&Value::Num(1.0)), "double");
        assert_eq!(
            underlying_type_for_value(&Value::Tensor(
                Tensor::new_with_dtype(vec![1.0, 2.0], vec![1, 2], NumericDType::F32)
                    .expect("single tensor")
            )),
            "single"
        );
        assert_eq!(
            underlying_type_for_value(&Value::Int(IntValue::U16(7))),
            "uint16"
        );
        assert_eq!(underlying_type_for_value(&Value::Bool(true)), "logical");
        assert_eq!(
            underlying_type_for_value(&Value::LogicalArray(
                LogicalArray::new(vec![1, 0], vec![1, 2]).expect("logical array")
            )),
            "logical"
        );
        assert_eq!(
            underlying_type_for_value(&Value::CharArray(CharArray::new_row("abc"))),
            "char"
        );
        assert_eq!(
            underlying_type_for_value(&Value::String("abc".into())),
            "string"
        );
    }

    #[test]
    fn underlying_type_reports_typed_sparse_integer_class() {
        let sparse = runmat_builtins::SparseTensor::new_integer(
            2,
            1,
            vec![0, 1],
            vec![1],
            IntegerStorage::I64(vec![i64::MIN]),
        )
        .expect("int64 sparse");

        assert_eq!(
            underlying_type_for_value(&Value::SparseTensor(sparse)),
            "int64"
        );
    }

    #[test]
    fn underlying_type_reports_every_typed_complex_integer_class() {
        let cases = [
            (
                "int8",
                IntegerStorage::I8(vec![-1]),
                IntegerStorage::I8(vec![2]),
            ),
            (
                "int16",
                IntegerStorage::I16(vec![-3]),
                IntegerStorage::I16(vec![4]),
            ),
            (
                "int32",
                IntegerStorage::I32(vec![-5]),
                IntegerStorage::I32(vec![6]),
            ),
            (
                "int64",
                IntegerStorage::I64(vec![-7]),
                IntegerStorage::I64(vec![8]),
            ),
            (
                "uint8",
                IntegerStorage::U8(vec![1]),
                IntegerStorage::U8(vec![2]),
            ),
            (
                "uint16",
                IntegerStorage::U16(vec![3]),
                IntegerStorage::U16(vec![4]),
            ),
            (
                "uint32",
                IntegerStorage::U32(vec![5]),
                IntegerStorage::U32(vec![6]),
            ),
            (
                "uint64",
                IntegerStorage::U64(vec![7]),
                IntegerStorage::U64(vec![8]),
            ),
        ];

        for (expected, real, imag) in cases {
            let storage = IntegerComplexStorage::new(real, imag).expect("matching components");
            let value = Value::ComplexTensor(
                ComplexTensor::new_integer(storage, vec![1, 1]).expect("typed complex"),
            );
            assert_eq!(underlying_type_for_value(&value), expected);
            assert_eq!(underlying_type_matches(&value, expected), true);
        }
    }

    #[test]
    fn underlying_type_matches_class_for_containers_and_objects() {
        let cell =
            Value::Cell(CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap());
        assert_eq!(underlying_type_for_value(&cell), "cell");

        let mut st = StructValue::new();
        st.fields.insert("x".into(), Value::Num(1.0));
        assert_eq!(underlying_type_for_value(&Value::Struct(st)), "struct");

        let object = ObjectInstance::new("pkg.Point".into());
        assert_eq!(
            underlying_type_for_value(&Value::Object(object)),
            "pkg.Point"
        );
    }

    #[test]
    fn is_underlying_type_accepts_string_and_char_names() {
        assert_eq!(
            is_underlying_type_builtin(Value::Int(IntValue::I32(3)), Value::from("int32"))
                .expect("string name"),
            Value::Bool(true)
        );
        assert_eq!(
            is_underlying_type_builtin(
                Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
                Value::CharArray(CharArray::new_row("double")),
            )
            .expect("char name"),
            Value::Bool(true)
        );
        assert_eq!(
            is_underlying_type_builtin(Value::Bool(true), Value::from("double"))
                .expect("false predicate"),
            Value::Bool(false)
        );
        assert_eq!(
            is_underlying_type_builtin(Value::Num(1.0), Value::from("Double"))
                .expect("case-sensitive mismatch"),
            Value::Bool(false)
        );
    }

    #[test]
    fn is_underlying_type_accepts_scalar_string_array_only() {
        let name = Value::StringArray(
            StringArray::new(vec!["single".into()], vec![1, 1]).expect("string scalar"),
        );
        let value = Value::Tensor(
            Tensor::new_with_dtype(vec![1.0], vec![1, 1], NumericDType::F32)
                .expect("single tensor"),
        );
        assert_eq!(
            is_underlying_type_builtin(value, name).expect("scalar string array"),
            Value::Bool(true)
        );

        let invalid = Value::StringArray(
            StringArray::new(vec!["double".into(), "single".into()], vec![1, 2])
                .expect("string row"),
        );
        let err = is_underlying_type_builtin(Value::Num(1.0), invalid)
            .expect_err("non-scalar string array");
        assert_eq!(
            err.identifier(),
            Some("RunMat:isUnderlyingType:TypeNameInvalid")
        );
    }

    #[test]
    fn underlying_type_uses_gpu_metadata_without_gather() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new_with_dtype(vec![1.0, 2.0], vec![1, 2], NumericDType::F32).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            runmat_accelerate_api::set_handle_precision(
                &handle,
                runmat_accelerate_api::ProviderPrecision::F32,
            );

            assert_eq!(
                underlying_type_builtin(Value::GpuTensor(handle.clone())).expect("underlying"),
                "single"
            );
            assert_eq!(
                is_underlying_type_builtin(Value::GpuTensor(handle), Value::from("single"))
                    .expect("predicate"),
                Value::Bool(true)
            );
        });
    }

    #[test]
    fn underlying_type_reports_integer_gpu_class_from_native_provider_storage() {
        test_support::with_test_provider(|_| {
            let tensor = Tensor::new(vec![1.2, -3.7, 123456.0], vec![3, 1]).unwrap();
            let handle = match crate::dispatcher::call_builtin(
                "gpuArray",
                &[Value::Tensor(tensor), Value::from("int32")],
            )
            .expect("gpuArray int32 native upload")
            {
                Value::GpuTensor(handle) => handle,
                other => panic!("expected gpu tensor, got {other:?}"),
            };

            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(runmat_accelerate_api::IntegerElementType::I32)
            );
            assert_eq!(
                underlying_type_builtin(Value::GpuTensor(handle.clone())).expect("underlying"),
                "int32"
            );
            assert_eq!(
                is_underlying_type_builtin(Value::GpuTensor(handle), Value::from("int32"))
                    .expect("predicate"),
                Value::Bool(true)
            );
        });
    }

    #[test]
    fn underlying_type_reports_logical_for_gpu_masks() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            runmat_accelerate_api::set_handle_logical(&handle, true);

            assert_eq!(
                underlying_type_builtin(Value::GpuTensor(handle.clone())).expect("underlying"),
                "logical"
            );
            assert_eq!(
                is_underlying_type_builtin(Value::GpuTensor(handle), Value::from("logical"))
                    .expect("predicate"),
                Value::Bool(true)
            );
        });
    }
}
