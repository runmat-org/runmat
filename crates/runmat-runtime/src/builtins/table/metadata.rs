use runmat_builtins::TABLE_ERRORS;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
pub use runmat_builtins::{DICTIONARY_DESCRIPTOR, TABLE_DESCRIPTOR};
pub use runmat_builtins::{PARQUETINFO_DESCRIPTOR, PARQUETREAD_DESCRIPTOR};
pub(super) use runmat_builtins::{
    TABLE_ERROR_INVALID_ARGUMENT, TABLE_ERROR_INVALID_INDEX, TABLE_ERROR_INVALID_VARIABLE,
    TABLE_ERROR_IO, TABLE_ERROR_UNSUPPORTED_FILE,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, RuntimeError};

use super::TABLE_CLASS;

use runmat_builtins::TABLE_IMPORT_ANY_OUTPUT as ANY_OUTPUT;
const VALUE_AND_ARGS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input value.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options or conversion arguments.",
    },
];
const ARRAY_DATASTORE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "arrds",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "ArrayDatastore object retaining the in-memory data and read properties.",
}];
const ARRAY_DATASTORE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "In-memory array managed by the datastore.",
}];
const ARRAY_DATASTORE_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "In-memory array managed by the datastore.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "ReadSize, IterationDimension, or OutputType options.",
    },
];
const VARIADIC_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Input values and name-value options.",
}];

const ARRAY_DATASTORE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "arrds = arrayDatastore(A)",
        inputs: &ARRAY_DATASTORE_INPUT,
        outputs: &ARRAY_DATASTORE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "arrds = arrayDatastore(A, nameValuePairs...)",
        inputs: &ARRAY_DATASTORE_INPUTS_NAME_VALUE,
        outputs: &ARRAY_DATASTORE_OUTPUT,
    },
];
const COMPAT_VALUE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = tabularBuiltin(A, ...)",
    inputs: &VALUE_AND_ARGS_INPUTS,
    outputs: &ANY_OUTPUT,
}];
const COMPAT_VARIADIC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = tabularBuiltin(args...)",
    inputs: &VARIADIC_INPUTS,
    outputs: &ANY_OUTPUT,
}];

pub use runmat_builtins::DETECT_IMPORT_OPTIONS_DESCRIPTOR;
pub use runmat_builtins::READTABLE_DESCRIPTOR;
pub use runmat_builtins::SPREADSHEET_IMPORT_OPTIONS_DESCRIPTOR;
pub const ARRAY_DATASTORE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ARRAY_DATASTORE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const TABLE_COMPAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COMPAT_VALUE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const TABLE_VARIADIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COMPAT_VARIADIC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::table")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "table",
    op_kind: GpuOpKind::Custom("table"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Tables are host containers. GPU variables are gathered when tabular algorithms need row-wise access.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::table")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "table",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Tables are structured host containers and are not fusion operands.",
};

pub(super) fn table_error(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(TABLE_CLASS);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

pub(super) fn table_error_with_source<E>(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: E,
) -> RuntimeError
where
    E: crate::runtime_error::RuntimeErrorSourceBound,
{
    let mut builder = build_runtime_error(message)
        .with_builtin(TABLE_CLASS)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

pub(super) fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    table_error(&TABLE_ERROR_INVALID_ARGUMENT, message)
}

pub(super) fn invalid_index(message: impl Into<String>) -> RuntimeError {
    table_error(&TABLE_ERROR_INVALID_INDEX, message)
}

pub(super) fn invalid_variable(message: impl Into<String>) -> RuntimeError {
    table_error(&TABLE_ERROR_INVALID_VARIABLE, message)
}

pub(super) fn map_control_flow(mut err: RuntimeError) -> RuntimeError {
    // Gather/control errors already have their owning typed identity/category.
    // Adding table context must not turn cancellation or admission into an
    // ordinary semantic error or grant an operational retry.
    err.gpu_gather_retry = crate::GpuGatherRetry::Never;
    err
}
