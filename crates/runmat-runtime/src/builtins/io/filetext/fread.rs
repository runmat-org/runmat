//! MATLAB-compatible `fread` builtin for RunMat.

#[cfg(test)]
use runmat_accelerate_api::HostTensorView;
use std::io::{ErrorKind, Read, Seek, SeekFrom};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, IntValue, IntegerStorage, LogicalArray, NumericDType, NumericScalar, NumericStorage,
    Tensor, Value,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::io::filetext::{helpers::extract_scalar_string, registry};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use runmat_filesystem::File;

const BUILTIN_NAME: &str = "fread";

const FREAD_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fread-like",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "the fread \"like\" prototype selector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FreadLikeExtension"),
};
const FREAD_INTEGER_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fread-integer-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "integer-class fread file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FreadIntegerIdExtension"),
};
const FREAD_INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fread-integer-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "typed integer fread size controls are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FreadIntegerSizeExtension"),
};
const FREAD_INTEGER_SKIP_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fread-integer-skip",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "typed integer fread skip controls are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FreadIntegerSkipExtension"),
};
const FREAD_LOGICAL_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fread-logical-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "logical fread file identifiers, sizes, and skips are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FreadLogicalControlExtension"),
};
const FREAD_SINGLE_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fread-single-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "single-precision fread file identifiers, sizes, and skips are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FreadSingleControlExtension"),
};
const FREAD_RESIDENT_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fread-resident-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "provider-resident fread control arguments are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FreadResidentControlExtension"),
};

pub const FREAD_EXTENSIONS: [BuiltinExtensionDescriptor; 7] = [
    FREAD_LIKE_EXTENSION,
    FREAD_INTEGER_ID_EXTENSION,
    FREAD_INTEGER_SIZE_EXTENSION,
    FREAD_INTEGER_SKIP_EXTENSION,
    FREAD_LOGICAL_CONTROL_EXTENSION,
    FREAD_SINGLE_CONTROL_EXTENSION,
    FREAD_RESIDENT_CONTROL_EXTENSION,
];

const FREAD_INTEGER_ID_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fileID",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents double file identifiers; typed integer identifiers are independently gated and checked exactly.",
    }];
const FREAD_INTEGER_SIZE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sizeA",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents double size values; typed integer scalar and two-element size controls are independently gated.",
    }];
const FREAD_INTEGER_SKIP_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "skip",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents a double skip scalar; typed integer skip values are independently gated.",
    }];
const FREAD_INTEGER_PROTOTYPE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "prototype",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "A typed integer host or resident prototype selects the exact output class; this like form is a RunMat extension.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 5] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "A = fread(fileID, ..., integer_output_precision)",
        inputs: &[],
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Documented source=>output and *source precision forms produce the requested exact integer storage; the count output remains double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "A = fread(integer_fileID, ...)",
        inputs: &FREAD_INTEGER_ID_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The registry identifier is validated exactly as an integer.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "A = fread(fileID, integer_sizeA, ...)",
        inputs: &FREAD_INTEGER_SIZE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Scalar and two-element sizes are decoded exactly and bounded before allocation or file access.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "A = fread(fileID, ..., integer_skip, ...)",
        inputs: &FREAD_INTEGER_SKIP_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The skip is decoded exactly and bounded to the seek domain.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "A = fread(fileID, ..., \"like\", integer_prototype)",
        inputs: &FREAD_INTEGER_PROTOTYPE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "The prototype determines one of all eight integer output classes. Host prototypes return exact host storage; resident prototypes upload through their actual owning provider after saturating conversion and strict output validation.",
    },
];

const FREAD_OUTPUT_DATA: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Read data as numeric tensor or character array depending on precision.",
}];
const FREAD_INPUTS_FID: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File identifier opened by fopen.",
}];
const FREAD_INPUTS_FID_SIZE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "size",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"inf\""),
        description: "Element count or size vector ([m n]); supports \"inf\".",
    },
];
const FREAD_INPUTS_FID_SIZE_PRECISION: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "size",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"inf\""),
        description: "Element count or size vector ([m n]); supports \"inf\".",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"uint8=>double\""),
        description: "Read precision label (for example \"double\", \"uint8\", \"*char\").",
    },
];
const FREAD_INPUTS_FID_SIZE_PRECISION_SKIP: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "size",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"inf\""),
        description: "Element count or size vector ([m n]); supports \"inf\".",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label (for example \"double\", \"uint8\", \"*char\").",
    },
    BuiltinParamDescriptor {
        name: "skip",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Bytes skipped after each element read.",
    },
];
const FREAD_INPUTS_FID_SIZE_PRECISION_SKIP_MACHINEFMT: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "size",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"inf\""),
        description: "Element count or size vector ([m n]); supports \"inf\".",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label (for example \"double\", \"uint8\", \"*char\").",
    },
    BuiltinParamDescriptor {
        name: "skip",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Bytes skipped after each element read.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Machine format label (native/little-endian/big-endian aliases).",
    },
];
const FREAD_INPUTS_FID_PRECISION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label when size is omitted.",
    },
];
const FREAD_INPUTS_FID_PRECISION_SKIP: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label when size is omitted.",
    },
    BuiltinParamDescriptor {
        name: "skip",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Bytes skipped after each element read.",
    },
];
const FREAD_INPUTS_FID_PRECISION_MACHINEFMT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label when size is omitted.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Machine format label (native/little-endian/big-endian aliases).",
    },
];
const FREAD_INPUTS_FID_PRECISION_SKIP_MACHINEFMT: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label when size is omitted.",
    },
    BuiltinParamDescriptor {
        name: "skip",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Bytes skipped after each element read.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Machine format label (native/little-endian/big-endian aliases).",
    },
];
const FREAD_INPUTS_WITH_LIKE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "arg",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Positional fread arguments before the like clause.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Optional,
        default: Some("\"like\""),
        description: "Prototype keyword; currently only 'like'.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Prototype value controlling output class/residency.",
    },
];
const FREAD_SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "data = fread(fid)",
        inputs: &FREAD_INPUTS_FID,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, size)",
        inputs: &FREAD_INPUTS_FID_SIZE,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, size, precision)",
        inputs: &FREAD_INPUTS_FID_SIZE_PRECISION,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, size, precision, skip)",
        inputs: &FREAD_INPUTS_FID_SIZE_PRECISION_SKIP,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, size, precision, skip, machinefmt)",
        inputs: &FREAD_INPUTS_FID_SIZE_PRECISION_SKIP_MACHINEFMT,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, precision)",
        inputs: &FREAD_INPUTS_FID_PRECISION,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, precision, skip)",
        inputs: &FREAD_INPUTS_FID_PRECISION_SKIP,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, precision, machinefmt)",
        inputs: &FREAD_INPUTS_FID_PRECISION_MACHINEFMT,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, precision, skip, machinefmt)",
        inputs: &FREAD_INPUTS_FID_PRECISION_SKIP_MACHINEFMT,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, ..., \"like\", prototype)",
        inputs: &FREAD_INPUTS_WITH_LIKE,
        outputs: &FREAD_OUTPUT_DATA,
    },
];

const FREAD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREAD.INVALID_INPUT",
    identifier: Some("RunMat:fread:InvalidInput"),
    when: "Identifier/argument cardinality/type constraints are violated.",
    message: "fread: invalid input arguments",
};
const FREAD_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREAD.INVALID_IDENTIFIER",
    identifier: Some("RunMat:fread:InvalidIdentifier"),
    when: "Identifier does not refer to a readable open file.",
    message: "fread: invalid file identifier. Use fopen to generate a valid file ID.",
};
const FREAD_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREAD.INVALID_OPTION",
    identifier: Some("RunMat:fread:InvalidOption"),
    when: "Precision, skip, machine format, or like option values are invalid.",
    message: "fread: invalid option configuration",
};
const FREAD_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREAD.IO",
    identifier: Some("RunMat:fread:IoFailure"),
    when: "Read/seek or data-shape materialization fails.",
    message: "fread: file read failed",
};
const FREAD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREAD.INTERNAL",
    identifier: None,
    when: "Internal runtime control-flow conversion failed.",
    message: "fread: internal error",
};
const FREAD_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FREAD_ERROR_INVALID_INPUT,
    FREAD_ERROR_INVALID_IDENTIFIER,
    FREAD_ERROR_INVALID_OPTION,
    FREAD_ERROR_IO,
    FREAD_ERROR_INTERNAL,
];
pub const FREAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FREAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FREAD_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::fread")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fread",
    op_kind: GpuOpKind::Custom("file-io-read"),
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
        "Host-only operation that reads from the shared file registry; GPU arguments are gathered to the CPU before I/O.",
};

fn fread_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let detail = detail.strip_prefix("fread: ").unwrap_or(detail);
    fread_error_with_message(format!("{}: {}", error.message, detail), error)
}

fn fread_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FREAD_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_string_result<T>(
    result: Result<T, String>,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<T> {
    result.map_err(|detail| fread_error_with_detail(error, detail))
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::fread")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fread",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O cannot participate in fusion; metadata is registered for completeness.",
};

#[runtime_builtin(
    name = "fread",
    category = "io/filetext",
    summary = "Read binary data from file identifiers.",
    keywords = "fread,file,io,binary,precision",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::fread_type),
    descriptor(crate::builtins::io::filetext::fread::FREAD_DESCRIPTOR),
    extensions(crate::builtins::io::filetext::fread::FREAD_EXTENSIONS),
    integer_capabilities(crate::builtins::io::filetext::fread::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::io::filetext::fread"
)]
async fn fread_builtin(fid: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&fid, &rest).await?;
    Ok(eval.first_output())
}

#[derive(Debug, Clone)]
pub struct FreadEval {
    data: Value,
    count: usize,
}

impl FreadEval {
    fn new(data: Value, count: usize) -> Self {
        Self { data, count }
    }

    pub fn first_output(&self) -> Value {
        self.data.clone()
    }

    pub fn outputs(&self) -> Vec<Value> {
        vec![self.data.clone(), Value::Num(self.count as f64)]
    }

    fn apply_like(
        &mut self,
        like_proto: Option<&Value>,
        precision: PrecisionSpec,
    ) -> BuiltinResult<()> {
        if let Some(proto) = like_proto {
            let adjusted = adjust_output_for_like(self.data.clone(), proto, precision)?;
            self.data = adjusted;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn data(&self) -> &Value {
        &self.data
    }

    #[cfg(test)]
    pub(crate) fn count(&self) -> usize {
        self.count
    }
}

pub async fn evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FreadEval> {
    let raw_refs: Vec<&Value> = rest.iter().collect();
    let (raw_size, raw_precision, raw_skip, _, raw_like) =
        map_string_result(classify_arguments(&raw_refs), &FREAD_ERROR_INVALID_INPUT)?;
    if raw_like.is_some() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FREAD_LIKE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if let Some(prototype) = raw_like {
        let precision =
            map_string_result(parse_precision(raw_precision), &FREAD_ERROR_INVALID_OPTION)?;
        map_string_result(
            validate_like_prototype(prototype),
            &FREAD_ERROR_INVALID_OPTION,
        )?;
        map_string_result(
            validate_like_output_kind(prototype, precision.output),
            &FREAD_ERROR_INVALID_OPTION,
        )?;
    }
    preflight_control(fid_value, ControlRole::FileId)?;
    if let Some(value) = raw_size {
        preflight_control(value, ControlRole::Size)?;
    }
    if let Some(value) = raw_skip {
        preflight_control(value, ControlRole::Skip)?;
    }
    let fid_host = gather_value(fid_value).await?;
    let fid = map_string_result(parse_fid(&fid_host), &FREAD_ERROR_INVALID_INPUT)?;
    if fid < 0 {
        return Err(fread_error_with_detail(
            &FREAD_ERROR_INVALID_INPUT,
            "file identifier must be non-negative",
        ));
    }
    if fid < 3 {
        return Err(fread_error_with_detail(
            &FREAD_ERROR_INVALID_INPUT,
            "standard input/output identifiers are not supported yet",
        ));
    }

    let info = registry::info_for(fid).ok_or_else(|| {
        fread_error_with_message(
            FREAD_ERROR_INVALID_IDENTIFIER.message,
            &FREAD_ERROR_INVALID_IDENTIFIER,
        )
    })?;
    if !info.access.read {
        return Err(fread_error_with_detail(
            &FREAD_ERROR_INVALID_IDENTIFIER,
            "file identifier is not open for reading",
        ));
    }
    let handle = registry::shared_handle(fid).ok_or_else(|| {
        fread_error_with_message(
            FREAD_ERROR_INVALID_IDENTIFIER.message,
            &FREAD_ERROR_INVALID_IDENTIFIER,
        )
    })?;

    let arg_refs: Vec<&Value> = rest.iter().collect();
    let (size_arg, precision_arg, skip_arg, machine_arg, like_arg) =
        map_string_result(classify_arguments(&arg_refs), &FREAD_ERROR_INVALID_INPUT)?;
    let size_host = match size_arg {
        Some(value) => Some(gather_value(value).await?),
        None => None,
    };
    let precision_host = match precision_arg {
        Some(value) => Some(gather_value(value).await?),
        None => None,
    };
    let skip_host = match skip_arg {
        Some(value) => Some(gather_value(value).await?),
        None => None,
    };
    let machine_host = match machine_arg {
        Some(value) => Some(gather_value(value).await?),
        None => None,
    };

    let size_spec = map_string_result(parse_size(size_host.as_ref()), &FREAD_ERROR_INVALID_INPUT)?;
    let precision = map_string_result(
        parse_precision(precision_host.as_ref()),
        &FREAD_ERROR_INVALID_OPTION,
    )?;
    let skip_bytes =
        map_string_result(parse_skip(skip_host.as_ref()), &FREAD_ERROR_INVALID_OPTION)?;
    let machine_format = map_string_result(
        parse_machine_format(machine_host.as_ref(), &info.machinefmt),
        &FREAD_ERROR_INVALID_OPTION,
    )?;

    let mut guard = handle.lock().map_err(|_| {
        fread_error_with_detail(
            &FREAD_ERROR_INTERNAL,
            "failed to lock file handle (poisoned mutex)",
        )
    })?;
    let file = guard.as_mut().ok_or_else(|| {
        fread_error_with_message(
            FREAD_ERROR_INVALID_IDENTIFIER.message,
            &FREAD_ERROR_INVALID_IDENTIFIER,
        )
    })?;

    let (mut eval, encountered_eof) = map_string_result(
        read_from_handle(file, &size_spec, &precision, skip_bytes, machine_format),
        &FREAD_ERROR_IO,
    )?;
    drop(guard);
    if encountered_eof {
        registry::mark_eof_encountered(fid);
    }
    eval.apply_like(like_arg, precision)?;
    Ok(eval)
}

fn validate_like_prototype(prototype: &Value) -> Result<(), String> {
    match prototype {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::provider_for_handle(handle)
                .is_none_or(|owner| owner.device_id() != handle.device_id())
            {
                return Err(
                    "fread: resident 'like' prototype has no registered owning provider"
                        .to_string(),
                );
            }
            if runmat_accelerate_api::handle_storage(handle)
                != runmat_accelerate_api::GpuTensorStorage::Real
            {
                return Err(
                    "fread: complex resident 'like' prototypes are not supported".to_string(),
                );
            }
            let integer = runmat_accelerate_api::handle_integer_type(handle).is_some();
            let logical = runmat_accelerate_api::handle_is_logical(handle);
            if !integer && !logical && runmat_accelerate_api::handle_precision(handle).is_none() {
                return Err(
                    "fread: resident 'like' prototype is missing precision metadata".to_string(),
                );
            }
            Ok(())
        }
        Value::ComplexTensor(_) | Value::Complex(_, _) => {
            Err("fread: complex prototypes are not supported yet".to_string())
        }
        Value::Cell(_) => Err("fread: cell prototypes are not supported".to_string()),
        Value::LogicalArray(_)
        | Value::Bool(_)
        | Value::CharArray(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::Tensor(_)
        | Value::Int(_)
        | Value::Num(_) => Ok(()),
        _ => Err("fread: unsupported 'like' prototype".to_string()),
    }
}

fn validate_like_output_kind(prototype: &Value, output: OutputKind) -> Result<(), String> {
    let character_prototype = matches!(
        prototype,
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_)
    );
    match (character_prototype, output) {
        (true, OutputKind::Char) | (false, OutputKind::Numeric(_)) => Ok(()),
        (true, OutputKind::Numeric(_)) => Err(
            "fread: character prototypes require a character precision such as '*char'".to_string(),
        ),
        (false, OutputKind::Char) => {
            Err("fread: character output requires a character prototype".to_string())
        }
    }
}

async fn gather_value(value: &Value) -> BuiltinResult<Value> {
    gather_if_needed_async(value)
        .await
        .map_err(map_control_flow)
}

#[derive(Clone, Copy)]
enum ControlRole {
    FileId,
    Size,
    Skip,
}

fn preflight_control(value: &Value, role: ControlRole) -> BuiltinResult<()> {
    let integer_extension = match role {
        ControlRole::FileId => &FREAD_INTEGER_ID_EXTENSION,
        ControlRole::Size => &FREAD_INTEGER_SIZE_EXTENSION,
        ControlRole::Skip => &FREAD_INTEGER_SKIP_EXTENSION,
    };
    match value {
        Value::Int(_) => {
            crate::compatibility::ensure_builtin_extension_enabled(integer_extension, BUILTIN_NAME)
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            crate::compatibility::ensure_builtin_extension_enabled(integer_extension, BUILTIN_NAME)
        }
        Value::Bool(_) | Value::LogicalArray(_) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FREAD_LOGICAL_CONTROL_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FREAD_SINGLE_CONTROL_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FREAD_RESIDENT_CONTROL_EXTENSION,
                BUILTIN_NAME,
            )?;
            if runmat_accelerate_api::handle_is_logical(handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FREAD_LOGICAL_CONTROL_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    integer_extension,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FREAD_SINGLE_CONTROL_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn parse_fid(value: &Value) -> Result<i32, String> {
    let number = match value {
        Value::Num(n) => *n,
        Value::Int(int) => {
            return int
                .try_to_i32()
                .ok_or_else(|| "file identifier is out of range".to_string());
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                return int
                    .try_to_i32()
                    .ok_or_else(|| "file identifier is out of range".to_string());
            }
            tensor::tensor_value_f64(t, 0)
        }
        _ => {
            return Err("file identifier must be numeric".to_string());
        }
    };
    if !number.is_finite() {
        return Err("file identifier must be finite".to_string());
    }
    let rounded = number.round();
    if (rounded - number).abs() > f64::EPSILON {
        return Err("file identifier must be an integer".to_string());
    }
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err("file identifier is out of range".to_string());
    }
    Ok(rounded as i32)
}

type ClassifiedArgs<'a> = (
    Option<&'a Value>,
    Option<&'a Value>,
    Option<&'a Value>,
    Option<&'a Value>,
    Option<&'a Value>,
);

fn classify_arguments<'a>(args: &'a [&'a Value]) -> Result<ClassifiedArgs<'a>, String> {
    let mut filtered_indices: Vec<usize> = Vec::with_capacity(args.len());
    let mut like_proto: Option<&Value> = None;
    let mut i = 0usize;

    while i < args.len() {
        let value = args[i];
        if matches_keyword(value, "like") {
            if like_proto.is_some() {
                return Err("multiple 'like' prototypes are not supported".to_string());
            }
            i += 1;
            let Some(proto_value) = args.get(i) else {
                return Err("expected prototype after 'like'".to_string());
            };
            like_proto = Some(*proto_value);
            i += 1;
            continue;
        }
        filtered_indices.push(i);
        i += 1;
    }

    if filtered_indices.len() > 4 {
        return Err("too many input arguments".to_string());
    }

    let (size_idx, precision_idx, skip_idx, machine_idx) =
        classify_ordered_indices(args, &filtered_indices)?;

    let size = size_idx.map(|index| args[index]);
    let precision = precision_idx.map(|index| args[index]);
    let skip = skip_idx.map(|index| args[index]);
    let machine = machine_idx.map(|index| args[index]);

    Ok((size, precision, skip, machine, like_proto))
}

type ClassifiedIndices = (Option<usize>, Option<usize>, Option<usize>, Option<usize>);

fn classify_ordered_indices(
    args: &[&Value],
    indices: &[usize],
) -> Result<ClassifiedIndices, String> {
    let mut position = 0usize;
    let mut size_idx: Option<usize> = None;
    let mut precision_idx: Option<usize> = None;
    let mut skip_idx: Option<usize> = None;
    let mut machine_idx: Option<usize> = None;

    if let Some(&first_index) = indices.get(position) {
        let first = args[first_index];
        if is_string_like(first) {
            precision_idx = Some(first_index);
        } else {
            size_idx = Some(first_index);
        }
        position += 1;
    }

    if let Some(&index) = indices.get(position) {
        let candidate = args[index];
        if precision_idx.is_none() && is_string_like(candidate) {
            precision_idx = Some(index);
            position += 1;
        } else if is_numeric_like(candidate) {
            skip_idx = Some(index);
            position += 1;
        } else if is_string_like(candidate) {
            machine_idx = Some(index);
            position += 1;
        } else {
            return Err("invalid argument combination".to_string());
        }
    }

    if let Some(&index) = indices.get(position) {
        let candidate = args[index];
        if skip_idx.is_none() && is_numeric_like(candidate) {
            skip_idx = Some(index);
            position += 1;
        } else if machine_idx.is_none() && is_string_like(candidate) {
            machine_idx = Some(index);
            position += 1;
        } else {
            return Err("invalid argument combination".to_string());
        }
    }

    if let Some(&index) = indices.get(position) {
        let candidate = args[index];
        if machine_idx.is_none() && is_string_like(candidate) {
            machine_idx = Some(index);
            position += 1;
        } else {
            return Err("too many input arguments".to_string());
        }
    }

    if position < indices.len() {
        return Err("too many input arguments".to_string());
    }

    Ok((size_idx, precision_idx, skip_idx, machine_idx))
}

fn is_string_like(value: &Value) -> bool {
    match value {
        Value::String(_) => true,
        Value::CharArray(ca) if ca.rows == 1 => true,
        Value::StringArray(sa) if sa.data.len() == 1 => true,
        _ => false,
    }
}

fn matches_keyword(value: &Value, keyword: &str) -> bool {
    extract_scalar_string(value)
        .map(|text| text.eq_ignore_ascii_case(keyword))
        .unwrap_or(false)
}

fn is_numeric_like(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::LogicalArray(_)
            | Value::GpuTensor(_)
    )
}

#[derive(Clone, Debug)]
enum SizeSpec {
    All,
    Count(usize),
    Matrix { rows: usize, cols: Option<usize> },
}

impl SizeSpec {
    fn element_limit(&self) -> Option<usize> {
        match self {
            SizeSpec::All => None,
            SizeSpec::Count(n) => Some(*n),
            SizeSpec::Matrix { rows, cols } => {
                if *rows == 0 {
                    return Some(0);
                }
                cols.map(|c| rows.saturating_mul(c))
            }
        }
    }
}

fn parse_size(arg: Option<&Value>) -> Result<SizeSpec, String> {
    match arg {
        None => Ok(SizeSpec::All),
        Some(Value::String(s)) => parse_size_string(s),
        Some(Value::CharArray(ca)) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            parse_size_string(&text)
        }
        Some(Value::StringArray(sa)) if sa.data.len() == 1 => parse_size_string(&sa.data[0]),
        Some(Value::Tensor(t)) => parse_size_tensor(t),
        Some(Value::Int(int)) => Ok(SizeSpec::Count(int_to_usize(
            int,
            "size argument must be a non-negative integer",
        )?)),
        Some(value) => {
            let scalar = value_to_scalar(value, "size argument must be numeric or a size vector")?;
            scalar_to_size(scalar)
        }
    }
}

fn parse_size_string(text: &str) -> Result<SizeSpec, String> {
    if text.trim().is_empty() {
        return Err("size argument must not be empty".to_string());
    }
    let lower = text.trim().to_ascii_lowercase();
    if lower == "inf" {
        Ok(SizeSpec::All)
    } else {
        let number = lower
            .parse::<f64>()
            .map_err(|_| "size argument must be numeric or 'inf'".to_string())?;
        scalar_to_size(number)
    }
}

fn parse_size_tensor(t: &Tensor) -> Result<SizeSpec, String> {
    if let Some(storage) = t.integer_storage() {
        return match storage.len() {
            0 => Ok(SizeSpec::Count(0)),
            1 => {
                let value = storage.value_at(0).expect("one-element integer storage");
                Ok(SizeSpec::Count(value.try_to_usize().ok_or_else(|| {
                    "size argument must be a non-negative integer".to_string()
                })?))
            }
            2 => {
                let rows_value = storage
                    .value_at(0)
                    .expect("integer storage length matches tensor length");
                let cols_value = storage
                    .value_at(1)
                    .expect("integer storage length matches tensor length");
                let rows = rows_value.try_to_usize().ok_or_else(|| {
                    "size vector components must be non-negative integers or Inf".to_string()
                })?;
                let cols = cols_value.try_to_usize().ok_or_else(|| {
                    "size vector components must be non-negative integers or Inf".to_string()
                })?;
                Ok(SizeSpec::Matrix {
                    rows,
                    cols: Some(cols),
                })
            }
            _ => Err("size vector must contain at most two elements".to_string()),
        };
    }

    match t.len() {
        0 => Ok(SizeSpec::Count(0)),
        1 => scalar_to_size(tensor::tensor_value_f64(t, 0)),
        2 => {
            let rows = scalar_to_size_component(
                tensor::tensor_value_f64(t, 0),
                "size vector components must be non-negative integers or Inf",
            )?;
            let cols_raw = tensor::tensor_value_f64(t, 1);
            if cols_raw.is_infinite() && cols_raw.is_sign_positive() {
                Ok(SizeSpec::Matrix { rows, cols: None })
            } else {
                let cols = scalar_to_size_component(
                    cols_raw,
                    "size vector components must be non-negative integers or Inf",
                )?;
                Ok(SizeSpec::Matrix {
                    rows,
                    cols: Some(cols),
                })
            }
        }
        _ => Err("size vector must contain at most two elements".to_string()),
    }
}

fn scalar_to_size(value: f64) -> Result<SizeSpec, String> {
    if value.is_infinite() && value.is_sign_positive() {
        return Ok(SizeSpec::All);
    }
    let count = scalar_to_size_component(value, "size argument must be a non-negative integer")?;
    Ok(SizeSpec::Count(count))
}

fn scalar_to_size_component(value: f64, err: &str) -> Result<usize, String> {
    if !value.is_finite() {
        return Err(err.to_string());
    }
    if value < 0.0 {
        return Err(err.to_string());
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(err.to_string());
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err("size argument is too large".to_string());
    }
    Ok(rounded as usize)
}

#[derive(Clone, Copy, Debug)]
enum InputType {
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    UInt64,
    Int64,
    Float32,
    Float64,
}

impl InputType {
    fn byte_len(&self) -> usize {
        match self {
            InputType::UInt8 | InputType::Int8 => 1,
            InputType::UInt16 | InputType::Int16 => 2,
            InputType::UInt32 | InputType::Int32 | InputType::Float32 => 4,
            InputType::UInt64 | InputType::Int64 | InputType::Float64 => 8,
        }
    }

    fn numeric_dtype(self) -> NumericDType {
        match self {
            Self::UInt8 => NumericDType::U8,
            Self::Int8 => NumericDType::I8,
            Self::UInt16 => NumericDType::U16,
            Self::Int16 => NumericDType::I16,
            Self::UInt32 => NumericDType::U32,
            Self::Int32 => NumericDType::I32,
            Self::UInt64 => NumericDType::U64,
            Self::Int64 => NumericDType::I64,
            Self::Float32 => NumericDType::F32,
            Self::Float64 => NumericDType::F64,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum OutputKind {
    Numeric(NumericDType),
    Char,
}

#[derive(Clone, Copy, Debug)]
struct PrecisionSpec {
    input: InputType,
    output: OutputKind,
    repeat: usize,
}

impl PrecisionSpec {
    fn default() -> Self {
        Self {
            input: InputType::UInt8,
            output: OutputKind::Numeric(NumericDType::F64),
            repeat: 1,
        }
    }
}

fn parse_precision(arg: Option<&Value>) -> Result<PrecisionSpec, String> {
    match arg {
        None => Ok(PrecisionSpec::default()),
        Some(value) => {
            let text = scalar_string(
                value,
                "precision argument must be a string scalar or character vector",
            )?;
            parse_precision_string(&text)
        }
    }
}

fn parse_precision_string(raw: &str) -> Result<PrecisionSpec, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("precision argument must not be empty".to_string());
    }
    let lower = trimmed.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix('*') {
        parse_star_precision(rest.trim())
    } else if let Some((lhs, rhs)) = lower.split_once("=>") {
        let (repeat, input) = parse_repeated_input_label(lhs.trim())?;
        let output = parse_output_label(rhs.trim())?;
        Ok(PrecisionSpec {
            input,
            output,
            repeat,
        })
    } else {
        let (repeat, input) = parse_repeated_input_label(lower.trim())?;
        let wants_char =
            lower == "char" || (matches!(input, InputType::UInt8) && lower.contains("char"));
        let output = if wants_char {
            OutputKind::Char
        } else {
            OutputKind::Numeric(NumericDType::F64)
        };
        Ok(PrecisionSpec {
            input,
            output,
            repeat,
        })
    }
}

fn parse_star_precision(label: &str) -> Result<PrecisionSpec, String> {
    if label == "char" {
        return Ok(PrecisionSpec {
            input: InputType::UInt8,
            output: OutputKind::Char,
            repeat: 1,
        });
    }
    let input = parse_input_label(label)?;
    Ok(PrecisionSpec {
        input,
        output: OutputKind::Numeric(input.numeric_dtype()),
        repeat: 1,
    })
}

fn parse_repeated_input_label(label: &str) -> Result<(usize, InputType), String> {
    if let Some((repeat, source)) = label.split_once('*') {
        if repeat.trim().chars().all(|ch| ch.is_ascii_digit()) {
            let repeat = repeat
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|&repeat| repeat > 0)
                .ok_or_else(|| "precision repeat count must be a positive integer".to_string())?;
            return Ok((repeat, parse_input_label(source.trim())?));
        }
    }
    Ok((1, parse_input_label(label)?))
}

fn parse_input_label(label: &str) -> Result<InputType, String> {
    match label {
        "double" | "float64" | "real*8" => Ok(InputType::Float64),
        "single" | "float" | "float32" | "real*4" => Ok(InputType::Float32),
        "int8" | "schar" | "signedchar" | "signed char" | "integer*1" => Ok(InputType::Int8),
        "uint8" | "uchar" | "unsignedchar" | "unsigned char" | "char" | "byte" => {
            Ok(InputType::UInt8)
        }
        "int16" | "short" | "integer*2" => Ok(InputType::Int16),
        "uint16" | "ushort" | "unsignedshort" | "unsigned short" => Ok(InputType::UInt16),
        "int32" | "int" | "integer*4" | "long" => Ok(InputType::Int32),
        "uint32" | "uint" | "unsignedint" | "unsigned int" | "unsignedlong" | "unsigned long" => {
            Ok(InputType::UInt32)
        }
        "int64" | "integer*8" | "longlong" | "long long" => Ok(InputType::Int64),
        "uint64" | "unsignedlonglong" | "unsigned long long" => Ok(InputType::UInt64),
        other => Err(format!("unsupported precision '{other}'")),
    }
}

fn parse_output_label(label: &str) -> Result<OutputKind, String> {
    match label {
        "double" | "float64" | "real*8" => Ok(OutputKind::Numeric(NumericDType::F64)),
        "single" | "float" | "float32" | "real*4" => Ok(OutputKind::Numeric(NumericDType::F32)),
        "int8" | "schar" | "signedchar" | "signed char" | "integer*1" => {
            Ok(OutputKind::Numeric(NumericDType::I8))
        }
        "int16" | "short" | "integer*2" => Ok(OutputKind::Numeric(NumericDType::I16)),
        "int32" | "int" | "integer*4" | "long" => Ok(OutputKind::Numeric(NumericDType::I32)),
        "int64" | "integer*8" | "longlong" | "long long" => {
            Ok(OutputKind::Numeric(NumericDType::I64))
        }
        "uint8" | "uchar" | "unsignedchar" | "unsigned char" => {
            Ok(OutputKind::Numeric(NumericDType::U8))
        }
        "uint16" | "ushort" | "unsignedshort" | "unsigned short" => {
            Ok(OutputKind::Numeric(NumericDType::U16))
        }
        "uint32" | "uint" | "unsignedint" | "unsigned int" | "unsignedlong" | "unsigned long" => {
            Ok(OutputKind::Numeric(NumericDType::U32))
        }
        "uint64" | "unsignedlonglong" | "unsigned long long" => {
            Ok(OutputKind::Numeric(NumericDType::U64))
        }
        "char" => Ok(OutputKind::Char),
        other => Err(format!("output class '{other}' is not implemented yet")),
    }
}

fn parse_skip(arg: Option<&Value>) -> Result<usize, String> {
    match arg {
        None => Ok(0),
        Some(Value::Int(int)) => int_to_skip(int),
        Some(Value::Tensor(t)) if tensor::is_scalar_tensor(t) => {
            if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                return int_to_skip(&int);
            }
            parse_skip_scalar(tensor::tensor_value_f64(t, 0))
        }
        Some(value) => {
            let scalar = value_to_scalar(value, "skip value must be numeric")?;
            parse_skip_scalar(scalar)
        }
    }
}

fn parse_skip_scalar(scalar: f64) -> Result<usize, String> {
    if !scalar.is_finite() {
        return Err("skip value must be finite".to_string());
    }
    if scalar < 0.0 {
        return Err("skip value must be non-negative".to_string());
    }
    let rounded = scalar.round();
    if (rounded - scalar).abs() > f64::EPSILON {
        return Err("skip value must be an integer".to_string());
    }
    if rounded >= i64::MAX as f64 {
        return Err("skip value is too large".to_string());
    }
    Ok(rounded as usize)
}

fn int_to_skip(value: &IntValue) -> Result<usize, String> {
    let skip = int_to_usize(value, "skip value must be non-negative")?;
    if skip > i64::MAX as usize {
        return Err("skip value is too large".to_string());
    }
    Ok(skip)
}

fn int_to_usize(value: &IntValue, err: &str) -> Result<usize, String> {
    value.try_to_usize().ok_or_else(|| err.to_string())
}

#[derive(Clone, Copy, Debug)]
enum MachineFormat {
    Native,
    LittleEndian,
    BigEndian,
}

#[derive(Clone, Copy, Debug)]
enum Endianness {
    Little,
    Big,
}

impl MachineFormat {
    fn to_endianness(self) -> Endianness {
        match self {
            MachineFormat::Native => {
                if cfg!(target_endian = "little") {
                    Endianness::Little
                } else {
                    Endianness::Big
                }
            }
            MachineFormat::LittleEndian => Endianness::Little,
            MachineFormat::BigEndian => Endianness::Big,
        }
    }
}

fn parse_machine_format(arg: Option<&Value>, default_label: &str) -> Result<MachineFormat, String> {
    match arg {
        Some(value) => {
            let text = scalar_string(
                value,
                "machine format must be a string scalar or character vector",
            )?;
            machine_format_from_label(&text)
        }
        None => machine_format_from_label(default_label),
    }
}

fn machine_format_from_label(label: &str) -> Result<MachineFormat, String> {
    let trimmed = label.trim();
    if trimmed.is_empty() {
        return Err("machine format must not be empty".to_string());
    }
    let lower = trimmed.to_ascii_lowercase();
    let collapsed: String = lower
        .chars()
        .filter(|c| !matches!(c, '-' | '_' | ' '))
        .collect();
    if matches!(collapsed.as_str(), "native" | "n" | "system" | "default") {
        return Ok(MachineFormat::Native);
    }
    if matches!(
        collapsed.as_str(),
        "l" | "le" | "littleendian" | "pc" | "intel"
    ) {
        return Ok(MachineFormat::LittleEndian);
    }
    if matches!(
        collapsed.as_str(),
        "b" | "be" | "bigendian" | "mac" | "motorola"
    ) {
        return Ok(MachineFormat::BigEndian);
    }
    if lower.starts_with("ieee-le") {
        return Ok(MachineFormat::LittleEndian);
    }
    if lower.starts_with("ieee-be") {
        return Ok(MachineFormat::BigEndian);
    }
    Err(format!("unsupported machine format '{trimmed}'"))
}

fn scalar_string(value: &Value, err: &str) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(err.to_string()),
    }
}

fn value_to_scalar(value: &Value, err: &str) -> Result<f64, String> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(int) => Ok(int.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Ok(tensor::tensor_value_f64(t, 0)),
        Value::LogicalArray(la) if la.data.len() == 1 => {
            Ok(if la.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => Err(err.to_string()),
    }
}

fn read_from_handle(
    file: &mut File,
    size_spec: &SizeSpec,
    precision: &PrecisionSpec,
    skip: usize,
    machine: MachineFormat,
) -> Result<(FreadEval, bool), String> {
    let endianness = machine.to_endianness();
    match precision.output {
        OutputKind::Numeric(output_dtype) => {
            let limit = size_spec.element_limit();
            let (values, count, encountered_eof) = read_numeric_values(
                file,
                precision.input,
                precision.repeat,
                limit,
                skip,
                endianness,
            )?;
            let (data, rows, cols) = finalize_numeric(size_spec, count, values);
            let tensor = Tensor::from_numeric_storage(data, vec![rows, cols])
                .map_err(|e| format!("fread: {e}"))?;
            let tensor = if tensor.numeric_dtype() == output_dtype {
                tensor
            } else {
                tensor::coerce_tensor_dtype(tensor, output_dtype)
            };
            Ok((
                FreadEval::new(Value::Tensor(tensor), count),
                encountered_eof,
            ))
        }
        OutputKind::Char => {
            let limit = size_spec.element_limit();
            let (values, count, encountered_eof) = read_char_values(
                file,
                precision.input,
                precision.repeat,
                limit,
                skip,
                endianness,
            )?;
            let (row_major, rows, cols) = finalize_char(size_spec, count, values);
            let char_array =
                CharArray::new(row_major, rows, cols).map_err(|e| format!("fread: {e}"))?;
            Ok((
                FreadEval::new(Value::CharArray(char_array), count),
                encountered_eof,
            ))
        }
    }
}

fn adjust_output_for_like(
    data: Value,
    prototype: &Value,
    precision: PrecisionSpec,
) -> BuiltinResult<Value> {
    if let Value::GpuTensor(prototype) = prototype {
        return match data {
            Value::Tensor(tensor) => tensor_to_gpu_value(tensor, prototype),
            Value::CharArray(_) => Err(fread_error_with_detail(
                &FREAD_ERROR_INVALID_OPTION,
                "character output cannot be returned on the GPU via 'like'",
            )),
            other => Ok(other),
        };
    }

    let result = (|| -> Result<Value, String> {
        match prototype {
            Value::LogicalArray(_) | Value::Bool(_) => convert_to_logical_value(data),
            Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
                if !matches!(precision.output, OutputKind::Char) {
                    return Err(
                        "fread: character prototypes require a character precision such as '*char'"
                            .to_string(),
                    );
                }
                ensure_char_result(data)
            }
            Value::Tensor(tensor) => tensor_to_numeric_like(data, tensor.numeric_dtype()),
            Value::Int(value) => tensor_to_numeric_like(
                data,
                IntegerStorage::from_scalar(value.clone()).numeric_dtype(),
            ),
            Value::Num(_) => tensor_to_numeric_like(data, NumericDType::F64),
            Value::ComplexTensor(_) | Value::Complex(_, _) => {
                Err("fread: complex prototypes are not supported yet".to_string())
            }
            Value::Cell(_) => Err("fread: cell prototypes are not supported".to_string()),
            _ => Ok(data),
        }
    })();
    map_string_result(result, &FREAD_ERROR_INVALID_OPTION)
}

fn tensor_to_numeric_like(data: Value, dtype: NumericDType) -> Result<Value, String> {
    match data {
        Value::Tensor(tensor) => Ok(Value::Tensor(tensor::coerce_tensor_dtype(tensor, dtype))),
        Value::CharArray(_) => Err(
            "fread: character output cannot be converted to a numeric 'like' prototype".to_string(),
        ),
        other => Ok(other),
    }
}

fn ensure_char_result(data: Value) -> Result<Value, String> {
    match data {
        Value::CharArray(_) => Ok(data),
        _ => Err("fread: expected character output when using a character prototype".to_string()),
    }
}

fn tensor_to_gpu_value(
    tensor: Tensor,
    prototype: &runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<Value> {
    use runmat_accelerate_api::{GpuTensorStorage, IntegerElementType, ProviderPrecision};

    let fail = |message: &str| fread_error_with_detail(&FREAD_ERROR_INVALID_OPTION, message);
    if prototype.physical().storage() != GpuTensorStorage::Real {
        return Err(fail("complex resident 'like' prototypes are not supported"));
    }
    let provider = gpu_helpers::exact_provider_for_handle(prototype)
        .ok_or_else(|| fail("resident 'like' prototype has no registered owning provider"))?;
    if prototype.view().is_logical() {
        let bits = (0..tensor.len())
            .map(|index| {
                u8::from(
                    tensor
                        .numeric_value_at(index)
                        .expect("validated tensor index")
                        .materialize_f64()
                        != 0.0,
                )
            })
            .collect::<Vec<_>>();
        let output = runmat_accelerate_api::upload_logical_protecting(
            provider,
            &runmat_accelerate_api::HostLogicalTensorView {
                data: &bits,
                shape: &tensor.shape,
            },
            &[prototype],
        )
        .map_err(|error| {
            crate::runtime_error::with_provider_execution_source(
                fail("logical Like upload failed"),
                runmat_accelerate_api::ProviderExecutionError::from_source(error),
            )
        })?;
        return Ok(Value::GpuTensor(output));
    }
    let expected_integer = runmat_accelerate_api::handle_integer_type(prototype);
    let expected_logical = false;
    let expected_precision = runmat_accelerate_api::handle_precision(prototype);
    let tensor = if let Some(integer) = expected_integer {
        let dtype = match integer {
            IntegerElementType::I8 => NumericDType::I8,
            IntegerElementType::I16 => NumericDType::I16,
            IntegerElementType::I32 => NumericDType::I32,
            IntegerElementType::I64 => NumericDType::I64,
            IntegerElementType::U8 => NumericDType::U8,
            IntegerElementType::U16 => NumericDType::U16,
            IntegerElementType::U32 => NumericDType::U32,
            IntegerElementType::U64 => NumericDType::U64,
        };
        tensor::coerce_tensor_dtype(tensor, dtype)
    } else {
        let dtype = match expected_precision {
            Some(ProviderPrecision::F32) => NumericDType::F32,
            Some(ProviderPrecision::F64) => NumericDType::F64,
            None => {
                return Err(fail(
                    "resident 'like' prototype is missing precision metadata",
                ))
            }
        };
        tensor::coerce_tensor_dtype(tensor, dtype)
    };
    let expected_shape = tensor.shape.clone();
    let output = gpu_helpers::upload_tensor_protecting(provider, &tensor, &[prototype]).map_err(
        |error| {
            crate::runtime_error::with_provider_execution_source(
                fail("prototype owner upload failed"),
                runmat_accelerate_api::ProviderExecutionError::from_source(error),
            )
        },
    )?;
    if let Err(message) = validate_resident_like_output(
        prototype,
        &output,
        &expected_shape,
        provider,
        expected_precision,
        expected_integer,
        expected_logical,
    ) {
        return Err(gpu_helpers::reject_gpu_output(output, fail(&message)));
    }
    Ok(Value::GpuTensor(output))
}

fn validate_resident_like_output(
    prototype: &runmat_accelerate_api::GpuTensorHandle,
    output: &runmat_accelerate_api::GpuTensorHandle,
    expected_shape: &[usize],
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    expected_precision: Option<runmat_accelerate_api::ProviderPrecision>,
    expected_integer: Option<runmat_accelerate_api::IntegerElementType>,
    expected_logical: bool,
) -> Result<(), String> {
    let aliases_prototype = output.shares_allocation_with(prototype);
    let valid = !aliases_prototype
        && output.device_id() == prototype.device_id()
        && output.shape() == expected_shape
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && runmat_accelerate_api::handle_storage(output)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(output) == expected_precision
        && runmat_accelerate_api::handle_integer_type(output) == expected_integer
        && runmat_accelerate_api::handle_is_logical(output) == expected_logical;
    if valid {
        return Ok(());
    }
    Err("fread: resident 'like' upload returned an invalid or aliased handle".to_string())
}

fn convert_to_logical_value(data: Value) -> Result<Value, String> {
    match data {
        Value::LogicalArray(_) => Ok(data),
        Value::Tensor(tensor) => {
            let bits = (0..tensor.len())
                .map(|index| {
                    let value = tensor
                        .numeric_value_at(index)
                        .expect("validated fread numeric tensor index");
                    match value {
                        NumericScalar::F64(value) => u8::from(value != 0.0),
                        NumericScalar::F32(value) => u8::from(value != 0.0),
                        value => u8::from(
                            !value
                                .into_int_value()
                                .expect("non-floating numeric scalar is integer")
                                .is_zero(),
                        ),
                    }
                })
                .collect();
            LogicalArray::new(bits, tensor.shape.clone())
                .map(Value::LogicalArray)
                .map_err(|e| format!("fread: {e}"))
        }
        Value::CharArray(ca) => {
            let total = ca.rows.saturating_mul(ca.cols);
            let mut bits = Vec::with_capacity(total);
            for c in 0..ca.cols {
                for r in 0..ca.rows {
                    let idx = r * ca.cols + c;
                    let ch = ca.data[idx];
                    bits.push(if ch != '\0' { 1 } else { 0 });
                }
            }
            LogicalArray::new(bits, vec![ca.rows, ca.cols])
                .map(Value::LogicalArray)
                .map_err(|e| format!("fread: {e}"))
        }
        _ => Err(
            "fread: logical prototypes require numeric or character output from the read"
                .to_string(),
        ),
    }
}

fn read_numeric_values<R: Read + Seek>(
    reader: &mut R,
    input: InputType,
    repeat: usize,
    limit: Option<usize>,
    skip: usize,
    endianness: Endianness,
) -> Result<(NumericStorage, usize, bool), String> {
    if let Some(0) = limit {
        return Ok((NumericStorage::zeros(input.numeric_dtype(), 0), 0, false));
    }
    let element_size = input.byte_len();
    let mut buffer = vec![0u8; element_size];
    let mut values = Vec::new();
    let mut count = 0usize;
    let target = limit.unwrap_or(usize::MAX);
    let mut encountered_eof = false;

    'outer: loop {
        if count >= target {
            break;
        }
        let mut remaining = element_size;
        while remaining > 0 {
            match reader.read(&mut buffer[element_size - remaining..element_size]) {
                Ok(0) => {
                    encountered_eof = true;
                    break 'outer;
                }
                Ok(n) => remaining -= n,
                Err(err) if err.kind() == ErrorKind::Interrupted => continue,
                Err(err) => {
                    return Err(format!("fread: failed to read from file ({err})"));
                }
            }
        }
        if remaining > 0 {
            break;
        }
        let value = decode_numeric_scalar(&buffer, input, endianness)?;
        values.push(value);
        count += 1;
        if skip > 0 && count.is_multiple_of(repeat) {
            reader
                .seek(SeekFrom::Current(skip as i64))
                .map_err(|err| format!("fread: failed to skip bytes ({err})"))?;
        }
    }
    Ok((
        numeric_storage_from_scalars(input, values)?,
        count,
        encountered_eof,
    ))
}

fn read_char_values<R: Read + Seek>(
    reader: &mut R,
    input: InputType,
    repeat: usize,
    limit: Option<usize>,
    skip: usize,
    endianness: Endianness,
) -> Result<(Vec<char>, usize, bool), String> {
    if let Some(0) = limit {
        return Ok((Vec::new(), 0, false));
    }
    let element_size = input.byte_len();
    let mut buffer = vec![0u8; element_size];
    let mut values = Vec::new();
    let mut count = 0usize;
    let target = limit.unwrap_or(usize::MAX);
    let mut encountered_eof = false;

    'outer: loop {
        if count >= target {
            break;
        }
        let mut remaining = element_size;
        while remaining > 0 {
            match reader.read(&mut buffer[element_size - remaining..element_size]) {
                Ok(0) => {
                    encountered_eof = true;
                    break 'outer;
                }
                Ok(n) => remaining -= n,
                Err(err) if err.kind() == ErrorKind::Interrupted => continue,
                Err(err) => {
                    return Err(format!("fread: failed to read from file ({err})"));
                }
            }
        }
        if remaining > 0 {
            break;
        }
        let ch = decode_to_char(&buffer, input, endianness)?;
        values.push(ch);
        count += 1;
        if skip > 0 && count.is_multiple_of(repeat) {
            reader
                .seek(SeekFrom::Current(skip as i64))
                .map_err(|err| format!("fread: failed to skip bytes ({err})"))?;
        }
    }

    Ok((values, count, encountered_eof))
}

fn decode_numeric_scalar(
    bytes: &[u8],
    input: InputType,
    endianness: Endianness,
) -> Result<NumericScalar, String> {
    Ok(match input {
        InputType::UInt8 => NumericScalar::U8(bytes[0]),
        InputType::Int8 => NumericScalar::I8(bytes[0] as i8),
        InputType::UInt16 => NumericScalar::U16(read_u16(bytes, endianness)),
        InputType::Int16 => NumericScalar::I16(read_u16(bytes, endianness) as i16),
        InputType::UInt32 => NumericScalar::U32(read_u32(bytes, endianness)),
        InputType::Int32 => NumericScalar::I32(read_u32(bytes, endianness) as i32),
        InputType::UInt64 => NumericScalar::U64(read_u64(bytes, endianness)),
        InputType::Int64 => NumericScalar::I64(read_u64(bytes, endianness) as i64),
        InputType::Float32 => {
            let bits = read_u32(bytes, endianness);
            NumericScalar::F32(f32::from_bits(bits))
        }
        InputType::Float64 => {
            let bits = read_u64(bytes, endianness);
            NumericScalar::F64(f64::from_bits(bits))
        }
    })
}

fn numeric_storage_from_scalars(
    input: InputType,
    values: Vec<NumericScalar>,
) -> Result<NumericStorage, String> {
    macro_rules! collect_variant {
        ($scalar_variant:ident, $storage_variant:ident) => {{
            let mut output = Vec::with_capacity(values.len());
            for value in values {
                let NumericScalar::$scalar_variant(value) = value else {
                    return Err("fread: decoded numeric source class mismatch".to_string());
                };
                output.push(value);
            }
            NumericStorage::$storage_variant(output)
        }};
    }
    Ok(match input {
        InputType::UInt8 => collect_variant!(U8, U8),
        InputType::Int8 => collect_variant!(I8, I8),
        InputType::UInt16 => collect_variant!(U16, U16),
        InputType::Int16 => collect_variant!(I16, I16),
        InputType::UInt32 => collect_variant!(U32, U32),
        InputType::Int32 => collect_variant!(I32, I32),
        InputType::UInt64 => collect_variant!(U64, U64),
        InputType::Int64 => collect_variant!(I64, I64),
        InputType::Float32 => collect_variant!(F32, F32),
        InputType::Float64 => collect_variant!(F64, F64),
    })
}

fn decode_to_char(bytes: &[u8], input: InputType, endianness: Endianness) -> Result<char, String> {
    let scalar = decode_numeric_scalar(bytes, input, endianness)?;
    let code = match scalar {
        NumericScalar::F64(value) => floating_char_code(value)?,
        NumericScalar::F32(value) => floating_char_code(f64::from(value))?,
        value => value
            .into_int_value()
            .and_then(|value| value.try_to_u64())
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| "fread: character value is outside the Unicode range".to_string())?,
    };
    char::from_u32(code).ok_or_else(|| {
        format!("value 0x{code:X} cannot be represented as a Unicode scalar for char output")
    })
}

fn floating_char_code(value: f64) -> Result<u32, String> {
    if !value.is_finite() || value < 0.0 || value > f64::from(u32::MAX) || value.round() != value {
        return Err("fread: character value is outside the Unicode range".to_string());
    }
    Ok(value as u32)
}

fn read_u16(bytes: &[u8], endianness: Endianness) -> u16 {
    match endianness {
        Endianness::Little => u16::from_le_bytes([bytes[0], bytes[1]]),
        Endianness::Big => u16::from_be_bytes([bytes[0], bytes[1]]),
    }
}

fn read_u32(bytes: &[u8], endianness: Endianness) -> u32 {
    match endianness {
        Endianness::Little => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        Endianness::Big => u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    }
}

fn read_u64(bytes: &[u8], endianness: Endianness) -> u64 {
    match endianness {
        Endianness::Little => u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
        Endianness::Big => u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
    }
}

fn finalize_numeric(
    size_spec: &SizeSpec,
    count_read: usize,
    mut values: NumericStorage,
) -> (NumericStorage, usize, usize) {
    match size_spec {
        SizeSpec::All | SizeSpec::Count(_) => {
            let rows = count_read;
            let cols = 1;
            (values, rows, cols)
        }
        SizeSpec::Matrix {
            rows,
            cols: Some(c),
        } => {
            let target = rows.saturating_mul(*c);
            resize_numeric_storage(&mut values, target);
            (values, *rows, *c)
        }
        SizeSpec::Matrix { rows, cols: None } => {
            if *rows == 0 {
                resize_numeric_storage(&mut values, 0);
                (values, 0, 0)
            } else {
                let cols = if count_read == 0 {
                    0
                } else {
                    count_read.div_ceil(*rows)
                };
                let target = rows.saturating_mul(cols);
                resize_numeric_storage(&mut values, target);
                (values, *rows, cols)
            }
        }
    }
}

fn resize_numeric_storage(storage: &mut NumericStorage, len: usize) {
    macro_rules! resize {
        ($values:expr, $zero:expr) => {{
            $values.resize(len, $zero);
        }};
    }
    match storage {
        NumericStorage::F64(values) => resize!(values, 0.0),
        NumericStorage::F32(values) => resize!(values, 0.0),
        NumericStorage::I8(values) => resize!(values, 0),
        NumericStorage::I16(values) => resize!(values, 0),
        NumericStorage::I32(values) => resize!(values, 0),
        NumericStorage::I64(values) => resize!(values, 0),
        NumericStorage::U8(values) => resize!(values, 0),
        NumericStorage::U16(values) => resize!(values, 0),
        NumericStorage::U32(values) => resize!(values, 0),
        NumericStorage::U64(values) => resize!(values, 0),
    }
}

fn finalize_char(
    size_spec: &SizeSpec,
    count_read: usize,
    mut column_major: Vec<char>,
) -> (Vec<char>, usize, usize) {
    match size_spec {
        SizeSpec::All | SizeSpec::Count(_) => {
            let rows = count_read;
            let cols = 1;
            let row_major = column_to_row_major(&column_major, rows, cols);
            (row_major, rows, cols)
        }
        SizeSpec::Matrix {
            rows,
            cols: Some(c),
        } => {
            let target = rows.saturating_mul(*c);
            if column_major.len() < target {
                column_major.resize(target, '\0');
            } else if column_major.len() > target {
                column_major.truncate(target);
            }
            let row_major = column_to_row_major(&column_major, *rows, *c);
            (row_major, *rows, *c)
        }
        SizeSpec::Matrix { rows, cols: None } => {
            if *rows == 0 {
                column_major.clear();
                (Vec::new(), 0, 0)
            } else {
                let cols = if count_read == 0 {
                    0
                } else {
                    count_read.div_ceil(*rows)
                };
                let target = rows.saturating_mul(cols);
                if column_major.len() < target {
                    column_major.resize(target, '\0');
                } else if column_major.len() > target {
                    column_major.truncate(target);
                }
                let row_major = column_to_row_major(&column_major, *rows, cols);
                (row_major, *rows, cols)
            }
        }
    }
}

fn column_to_row_major(data: &[char], rows: usize, cols: usize) -> Vec<char> {
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    let mut output = vec!['\0'; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            let src = c * rows + r;
            let dst = r * cols + c;
            if let Some(ch) = data.get(src) {
                output[dst] = *ch;
            }
        }
    }
    output
}

#[cfg(test)]
pub(crate) mod tests;
